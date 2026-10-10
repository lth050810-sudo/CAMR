"""Train the released RLEdit editor on independent facts, then edit frozen 3K."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from baselines.common import (AXES, append_jsonl, atomic_json, evaluate_checkpoint,
                              group_by_stage, load_manifest, load_model, load_tokenizer,
                              manifest_panel, sha256_file)
from baselines.editor_dev_selection import check_disjoint
from baselines.adaptation_provenance import implementation_identity
from baselines.rledit_qwen_core import RLEditQwen, UPSTREAM
from baselines.run_ultraedit import get_module, ultraedit_tokenize


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, default=Path('/root/engram/Qwen3-8B-Base'))
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--meta-manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--source-dir', type=Path, default=UPSTREAM)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--edit-batch-size', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--network-batch-size', type=int, default=128)
    p.add_argument('--rank', type=int, default=1024)
    p.add_argument('--n-blocks', type=int, default=4)
    p.add_argument('--edit-lr', type=float, default=1e-6)
    p.add_argument('--meta-lr', type=float, default=1e-5)
    p.add_argument('--meta-epochs', type=int, default=1)
    p.add_argument('--back-depth', type=int, default=10)
    p.add_argument('--time-decay', type=float, default=0.95)
    p.add_argument('--locality-coefficient', type=float, default=0.6)
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--max-new-tokens', type=int, default=16)
    p.add_argument('--eval-batch-size', type=int, default=32)
    p.add_argument('--max-meta-facts', type=int, default=0)
    p.add_argument('--max-test-facts', type=int, default=0)
    p.add_argument('--memory-fraction', type=float, default=0.92)
    p.add_argument('--no-stage-eval', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if min(args.batch_size, args.edit_batch_size, args.meta_epochs, args.back_depth) < 1:
        p.error('Positive training budgets required')
    return args


def network_digest(net):
    digest = hashlib.sha256()
    for name, parameter in net.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.pt')
    torch.save(payload, temporary)
    temporary.replace(path)


def meta_examples(rows, tokenizer, max_length):
    def make(question, answer):
        return ultraedit_tokenize(tokenizer, question, answer, max_length)
    return {'edit': [make(r['update'], r['ans']) for r in rows],
            'equiv': [make(r['rephrase'], r['ans']) for r in rows],
            'locality': [make(r['loc'], r['loc_ans']) for r in rows
                         if r.get('loc') and r.get('loc_ans')]}


def main():
    a = parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()) and not a.resume:
        raise FileExistsError('Refusing to overwrite a previous run')
    check_disjoint(a.meta_manifest, a.manifest)
    rows, manifest = load_manifest(a.manifest)
    meta_rows, _ = load_manifest(a.meta_manifest)
    if a.max_meta_facts:
        meta_rows = meta_rows[:a.max_meta_facts]
    if a.max_test_facts:
        rows = rows[:a.max_test_facts]
    locality = manifest_panel(manifest, 'locality', limit=512)
    multihop = manifest_panel(manifest, 'mhop', limit=512)
    if a.max_test_facts:
        locality, multihop = locality[:2], multihop[:2]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    a.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction, 0)
    device = torch.device('cuda:0')
    tokenizer = load_tokenizer(a.model_dir)
    model = load_model(a.model_dir, device)
    model.config.use_cache = False
    # Transfer the released active Llama-3 32-layer preset by block-center depth.
    depth = model.config.num_hidden_layers
    layers = [round((layer + 0.5) * depth / 32 - 0.5) for layer in range(22, 31)]
    names = [f'model.layers.{layer}.mlp.{projection}' for projection in ('gate_proj', 'up_proj')
             for layer in layers]
    editor = RLEditQwen(model, names, rank=a.rank, n_blocks=a.n_blocks,
                       edit_lr=a.edit_lr, meta_lr=a.meta_lr,
                       network_batch_size=a.network_batch_size, source=a.source_dir)
    source = json.loads((a.source_dir / 'SOURCE.json').read_text())
    for name, expected in source['file_sha256'].items():
        if sha256_file(a.source_dir / name) != expected:
            raise ValueError('Pinned RLEdit source changed: ' + name)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()}
    config.update(method='RLEdit-Qwen3-Base-transfer', num_facts=len(rows),
                  implementation=implementation_identity('rledit'),
                  meta_num_facts=len(meta_rows), model_layers=depth, edit_modules=names,
                  manifest_sha256=sha256_file(a.manifest), meta_manifest_sha256=sha256_file(a.meta_manifest),
                  source=source, new_fact_epochs=1, editor_training_epochs=a.meta_epochs,
                  training_fields=['update', 'ans'], closed_book_evaluation=True,
                  meta_training_fields=['update', 'rephrase', 'ans', 'loc', 'loc_ans'],
                  meta_test_subject_fact_question_disjoint=True,
                  initial_hypernetwork_sha256=network_digest(editor.net),
                  hypernetwork_parameters=sum(p.numel() for p in editor.net.parameters()),
                  llm_gradient_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  direct_llm_optimizer_steps=0,
                  adaptations=['Qwen3-8B-Base completion QA template and answer EOS.',
                               'Independent dev1600 editor training; same 3K editing stream as other methods.',
                               'Relative-depth transfer of released Llama gate/up layer preset; not test-tuned.',
                               'Equivalent dual ridge and adjoint; immediate microbatch backward to bound memory.',
                               'Exact restoration of temporary locality branches and of base before final editing.',
                               'Upstream detached L2 term has no gradient; no extra regularizer is introduced.',
                               'Checkpointed editor, model edits and generated closed-book evaluation.'])
    meta_path = a.checkpoint_dir / 'trained_editor.pt'
    state_path = a.checkpoint_dir / 'latest_edit_state.pt'
    started = time.time()
    meta_logs = []
    if a.resume:
        saved_config = json.loads((a.output_dir / 'run_config.json').read_text())
        for key in ('manifest_sha256', 'meta_manifest_sha256', 'edit_modules', 'edit_batch_size',
                    'meta_epochs', 'rank', 'n_blocks', 'seed', 'max_length', 'max_new_tokens',
                    'max_meta_facts', 'max_test_facts', 'batch_size', 'network_batch_size',
                    'back_depth', 'time_decay', 'locality_coefficient', 'edit_lr', 'meta_lr',
                    'model_dir', 'source', 'implementation', 'no_stage_eval'):
            if config[key] != saved_config[key]:
                raise ValueError('Resume configuration changed: ' + key)
        config = saved_config
    else:
        atomic_json(a.output_dir / 'run_config.json', config)
    if a.resume and meta_path.exists():
        checkpoint = torch.load(meta_path, map_location='cpu', weights_only=False)
        editor.net.load_state_dict(checkpoint['net'])
        editor.optimizer.load_state_dict(checkpoint['optimizer'])
        editor.meta_optimizer_steps = checkpoint['meta_optimizer_steps']
        editor.forward_counts = Counter(checkpoint['forward_counts'])
        meta_logs = checkpoint['meta_logs']
        elapsed_before = checkpoint['elapsed_seconds']
    else:
        elapsed_before = 0.0
        for epoch in range(a.meta_epochs):
            history = []
            for start in range(0, len(meta_rows), a.edit_batch_size):
                part = meta_rows[start:start + a.edit_batch_size]
                current = meta_examples(part, tokenizer, a.max_length)
                history.append(current)
                log = editor.meta_edit(current, history, batch_size=a.batch_size,
                                       pad_token_id=tokenizer.pad_token_id, time_decay=a.time_decay,
                                       back_depth=a.back_depth, locality_coefficient=a.locality_coefficient)
                log.update(epoch=epoch + 1, completed_meta_facts=start + len(part),
                           sample_ids=[r['sample_id'] for r in part])
                meta_logs.append(log)
                atomic_json(a.output_dir / 'meta_training_log.json', meta_logs)
                atomic_json(a.output_dir / 'progress.json',
                            {'status': 'meta_training', 'epoch': epoch + 1,
                             'meta_facts': start + len(part), 'expected_meta_facts': len(meta_rows),
                             'meta_optimizer_steps': editor.meta_optimizer_steps})
                print(json.dumps({k: v for k, v in log.items() if k != 'sample_ids'}), flush=True)
            editor.finish_meta_sequence()
            editor.reset_language_model()
        trained_digest = network_digest(editor.net)
        if trained_digest == config['initial_hypernetwork_sha256']:
            raise RuntimeError('Editor training did not change the hypernetwork')
        save_checkpoint(meta_path, {'net': editor.net.state_dict(), 'optimizer': editor.optimizer.state_dict(),
                                   'meta_logs': meta_logs, 'meta_optimizer_steps': editor.meta_optimizer_steps,
                                   'forward_counts': dict(editor.forward_counts),
                                   'elapsed_seconds': time.time() - started,
                                   'trained_hypernetwork_sha256': trained_digest, 'config': config})
    # A trained editor must be reused; the 3K edit phase starts from untouched base weights.
    editor.reset_language_model()
    base_restored = all(torch.equal(get_module(model, name).weight.detach().cpu(), original)
                        for name, original in editor.original.items())
    if not base_restored:
        raise RuntimeError('Base weights were not restored before final editing')
    for p in editor.net.parameters():
        p.requires_grad_(False)
    atomic_json(a.output_dir / 'progress.json', {'status': 'editor_trained',
                'meta_facts': len(meta_rows), 'meta_optimizer_steps': editor.meta_optimizer_steps,
                'base_restored_before_3k': base_restored})
    actions, metrics, cursor = [], [], 0
    if a.resume and state_path.exists():
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        editor.restore(state['edited_weights'])
        editor.net.load_state_dict(state['net'])
        actions, metrics, cursor = state['actions'], state['metrics'], state['cursor']
        editor.forward_counts = Counter(state['forward_counts'])
        elapsed_before = state['elapsed_seconds']
        (a.output_dir / 'actions.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in actions))
        (a.output_dir / 'predictions.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in state['predictions']))
        predictions = state['predictions']
    else:
        predictions = []
        if a.resume:
            # Roll back any uncommitted first-stage artifacts along with the model.
            (a.output_dir / 'actions.jsonl').write_text('')
            (a.output_dir / 'predictions.jsonl').write_text('')
            atomic_json(a.output_dir / 'metrics.json', [])
    processed = 0
    for stage, stage_rows in group_by_stage(rows).items():
        for start in range(0, len(stage_rows), a.edit_batch_size):
            part = stage_rows[start:start + a.edit_batch_size]
            processed += len(part)
            if processed <= cursor:
                continue
            examples = [ultraedit_tokenize(tokenizer, r['update'], r['ans'], a.max_length) for r in part]
            log = editor.edit(examples, batch_size=a.batch_size, pad_token_id=tokenizer.pad_token_id)
            log.update(stage=stage, event=len(actions), sample_ids=[r['sample_id'] for r in part],
                       facts_completed=processed, new_fact_count=len(part), history_replay_count=0)
            actions.append(log)
            append_jsonl(a.output_dir / 'actions.jsonl', [log])
            atomic_json(a.output_dir / 'progress.json', {'status': 'editing', 'stage': stage,
                        'facts_completed': processed, 'num_facts': len(rows), 'edit_batches': len(actions)})
            print(json.dumps(log), flush=True)
        if processed <= cursor:
            continue
        if not a.no_stage_eval or processed == len(rows):
            result, records = evaluate_checkpoint(model, tokenizer, rows[:processed], device,
                checkpoint=f'tau{stage}', axes=AXES, batch_size=a.eval_batch_size,
                max_length=a.max_length, max_new_tokens=a.max_new_tokens,
                locality_panel=locality, multihop_panel=multihop)
            result.update(current_stage=stage)
            for axis in AXES:
                result[axis + '_accuracy'] = result[axis]['match_accuracy']
            metrics.append(result)
            predictions.extend(records)
            append_jsonl(a.output_dir / 'predictions.jsonl', records)
            atomic_json(a.output_dir / 'metrics.json', metrics)
        save_checkpoint(state_path, {'cursor': processed, 'net': editor.net.state_dict(),
                        'edited_weights': {n: get_module(model, n).weight.detach().cpu() for n in names},
                        'forward_counts': dict(editor.forward_counts),
                        'elapsed_seconds': elapsed_before + time.time() - started,
                        'actions': actions, 'metrics': metrics, 'predictions': predictions, 'config': config})
    ids = [sample_id for action in actions for sample_id in action['sample_ids']]
    if ids != [r['sample_id'] for r in rows] or len(set(ids)) != len(rows):
        raise RuntimeError('Final edit stream is not a single pass')
    done = {'status': 'complete', 'method': config['method'], 'num_facts': len(rows),
            'new_fact_epochs': 1, 'edit_batches': len(actions), 'meta_num_facts': len(meta_rows),
            'meta_epochs': a.meta_epochs, 'meta_optimizer_steps': editor.meta_optimizer_steps,
            'editor_checkpoint': str(meta_path), 'edit_checkpoint': str(state_path),
            'trained_hypernetwork_sha256': network_digest(editor.net),
            'initial_hypernetwork_sha256': config['initial_hypernetwork_sha256'],
            'base_restored_before_3k': base_restored, 'forward_counts': dict(editor.forward_counts),
            'peak_vram_gib': torch.cuda.max_memory_allocated() / 2**30,
            'smoke_only': bool(a.max_meta_facts or a.max_test_facts),
            'elapsed_seconds': elapsed_before + time.time() - started}
    atomic_json(a.output_dir / 'completed.json', done)
    print(json.dumps(done), flush=True)


if __name__ == '__main__':
    main()
