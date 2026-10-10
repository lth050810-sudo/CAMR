"""Evaluate the final longstream LoRA adapter on unseen WikiBigEdit facts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

from baselines.common import answer_scores, qa_prompt
from baselines.longstream_protocol import atomic, norm, slot
from baselines.wikibigedit_protocol import (
    TokenLengthFilter, _basic_invalid_reason, _fact_fingerprint,
    _load_source_descriptors, _materialize_record, stable_sample_id,
)


AXES = {'update': ('update', 'ans'), 'rephrase': ('rephrase', 'ans'),
        'persona': ('personas', 'ans')}


def holdout_rows(training, data_dir, model_dir, per_stage_tag, seed):
    from transformers import AutoTokenizer

    source, descriptors = _load_source_descriptors(data_dir, verify_hashes=True)
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    lengths = TokenLengthFilter(tok, 128, 16)
    used_ids = set(training['selection']['sample_ids'])
    used_facts = {r['fact_fingerprint'] for r in training['records']}
    used_qa = {(norm(r['update']), norm(r['ans'])) for r in training['records']}
    used_slots = {slot(r) for r in training['records']}
    selected, seen_facts, seen_qa, seen_slots = [], set(), set(), set()
    source_paths = []
    for desc in descriptors:
        path = data_dir / desc['path']
        source_paths.append(path)
        raw = json.loads(path.read_text())
        by_tag = defaultdict(list)
        for index, row in enumerate(raw):
            if _basic_invalid_reason(row, {'new', 'update'}):
                continue
            sid = stable_sample_id(desc['interval'], index)
            fp = _fact_fingerprint(row)
            qa = (norm(row['update']), norm(row['ans']))
            key = slot(row)
            if (sid in used_ids or fp in used_facts or qa in used_qa or key in used_slots
                    or any(answer_scores(row[q], row[a])['match'] for q, a in AXES.values()
                           if isinstance(row.get(q), str) and isinstance(row.get(a), str))):
                continue
            rank = hashlib.sha256(f'{seed}:{sid}'.encode()).hexdigest()
            by_tag[row['tag']].append((rank, index, row, fp, qa, key))
        for tag in ('new', 'update'):
            candidates = sorted(by_tag[tag])[:max(per_stage_tag * 4, 256)]
            for offset in range(0, len(candidates), 128):
                batch = candidates[offset:offset + 128]
                audits = lengths.audit([item[2] for item in batch])
                for (_, index, row, fp, qa, key), audit in zip(batch, audits):
                    if not audit['core_valid'] or fp in seen_facts or qa in seen_qa or key in seen_slots:
                        continue
                    if any(not audit['panels'][axis]['available'] for axis in ('update', 'rephrase', 'personas')):
                        continue
                    rec = _materialize_record(row, desc, index, index + 1, audit)
                    selected.append(rec)
                    seen_facts.add(fp)
                    seen_qa.add(qa)
                    seen_slots.add(key)
                    if sum(r['stage'] == desc['stage'] and r['tag'] == tag for r in selected) >= per_stage_tag * 4:
                        break
                if sum(r['stage'] == desc['stage'] and r['tag'] == tag for r in selected) >= per_stage_tag * 4:
                    break
            if sum(r['stage'] == desc['stage'] and r['tag'] == tag for r in selected) < per_stage_tag:
                raise ValueError(f'Not enough disjoint held-out rows in stage {desc["stage"]}, tag {tag}')
    # Guard against a question with multiple answers elsewhere in the full source.
    questions = {axis: {norm(r[q]) for r in selected} for axis, (q, _) in AXES.items()}
    answer_sets = {axis: defaultdict(set) for axis in AXES}
    for path in source_paths:
        for row in json.loads(path.read_text()):
            for axis, (q, a) in AXES.items():
                question = norm(row.get(q))
                if question in questions[axis]:
                    answer_sets[axis][question].add(norm(row.get(a)))
    ambiguous = {axis: {q for q, answers in mapping.items() if len(answers) > 1}
                 for axis, mapping in answer_sets.items()}
    unambiguous = [r for r in selected if all(norm(r[q]) not in ambiguous[axis]
                                           for axis, (q, _) in AXES.items())]
    selected = []
    for stage in range(1, 9):
        for tag in ('new', 'update'):
            group = [r for r in unambiguous if r['stage'] == stage and r['tag'] == tag]
            if len(group) < per_stage_tag:
                raise ValueError(f'Only {len(group)} unambiguous rows in stage {stage}, tag {tag}')
            selected.extend(group[:per_stage_tag])
    return {'schema_version': 'wikibigedit-disjoint-holdout/v1', 'source': source,
            'training_manifest_sha256': hashlib.sha256(json.dumps(training, sort_keys=True,
                ensure_ascii=False).encode()).hexdigest(),
            'selection': {'seed': seed, 'per_stage_tag_target': per_stage_tag,
                          'excludes_training_ids_facts_qa_slots': True,
                          'excludes_question_answer_leakage': True,
                          'excludes_source_ambiguous_questions': True,
                          'counts_by_stage_tag': {f'{s}:{t}': sum(r['stage'] == s and r['tag'] == t for r in selected)
                                                  for s in range(1, 9) for t in ('new', 'update')}},
            'records': selected}


def load_adapter(checkpoint, model):
    import torch
    import torch.distributed as dist
    from torch.distributed.checkpoint import FileSystemReader, load

    path = checkpoint / 'distributed'
    metadata = json.loads((checkpoint / 'checkpoint_meta.json').read_text())
    if not metadata.get('complete') or metadata.get('parameter_mode') != 'lora':
        raise ValueError('Expected a complete LoRA checkpoint')
    reader = FileSystemReader(str(path))
    names = [name for name in reader.read_metadata().state_dict_metadata
             if name.startswith('model.') and '.lora_' in name]
    expected = {name for name in model.state_dict() if '.lora_' in name}
    checkpoint_names = {name.removeprefix('model.') for name in names}
    if checkpoint_names != expected:
        raise ValueError(f'Adapter tensor names differ from model: '
                         f'checkpoint-only={sorted(checkpoint_names - expected)[:3]}, '
                         f'model-only={sorted(expected - checkpoint_names)[:3]}')
    state = {'model': {name.removeprefix('model.'): torch.empty_like(model.state_dict()[name.removeprefix('model.')],
                                                       device='cpu', dtype=torch.float32)
                       for name in names}}
    dist.init_process_group('gloo', init_method='tcp://127.0.0.1:29742', rank=0, world_size=1)
    try:
        load(state, checkpoint_id=str(path))
    finally:
        dist.destroy_process_group()
    loaded = model.load_state_dict(state['model'], strict=False)
    if loaded.unexpected_keys or set(loaded.missing_keys) != set(model.state_dict()) - expected:
        raise ValueError('Adapter load did not match expected model tensors')
    return len(names)


def generate(model, tokenizer, questions, device, batch_size):
    import torch

    answers = []
    tokenizer.padding_side = 'left'
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    with torch.inference_mode():
        for start in range(0, len(questions), batch_size):
            batch = questions[start:start + batch_size]
            encoded = tokenizer([qa_prompt(q) for q in batch], return_tensors='pt', padding=True,
                                truncation=True, max_length=128, add_special_tokens=True)
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            finished = torch.zeros(len(batch), dtype=torch.bool, device=device)
            output = []
            for _ in range(16):
                positions = attention_mask.long().cumsum(dim=-1) - 1
                positions.masked_fill_(attention_mask == 0, 0)
                logits = model(input_ids=input_ids, attention_mask=attention_mask,
                               position_ids=positions, use_cache=False).logits
                next_tokens = logits[:, -1].argmax(dim=-1)
                active = ~finished
                next_tokens = torch.where(active, next_tokens, torch.full_like(next_tokens, pad))
                output.append(next_tokens)
                finished |= next_tokens == eos
                if bool(finished.all()):
                    break
                input_ids = torch.cat((input_ids, next_tokens[:, None]), dim=1)
                attention_mask = torch.cat((attention_mask, active.long()[:, None]), dim=1)
            answers.extend(tokenizer.batch_decode(torch.stack(output, dim=1), skip_special_tokens=True))
            print(json.dumps({'generated': len(answers), 'total': len(questions)}), flush=True)
    return answers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-manifest', type=Path, required=True)
    p.add_argument('--source-dir', type=Path, default=Path('/root/engram/WikiBigEdit/full_hf_viewer'))
    p.add_argument('--model-dir', type=Path, default=Path('/root/engram/Qwen3-8B-Base'))
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--per-stage-tag', type=int, default=64)
    p.add_argument('--seed', type=int, default=20260926)
    p.add_argument('--device', default='cuda:7')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--reference-predictions', type=Path)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    if a.per_stage_tag < 1 or a.batch_size < 1:
        p.error('Panel and batch sizes must be positive')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    holdout_path = a.output_dir / 'holdout.json'
    if holdout_path.exists():
        holdout = json.loads(holdout_path.read_text())
    else:
        training = json.loads(a.training_manifest.read_text())
        holdout = holdout_rows(training, a.source_dir, a.model_dir, a.per_stage_tag, a.seed)
        atomic(holdout_path, holdout)
    rows = holdout['records']
    print(json.dumps({'holdout_rows': len(rows), 'counts': holdout['selection']['counts_by_stage_tag']}), flush=True)
    if a.prepare_only:
        return

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.cuda.set_device(torch.device(a.device))
    tokenizer = AutoTokenizer.from_pretrained(a.model_dir, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(a.model_dir, torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', local_files_only=True).to(a.device)
    model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', inference_mode=False,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], bias='none'))
    adapter_count = load_adapter(a.checkpoint, model)
    model.eval()
    started = time.monotonic()
    reference_agreement = None
    if a.reference_predictions:
        reference = []
        with a.reference_predictions.open() as stream:
            for line in stream:
                item = json.loads(line)
                if item['checkpoint'] == 'prefix100000' and item['axis'] == 'update':
                    reference.append(item)
                    if len(reference) == 16:
                        break
        if len(reference) != 16:
            raise ValueError('Reference predictions lack 16 final-checkpoint update probes')
        regenerated = generate(model, tokenizer, [r['question'] for r in reference], a.device, a.batch_size)
        reference_agreement = sum(answer == row['prediction']
                                  for answer, row in zip(regenerated, reference))
        print(json.dumps({'reference_exact_generation_agreement': reference_agreement,
                          'reference_count': len(reference)}), flush=True)
        if reference_agreement < 14:
            raise ValueError('Single-GPU adapter inference does not reproduce the original checkpoint')
    predictions = []
    for axis, (q, gold) in AXES.items():
        answers = generate(model, tokenizer, [r[q] for r in rows], a.device, a.batch_size)
        for row, answer in zip(rows, answers):
            predictions.append({'axis': axis, 'sample_id': row['sample_id'], 'stage': row['stage'],
                                'tag': row['tag'], 'question': row[q], 'gold': row[gold],
                                'prediction': answer, **answer_scores(answer, row[gold])})
    atomic(a.output_dir / 'predictions.json', predictions)
    summary = {'checkpoint': str(a.checkpoint), 'events': 100000, 'new_test_rows': len(rows),
               'adapter_tensors_loaded': adapter_count, 'reference_agreement_16': reference_agreement,
               'eval_seconds': time.monotonic() - started,
               'axes': {axis: {'count': len(items),
                               'match_accuracy': sum(x['match'] for x in items) / len(items),
                               'exact_accuracy': sum(x['exact'] for x in items) / len(items),
                               'by_stage': {str(s): {'count': sum(x['stage'] == s for x in items),
                                                    'match_accuracy': sum(x['match'] for x in items if x['stage'] == s)
                                                    / sum(x['stage'] == s for x in items)} for s in range(1, 9)}}
                        for axis in AXES for items in [[x for x in predictions if x['axis'] == axis]]}}
    atomic(a.output_dir / 'results.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
