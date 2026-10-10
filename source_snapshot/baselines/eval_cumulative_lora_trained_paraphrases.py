"""Test final cumulative LoRA on new paraphrases of already trained facts."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import time

from baselines.common import answer_scores
from baselines.eval_cumulative_lora_heldout import generate, load_adapter
from baselines.longstream_protocol import atomic, norm
from baselines.engram_training_views import text_sha256


def select_panel(manifest, views, per_stage_new, per_stage_update, seed):
    records = {row['sample_id']: row for row in manifest['records']}
    eligible = (set(manifest['primary_evaluation_ids']['update'])
                & set(manifest['primary_evaluation_ids']['rephrase'])
                & set(manifest['primary_evaluation_ids']['persona']))
    blocked = set(manifest['audit']['unresolved_slot_ids'])
    official_questions = {norm(row[field]) for row in manifest['records']
                          for field in ('update', 'rephrase', 'personas') if row.get(field)}
    teacher_answers = defaultdict(set)
    for sid, item in views['rows'].items():
        if item['accepted'] and item['view_source'] == 'teacher_constrained' and sid in records:
            teacher_answers[norm(item['question'])].add(norm(records[sid]['ans']))
    pools = defaultdict(list)
    for sid, item in views['rows'].items():
        row = records.get(sid)
        if (row is None or sid not in eligible or sid in blocked or not item['accepted']
                or item['view_source'] != 'teacher_constrained'):
            continue
        question = item['question']
        if (item['update_sha256'] != text_sha256(row['update'])
                or item['answer_sha256'] != text_sha256(row['ans'])):
            raise ValueError(f'Teacher view does not match training fact: {sid}')
        if (norm(question) in official_questions or len(teacher_answers[norm(question)]) != 1
                or answer_scores(question, row['ans'])['match']):
            continue
        rank = hashlib.sha256(f'{seed}:{sid}'.encode()).hexdigest()
        pools[(int(row['stage']), row['tag'])].append((rank, sid))
    selected = []
    for stage in range(1, 9):
        for tag, target in (('new', per_stage_new), ('update', per_stage_update)):
            chosen = []
            relation_counts = Counter()
            for _, sid in sorted(pools[(stage, tag)]):
                relation = records[sid].get('relation_id') or records[sid].get('relation')
                if relation_counts[relation] >= 8:
                    continue
                chosen.append(sid)
                relation_counts[relation] += 1
                if len(chosen) == target:
                    break
            if len(chosen) != target:
                raise ValueError(f'Only {len(chosen)}/{target} samples in stage {stage}, tag {tag}; '
                                 f'candidate pool {len(pools[(stage, tag)])}')
            selected.extend(chosen)
    return {'schema_version': 'longstream-trained-new-paraphrases/v1',
            'training_manifest_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True,
                ensure_ascii=False).encode()).hexdigest(),
            'views_sha256': hashlib.sha256(json.dumps(views, sort_keys=True,
                ensure_ascii=False).encode()).hexdigest(),
            'seed': seed, 'selection': 'trained facts; accepted teacher_constrained question; not original/official question; no answer in question; unambiguous; relation cap 8',
            'per_stage_new': per_stage_new, 'per_stage_update': per_stage_update,
            'sample_ids': selected,
            'rows': [{'sample_id': sid, 'stage': records[sid]['stage'], 'tag': records[sid]['tag'],
                      'relation_id': records[sid].get('relation_id'),
                      'original': records[sid]['update'], 'official_rephrase': records[sid]['rephrase'],
                      'persona_question': records[sid]['personas'],
                      'new_rephrase': views['rows'][sid]['question'], 'gold': records[sid]['ans']}
                     for sid in selected]}


def read_existing(path, ids):
    seen = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['checkpoint'] != 'prefix100000' or row['sample_id'] not in ids:
                continue
            if row['axis'] not in ('update', 'rephrase', 'persona'):
                continue
            key = (row['sample_id'], row['axis'])
            if key in seen:
                raise ValueError(f'Duplicate original prediction: {key}')
            seen[key] = row
    if len(seen) != 3 * len(ids):
        raise ValueError(f'Expected {3 * len(ids)} original/rephrase/persona predictions; got {len(seen)}')
    return seen


def summarize(rows, field):
    return {'n': len(rows), 'match': sum(r[field]['match'] for r in rows) / len(rows),
            'exact': sum(r[field]['exact'] for r in rows) / len(rows),
            'by_tag': {tag: {'n': sum(r['tag'] == tag for r in rows),
                             'match': sum(r[field]['match'] for r in rows if r['tag'] == tag)
                                      / sum(r['tag'] == tag for r in rows)} for tag in ('new', 'update')},
            'by_stage': {str(stage): {'n': sum(r['stage'] == stage for r in rows),
                                     'match': sum(r[field]['match'] for r in rows if r['stage'] == stage)
                                              / sum(r['stage'] == stage for r in rows)} for stage in range(1, 9)}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-manifest', type=Path, required=True)
    p.add_argument('--training-views', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--existing-predictions', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, default=Path('/root/engram/Qwen3-8B-Base'))
    p.add_argument('--per-stage-new', type=int, default=32)
    p.add_argument('--per-stage-update', type=int, default=16)
    p.add_argument('--seed', type=int, default=20260926)
    p.add_argument('--device', default='cuda:7')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--rescore-only', action='store_true')
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = a.output_dir / 'panel.json'
    if panel_path.exists():
        panel = json.loads(panel_path.read_text())
    else:
        manifest = json.loads(a.training_manifest.read_text())
        views = json.loads(a.training_views.read_text())
        panel = select_panel(manifest, views, a.per_stage_new, a.per_stage_update, a.seed)
        atomic(panel_path, panel)
    rows = panel['rows']
    print(json.dumps({'trained_facts': len(rows), 'selection': panel['selection']}), flush=True)
    if a.prepare_only:
        return
    old = read_existing(a.existing_predictions, set(panel['sample_ids']))
    if a.rescore_only:
        rows = json.loads((a.output_dir / 'predictions.json').read_text())
        manifest_rows = {r['sample_id']: r for r in json.loads(a.training_manifest.read_text())['records']}
        for row in rows:
            row['persona_question'] = manifest_rows[row['sample_id']]['personas']
    for row in rows:
        for axis, field in (('update', 'original'), ('rephrase', 'official_rephrase'),
                            ('persona', 'persona_question')):
            item = old[(row['sample_id'], axis)]
            if (item['question'], item['gold']) != (row[field], row['gold']):
                raise ValueError(f'Existing prediction differs: {row["sample_id"]} {axis}')
            row[axis] = {key: item[key] for key in ('prediction', 'match', 'exact')}

    loaded, elapsed = 504, 0.0
    if not a.rescore_only:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.cuda.set_device(torch.device(a.device))
        tokenizer = AutoTokenizer.from_pretrained(a.model_dir, local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.bfloat16,
            attn_implementation='flash_attention_2', local_files_only=True).to(a.device)
        model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', inference_mode=False,
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], bias='none'))
        loaded = load_adapter(a.checkpoint, model)
        model.eval()
        started = time.monotonic()
        answers = generate(model, tokenizer, [r['new_rephrase'] for r in rows], a.device, a.batch_size)
        for row, answer in zip(rows, answers):
            row['new'] = {'prediction': answer, **answer_scores(answer, row['gold'])}
        elapsed = time.monotonic() - started
    else:
        elapsed = json.loads((a.output_dir / 'results.json').read_text())['eval_seconds']
    results = {'checkpoint': str(a.checkpoint), 'events': 100000, 'adapter_tensors_loaded': loaded,
               'trained_facts': len(rows), 'eval_seconds': elapsed,
               'original': summarize(rows, 'update'), 'official_rephrase': summarize(rows, 'rephrase'),
               'new_rephrase': summarize(rows, 'new'), 'persona': summarize(rows, 'persona')}
    atomic(a.output_dir / 'predictions.json', rows)
    atomic(a.output_dir / 'results.json', results)
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
