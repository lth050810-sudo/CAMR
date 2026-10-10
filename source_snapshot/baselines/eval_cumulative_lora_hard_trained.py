"""Data-defined hard panel of trained facts for final cumulative LoRA."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from baselines.common import answer_scores
from baselines.eval_cumulative_lora_heldout import generate, load_adapter
from baselines.longstream_protocol import atomic


AXES = {'Original': ('update', 'ans', 'update'),
        'Rephrase': ('rephrase', 'ans', 'rephrase'),
        'Persona': ('personas', 'ans', 'persona'),
        'Locality': ('loc', 'loc_ans', 'locality')}


def select_panel(manifest, min_answer_words=3, max_relation_frequency=25):
    records = manifest['records']
    frequencies = Counter(row['relation_id'] for row in records)
    allowed = set.intersection(*(set(manifest['primary_evaluation_ids'][axis])
                                 for axis in ('update', 'rephrase', 'persona', 'locality')))
    selected = []
    for row in records:
        if (row['sample_id'] not in allowed or frequencies[row['relation_id']] > max_relation_frequency
                or len(row['ans'].split()) < min_answer_words
                or any(answer_scores(row[question], row['ans'])['match']
                       for question in ('update', 'rephrase', 'personas'))):
            continue
        selected.append({key: row[key] for key in ('sample_id', 'stage', 'tag', 'relation_id',
                                                  'update', 'rephrase', 'personas', 'loc',
                                                  'ans', 'loc_ans')})
    if len(selected) < 100 or len({row['sample_id'] for row in selected}) != len(selected):
        raise ValueError('Hard panel is too small or contains duplicate IDs')
    return {'schema_version': 'longstream-hard-trained-panel/v1',
            'training_manifest_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True,
                ensure_ascii=False).encode()).hexdigest(),
            'selection': (f'trained 100K facts; all four axes eligible; relation frequency '
                          f'<={max_relation_frequency}; answer >={min_answer_words} whitespace words; '
                          'direct questions do not contain gold; no model-output selection'),
            'records': selected}


def read_existing(path, ids):
    seen = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if (row['checkpoint'] != 'prefix100000' or row['sample_id'] not in ids
                    or row['axis'] not in ('update', 'rephrase', 'persona')):
                continue
            key = (row['sample_id'], row['axis'])
            if key in seen:
                raise ValueError(f'Duplicate prediction: {key}')
            seen[key] = row
    if len(seen) != 3 * len(ids):
        raise ValueError(f'Missing existing direct-axis predictions: {len(seen)}/{3 * len(ids)}')
    return seen


def summary(items):
    n = len(items)
    return {'count': n, 'match_accuracy': sum(row['match'] for row in items) / n,
            'exact_accuracy': sum(row['exact'] for row in items) / n,
            'by_stage': {str(stage): {'count': len(group),
                                     'match_accuracy': sum(row['match'] for row in group) / len(group)}
                         for stage in range(1, 9)
                         if (group := [row for row in items if row['stage'] == stage])}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--existing-predictions', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, default=Path('/root/engram/Qwen3-8B-Base'))
    p.add_argument('--device', default='cuda:7')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--min-answer-words', type=int, default=3)
    p.add_argument('--max-relation-frequency', type=int, default=25)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = a.output_dir / 'panel.json'
    if panel_path.exists():
        panel = json.loads(panel_path.read_text())
    else:
        panel = select_panel(json.loads(a.manifest.read_text()), a.min_answer_words,
                             a.max_relation_frequency)
        atomic(panel_path, panel)
    rows = panel['records']
    print(json.dumps({'trained_hard_facts': len(rows)}), flush=True)
    if a.prepare_only:
        return

    old = read_existing(a.existing_predictions, {row['sample_id'] for row in rows})
    predictions = []
    for label, (q, gold, axis) in AXES.items():
        if label == 'Locality':
            continue
        for row in rows:
            item = old[(row['sample_id'], axis)]
            if (item['question'], item['gold']) != (row[q], row[gold]):
                raise ValueError(f'Existing prediction disagrees with panel: {row["sample_id"]} {axis}')
            predictions.append({'axis': label, 'sample_id': row['sample_id'], 'stage': row['stage'],
                                'tag': row['tag'], 'gold': row[gold], 'question': row[q],
                                'prediction': item['prediction'], 'match': item['match'],
                                'exact': item['exact']})

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
    answers = generate(model, tokenizer, [row['loc'] for row in rows], a.device, a.batch_size)
    for row, answer in zip(rows, answers):
        predictions.append({'axis': 'Locality', 'sample_id': row['sample_id'],
                            'stage': row['stage'], 'tag': row['tag'], 'gold': row['loc_ans'],
                            'question': row['loc'], 'prediction': answer,
                            **answer_scores(answer, row['loc_ans'])})
    result = {'checkpoint': str(a.checkpoint), 'events': 100000,
              'trained_facts': len(rows), 'adapter_tensors_loaded': loaded,
              'selection': panel['selection'],
              'axes': {axis: summary([item for item in predictions if item['axis'] == axis]) for axis in AXES}}
    atomic(a.output_dir / 'predictions.json', predictions)
    atomic(a.output_dir / 'results.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
