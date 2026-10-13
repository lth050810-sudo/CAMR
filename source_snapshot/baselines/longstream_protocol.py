"""Frozen chronological WikiBigEdit prefixes and shared evaluation/cost helpers.

Training receives only original QA and structural metadata. Ambiguous value
changes without an authoritative predecessor are audited, never automatically
called forgetting. No base-unknown claim is made without a base screen.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

PREFIXES = (3000, 10000, 50000, 100000)
AXIS_FIELDS = {'update': ('update', 'ans'), 'rephrase': ('rephrase', 'ans'),
               'persona': ('personas', 'ans'), 'locality': ('loc', 'loc_ans'),
               'multihop': ('mhop', 'mhop_ans')}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    tmp.replace(path)


def norm(value):
    import unicodedata
    return ' '.join(unicodedata.normalize('NFC', str(value or '')).casefold().split())


def slot(row):
    return (row.get('subject_id') or norm(row.get('subject')),
            row.get('relation_id') or norm(row.get('relation')))


def apportioned(groups, total):
    """Largest-remainder proportional allocation, with deterministic ties."""
    size = sum(len(v) for v in groups.values())
    if size < total:
        raise ValueError(f'Only {size} eligible rows for {total} requested')
    quotas = {k: len(v) * total // size for k, v in groups.items()}
    order = sorted(groups, key=lambda k: (-(len(groups[k]) * total % size), k))
    for k in order[:total - sum(quotas.values())]:
        quotas[k] += 1
    return quotas


def blocks(rows, size=375, target=None):
    """Native stages remain intact; split blocks at requested prefix boundaries."""
    end = min(len(rows), target or len(rows))
    start = 0
    index = 0
    while start < end:
        stage = int(rows[start]['stage'])
        stage_end = start + 1
        while stage_end < end and int(rows[stage_end]['stage']) == stage:
            stage_end += 1
        stop = min(start + size, stage_end, *[p for p in PREFIXES if p > start])
        yield {'index': index, 'stage': stage, 'start': start, 'end': stop,
               'stage_end': stop == stage_end, 'prefix_end': stop in PREFIXES}
        start, index = stop, index + 1


def cost_plan(rows, target=100000, batch_size=8):
    bs = list(blocks(rows, target=target))
    return {'target': target, 'blocks': len(bs), 'unique_new_events': target,
            'cumulative_replay_selections': sum(b['start'] for b in bs),
            'cumulative_training_presentations': sum(b['end'] for b in bs),
            'cumulative_optimizer_steps': sum(math.ceil(b['end'] / batch_size) for b in bs),
            'random_k8_max_replay_selections': sum(min(i, 8) for i in range(target)),
            'ours_current_steps_per_event': 1,
            'ours_max_fact_presentations': sum(1 + min(i, 8) for i in range(target)),
            'ours_max_view_presentations': 2 * sum(1 + min(i, 8) for i in range(target))}


def quotas_with_unlabeled_fallback(groups, total):
    labeled = {k: v for k, v in groups.items() if k[0] in ('new', 'update')}
    unlabeled = {k: v for k, v in groups.items() if k[0] not in ('new', 'update')}
    labeled_total = min(total, sum(len(v) for v in labeled.values()))
    result = dict.fromkeys(groups, 0)
    result.update(apportioned(labeled, labeled_total))
    if labeled_total < total:
        result.update(apportioned(unlabeled, total - labeled_total))
    return result


def build(output, data_dir, model_dir, total=100000, seed=42):
    from transformers import AutoTokenizer
    from baselines.wikibigedit_protocol import (
        TokenLengthFilter, _load_source_descriptors, _materialize_record,
        _basic_invalid_reason, _fact_fingerprint, stable_sample_id,
    )
    if output.exists():
        raise FileExistsError(output)
    source, descriptors = _load_source_descriptors(data_dir, verify_hashes=True)
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    lengths = TokenLengthFilter(tok, 128, 16)
    seen_fact, seen_qa = set(), set()
    records, audits = [], []
    stage_targets = [total // 8 + (i < total % 8) for i in range(8)]
    for desc, target in zip(descriptors, stage_targets):
        raw = json.loads((data_dir / desc['path']).read_text())
        invalid = Counter()
        groups = defaultdict(list)
        # Dedup in source order before sampling, including across earlier stages.
        local_fact, local_qa = set(seen_fact), set(seen_qa)
        for idx, row in enumerate(raw):
            reason = _basic_invalid_reason(row, {'new', 'update', ''})
            if reason:
                invalid[reason] += 1
                continue
            fp, qa = _fact_fingerprint(row), (norm(row['update']), norm(row['ans']))
            if fp in local_fact or qa in local_qa:
                invalid['duplicate_fact_or_qa'] += 1
                continue
            local_fact.add(fp)
            local_qa.add(qa)
            groups[(str(row.get('tag')), str(row.get('relation_id') or row.get('relation')))].append((idx, row))
        previously_eligible = [r for r in raw if _basic_invalid_reason(r, {'new', 'update'}) is None]
        seen_fact.update(_fact_fingerprint(r) for r in previously_eligible)
        seen_qa.update((norm(r['update']), norm(r['ans'])) for r in previously_eligible)
        quotas = quotas_with_unlabeled_fallback(groups, target)
        chosen, reserve = [], []
        for key in sorted(groups):
            population = sorted(groups[key], key=lambda item: digest([seed, desc['stage'], item[0]]))
            need = quotas[key]
            accepted = []
            # Reserve supports token-filter shortfalls without changing total.
            count = min(len(population), max(need + 4, math.ceil(need * 1.04)))
            for offset in range(0, count, 256):
                batch = population[offset:min(offset + 256, count)]
                for (idx, row), audit in zip(batch, lengths.audit([r for _, r in batch])):
                    if not audit['core_valid']:
                        invalid['token_length_or_missing_direct_panel'] += 1
                        continue
                    rec = _materialize_record(row, desc, idx, idx + 1, audit)
                    # Audit actual training tokenization (prompt and answer separately).
                    prompt = f"Question: {row['update'].strip()}\nAnswer:"
                    answer = ' ' + row['ans'].strip() + (tok.eos_token or '')
                    n = len(tok(prompt, add_special_tokens=True).input_ids) + len(tok(answer, add_special_tokens=False).input_ids)
                    if n > 128:
                        invalid['actual_training_token_length'] += 1
                        continue
                    rec['original_training_sequence_tokens'] = n
                    accepted.append(rec)
            chosen.extend(accepted[:need])
            reserve.extend(accepted[need:])
        if len(chosen) < target:
            reserve.sort(key=lambda r: (r.get('tag') not in ('new', 'update'), digest([seed, 'fill', r['sample_id']])))
            chosen.extend(reserve[:target - len(chosen)])
        if len(chosen) != target:
            raise ValueError(f'Stage {desc["stage"]} shortfall: {len(chosen)}/{target}')
        chosen.sort(key=lambda r: r['source_global_row_idx'])
        seen_fact.update(_fact_fingerprint(r) for r in chosen)
        seen_qa.update((norm(r['update']), norm(r['ans'])) for r in chosen)
        records.extend(chosen)
        audits.append({'stage': desc['stage'], 'interval': desc['interval'], 'source_rows': len(raw),
                       'selected': len(chosen), 'rejections': dict(invalid),
                       'strata_eligible': {str(k): len(v) for k, v in sorted(groups.items())},
                       'strata_selected': dict(Counter(str((r['tag'], r['relation_id'])) for r in chosen))})
        print(json.dumps({'stage': desc['stage'], 'selected': len(chosen), 'total': len(records)}), flush=True)
    by_slot = defaultdict(list)
    for r in records:
        by_slot[slot(r)].append(r)
    conflict_groups = [group for group in by_slot.values()
                       if len({(r.get('object_id') or norm(r.get('object')), norm(r['ans'])) for r in group}) > 1]
    conflict_ids = {r['sample_id'] for group in conflict_groups for r in group}
    # Freeze question-specific multi-answer masks, including held-out probes.
    panels, question_audit = {}, {}
    for axis, (question, answer) in AXIS_FIELDS.items():
        manifest_axis = {'persona': 'personas', 'locality': 'locality', 'multihop': 'mhop'}.get(axis, axis)
        mapping = defaultdict(set)
        for r in records:
            if r.get(question) and r.get(answer):
                mapping[norm(r[question])].add(norm(r[answer]))
        ambiguous = {q for q, answers in mapping.items() if len(answers) > 1}
        ids = [r['sample_id'] for r in records if r['panel_audit'][manifest_axis]['available']
               and norm(r.get(question)) not in ambiguous
               and (axis not in ('update', 'rephrase', 'persona') or r['sample_id'] not in conflict_ids)]
        panels[axis] = ids
        question_audit[axis] = {'ambiguous_questions': len(ambiguous), 'eligible': len(ids)}
    stage_panels = {}
    for axis, ids in panels.items():
        allowed = set(ids)
        stage_panels[axis] = []
        per_stage = 64 if axis in ('locality', 'multihop') else 512
        for stage in range(1, 9):
            pool = [r for r in records if r['stage'] == stage and r['sample_id'] in allowed]
            pool.sort(key=lambda r: digest([seed, 'evaluation', axis, r['sample_id']]))
            stage_panels[axis].extend(r['sample_id'] for r in pool[:per_stage])
    manifest = {'schema_version': 'wikibigedit-longstream/v1', 'source': source,
                'sampling': {'seed': seed, 'target_total': total, 'native_stage_targets': stage_targets,
                             'stratification': ['native_interval', 'tag', 'relation_id'],
                             'ordering': 'native stage then numeric original source row',
                             'base_unknown_screened': False, 'max_length': 128, 'max_new_tokens': 16},
                'selection': {'sample_ids': [r['sample_id'] for r in records], 'total': total},
                'records': records, 'primary_evaluation_ids': panels, 'stage_evaluation_ids': stage_panels,
                'prefixes': {str(n): {'events': n, 'ids_sha256': digest([r['sample_id'] for r in records[:n]]),
                                      'stage_counts': dict(Counter(str(r['stage']) for r in records[:n]))}
                             for n in PREFIXES if n <= total},
                'audit': {'unique_new_events': total, 'unique_fact_fingerprints': len({r['fact_fingerprint'] for r in records}),
                          'unique_training_qa': len({(norm(r['update']), norm(r['ans'])) for r in records}),
                          'stages': audits, 'multi_answer_slot_groups': len(conflict_groups),
                          'multi_answer_slot_rows': len(conflict_ids), 'question_panels': question_audit,
                          'unresolved_slot_ids': sorted(conflict_ids), 'authoritative_replacements': 0,
                          'replacement_policy': 'Raw new/update tags do not prove single-valued replacement. Unresolved slots and ambiguous questions are excluded from primary retention denominators, but retained in the training stream and audit. No claim about resolved temporal replacement quality.'}}
    manifest['protocol_id'] = 'wikibigedit-longstream-' + digest(manifest)[:16]
    atomic(output, manifest)
    atomic(output.parent / 'data_audit.json', manifest['audit'])
    atomic(output.parent / 'prefix_cost_plan.json', {str(n): cost_plan(records, n) for n in PREFIXES if n <= total})
    # Strict QA-only teacher boundary, matching the selected enhanced V3 recipe.
    train_only = [{'sample_id': r['sample_id'], 'update': r['update'], 'ans': r['ans'],
                   'subject': r.get('subject') or ''} for r in records]
    for shard in range(8):
        atomic(output.parent / 'training_views' / f'train_only_{shard}.json',
               {'schema_version': 'engram-training-only/v1', 'rows': train_only[shard::8]})
    print(json.dumps({'manifest': str(output), 'protocol_id': manifest['protocol_id'],
                      'unique_new_events': total, 'unresolved_slot_rows': len(conflict_ids)}), flush=True)


def evaluate(driver, rows, metadata, cursor, checkpoint, full_prefix=False, base=False):
    """Exact all-prefix direct axes; bounded frozen panels at native boundaries."""
    from baselines.common import answer_scores
    selected = rows if base else rows[:cursor]
    ids_key = 'primary_evaluation_ids' if full_prefix else 'stage_evaluation_ids'
    previous_ids = {r['sample_id'] for r in rows[:max([p for p in PREFIXES if p < cursor], default=0)]}
    records, metrics = [], {'checkpoint': checkpoint, 'events': cursor,
                            'num_facts': len(selected), 'current_stage': selected[-1]['stage'] if selected else None,
                            'evaluation_scope': 'all_eligible_prefix_direct_axes' if full_prefix else 'frozen_stratified_panels',
                            'primary_excludes_unresolved_replacements': True}
    started = time.monotonic()
    for axis, (q, a) in AXIS_FIELDS.items():
        key = 'stage_evaluation_ids' if axis in ('locality', 'multihop') else ids_key
        allowed = set(metadata[key][axis])
        pool = [r for r in selected if r['sample_id'] in allowed]
        predictions = driver.generate([r[q] for r in pool]) if pool else []
        if len(predictions) != len(pool):
            raise ValueError('Evaluation output count differs')
        axis_records = [{'checkpoint': checkpoint, 'sample_id': r['sample_id'], 'fact_stage': r['stage'],
                         'axis': axis, 'question': r[q], 'gold': r[a], 'prediction': p,
                         **answer_scores(p, r[a])} for r, p in zip(pool, predictions)]
        def summary(items):
            n = len(items)
            return {'count': n, 'match_accuracy': sum(r['match'] for r in items) / n if n else None,
                    'exact_accuracy': sum(r['exact'] for r in items) / n if n else None}
        metrics[axis] = summary(axis_records)
        metrics[axis]['historical_prefix'] = summary([r for r in axis_records if r['sample_id'] in previous_ids])
        metrics[axis]['by_native_stage'] = {str(s): summary([r for r in axis_records if r['fact_stage'] == s])
                                           for s in sorted({r['stage'] for r in selected})}
        records.extend(axis_records)
    metrics['evaluation_seconds'] = time.monotonic() - started
    return metrics, records


if __name__ == '__main__':
    import os
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, default=os.getenv('CAMR_DATA_ROOT'))
    p.add_argument('--model-dir', type=Path, default=os.getenv('CAMR_MODEL_DIR'))
    p.add_argument('--total', type=int, default=100000)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.data_dir is None or a.model_dir is None:
        p.error('--data-dir/--model-dir or CAMR_DATA_ROOT/CAMR_MODEL_DIR is required')
    build(a.output, a.data_dir, a.model_dir, a.total, a.seed)
