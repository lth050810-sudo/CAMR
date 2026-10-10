"""Freeze the 150K arrival stream and audit every training/evaluation input."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from baselines.common import atomic_json, qa_prompt, normalize_answer as normalize_text


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', choices=['contiguous', 'balanced'], default='contiguous')
    opts = parser.parse_args()
    if (HERE / 'frozen_inputs.json').exists():
        raise FileExistsError('This experiment is already frozen')
    reference = ROOT / 'runs/wikibigedit_camr_k8_tracks_20261008'
    snapshot = HERE / 'source_snapshot/baselines'
    snapshot.mkdir(parents=True, exist_ok=True)
    for p in (ROOT / 'baselines').glob('*.py'):
        old = reference / 'source_snapshot/baselines' / p.name
        if old.exists() and sha(old) != sha(p):
            raise ValueError(f'Latest validated source differs: {p.name}')
        shutil.copy2(p, snapshot / p.name)
    for name in ['trace_backend.py', 'trace_protocol.py']:
        shutil.copy2(reference / name, HERE / name)
    # Evaluation is a separate resumable job after the final checkpoint commits.
    # The training/replay/reward/update code is unchanged.
    runner = snapshot / 'run_engram_full.py'
    text = runner.read_text()
    old = 'elif stage_done and (not args.no_stage_eval or event_cursor == len(events)):'
    new = 'elif stage_done and (not args.no_stage_eval or event_cursor == len(events)) and not getattr(args, "external_final_evaluation", False):'
    assert text.count(old) == 1
    runner.write_text(text.replace(old, new))
    changes = {
        'reference': str(reference),
        'base_source': 'Identical to the validated 2026-10-08 more_learned sources',
        'changes': {'source_snapshot/baselines/run_engram_full.py': {
            'before': old, 'after': new,
            'reason': 'Save training state before running the separate resumable four-axis evaluation.'}},
        'training_algorithm_changed': False,
    }
    data_root = ROOT / 'WikiBigEdit/full_hf_viewer'
    source = json.loads((data_root / 'manifest.json').read_text())
    rows, sources = [], []
    for desc in sorted(source['files'], key=lambda x: x['stage']):
        if opts.selection == 'contiguous' and len(rows) == 150000:
            break
        path = data_root / desc['path']
        assert sha(path) == desc['sha256_reconstructed']
        raw = json.loads(path.read_text())
        assert isinstance(raw, list) and len(raw) == desc['rows']
        count = min(len(raw), 150000 - len(rows)) if opts.selection == 'contiguous' else 18750
        assert len(raw) >= count
        sources.append(dict(path=str(path), sha256=sha(path), stage=desc['stage'],
                            interval=desc['interval'], source_rows=len(raw), selected_rows=count))
        for index, item in enumerate(raw[:count]):
            row = dict(item, sample_id=f"{desc['interval']}:row{index:06d}",
                       stage=desc['stage'], source_row_idx=index)
            assert all(isinstance(row.get(k), str) and row[k].strip() for k in ['update', 'ans']), row['sample_id']
            rows.append(row)
    assert len(rows) == len({r['sample_id'] for r in rows}) == 150000
    old_rows = json.loads((reference / 'manifest.json').read_text())['samples']
    protocol = None
    if opts.selection == 'balanced':
        # Match the historical 100K tag/relation stratification, eligibility,
        # deduplication and deterministic sampling; change only the stage quota.
        import importlib.util
        protocol_source = snapshot / 'longstream_protocol.py'
        protocol_text = protocol_source.read_text()
        previous = "            reason = _basic_invalid_reason(row, {'new', 'update'})"
        assert protocol_text.count(previous) == 1
        protocol_text = protocol_text.replace(previous, "            reason = _basic_invalid_reason(row, {'new', 'update', ''})")
        previous = '        seen_fact, seen_qa = local_fact, local_qa\n'
        assert protocol_text.count(previous) == 1
        protocol_text = protocol_text.replace(previous,
            "        previously_eligible = [r for r in raw if _basic_invalid_reason(r, {'new', 'update'}) is None]\n"
            "        seen_fact.update(_fact_fingerprint(r) for r in previously_eligible)\n"
            "        seen_qa.update((norm(r['update']), norm(r['ans'])) for r in previously_eligible)\n")
        insertion = '        records.extend(chosen)\n'
        replacement = ('        seen_fact.update(_fact_fingerprint(r) for r in chosen)\n'
                       "        seen_qa.update((norm(r['update']), norm(r['ans'])) for r in chosen)\n"
                       + insertion)
        assert protocol_text.count(insertion) == 1
        protocol_text = protocol_text.replace(insertion, replacement)
        protocol_text = protocol_text.replace('        quotas = apportioned(groups, target)',
                                              '        quotas = quotas_with_unlabeled_fallback(groups, target)')
        protocol_text = protocol_text.replace("reserve.sort(key=lambda r: digest([seed, 'fill', r['sample_id']]))",
            "reserve.sort(key=lambda r: (r.get('tag') not in ('new', 'update'), digest([seed, 'fill', r['sample_id']])))")
        quota_helper = '''\ndef quotas_with_unlabeled_fallback(groups, total):
    labeled = {k: v for k, v in groups.items() if k[0] in ('new', 'update')}
    unlabeled = {k: v for k, v in groups.items() if k[0] not in ('new', 'update')}
    labeled_total = min(total, sum(len(v) for v in labeled.values()))
    result = dict.fromkeys(groups, 0)
    result.update(apportioned(labeled, labeled_total))
    if labeled_total < total:
        result.update(apportioned(unlabeled, total - labeled_total))
    return result

'''
        protocol_text = protocol_text.replace('\ndef build(output,', quota_helper + '\ndef build(output,')
        protocol_source.write_text(protocol_text)
        changes['changes']['source_snapshot/baselines/longstream_protocol.py'] = {
            'before': 'Only source rows tagged new/update are eligible',
            'after': 'Prioritize the same new/update strata; fill an interval shortfall with QA-complete native rows whose original tag is empty. Keep empty tags, token filtering and deduplication. Prefer labeled reserves. Historical exclusion of earlier labeled source facts is retained; selected unlabeled facts are also excluded from later intervals.',
            'reason': 'Interval 2 has only 18,369 valid unique labeled rows before token filtering, independently of cross-stage deduplication; the requested quota is 18,750.',
            'discarded_attempt': 'Restricting cross-stage deduplication to selected facts did not resolve the shortfall and is not used.'}
        spec = importlib.util.spec_from_file_location('protocol_150k', protocol_source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.build(HERE / 'protocol' / 'manifest.json', data_root, ROOT / 'Qwen3-8B-Base', total=150000, seed=42)
        protocol = json.loads((HERE / 'protocol/manifest.json').read_text())
        rows = protocol['records']
        assert len(rows) == 150000
    else:
        assert rows[:5000] == old_rows
    selection = ('First 150,000 consecutive native records in chronological interval order; no shuffle, filtering, deduplication or resampling.'
                 if opts.selection == 'contiguous' else
                 '18,750 records per native interval, eight chronological intervals. Historical 100K tag/relation stratification, token eligibility, deduplication and source ordering; seed 42. Prefer new/update labels; fill interval shortages with QA-complete native rows whose original tag is empty, retaining that tag and auditing their count.')
    atomic_json(HERE / 'source_changes.json', changes)
    payload = dict(schema_version='wikibigedit-camr-150k/v1', selection=selection,
                   selection_mode=opts.selection, sources=sources, samples=rows)
    if protocol:
        payload['primary_evaluation_ids'] = protocol['primary_evaluation_ids']
        payload['stage_evaluation_ids'] = protocol['stage_evaluation_ids']
        payload['historical_protocol_id'] = protocol['protocol_id']
        payload['sampling'] = protocol['sampling']
    atomic_json(HERE / 'manifest.json', payload)
    print(json.dumps({'status': 'auditing_token_lengths', 'rows': len(rows),
                      'stages': dict(Counter(r['stage'] for r in rows))}), flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'Qwen3-8B-Base', local_files_only=True)
    encoder = AutoTokenizer.from_pretrained('/mnt/yy2607004_hs/embedding_models/all-mpnet-base-v2', local_files_only=True)
    maximum = Counter()
    axes = {'update': ('update', 'ans'), 'rephrase': ('rephrase', 'ans'),
            'persona': ('personas', 'ans'), 'locality': ('loc', 'loc_ans')}
    counts, stress_ids = Counter(), set()
    train_lengths = []
    for start in range(0, len(rows), 2000):
        chunk = rows[start:start + 2000]
        prompts = tokenizer([qa_prompt(r['update']) for r in chunk], add_special_tokens=True)['input_ids']
        answers = tokenizer([' ' + r['ans'] for r in chunk], add_special_tokens=False)['input_ids']
        emb = encoder([f"Question: {r['update'].strip()}\nAnswer: {r['ans'].strip()}" for r in chunk],
                      add_special_tokens=True)['input_ids']
        for i, (p, a, e) in enumerate(zip(prompts, answers, emb)):
            length = len(p) + len(a) + 1
            assert length <= 512 and len(e) <= 512, chunk[i]['sample_id']
            train_lengths.append((length, start + i))
            maximum['training_tokens'] = max(maximum['training_tokens'], length)
            maximum['embedding_tokens'] = max(maximum['embedding_tokens'], len(e))
            maximum['answer_tokens_with_eos'] = max(maximum['answer_tokens_with_eos'], len(a) + 1)
        for axis, (q, a) in axes.items():
            valid = [r for r in chunk if isinstance(r.get(q), str) and r[q].strip()
                     and isinstance(r.get(a), str) and r[a].strip()]
            counts[axis] += len(valid)
            if valid:
                lengths = [len(x) for x in tokenizer([qa_prompt(r[q]).rstrip() for r in valid],
                                                   add_special_tokens=False)['input_ids']]
                assert max(lengths) <= 512, (axis, start, max(lengths))
                maximum[axis + '_prompt_tokens'] = max(maximum[axis + '_prompt_tokens'], max(lengths))
        if (start + len(chunk)) % 20000 == 0:
            print(json.dumps({'status': 'auditing', 'rows': start + len(chunk)}), flush=True)
    stress_ids.update(index for _, index in sorted(train_lengths, reverse=True)[:48])
    for stage in sorted({r['stage'] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r['stage'] == stage]
        stress_ids.update(indices[:2] + indices[-2:])
    for i in range(len(rows)):
        if len(stress_ids) >= 80:
            break
        stress_ids.add(i)
    stress_rows = [rows[i] for i in sorted(stress_ids)]
    atomic_json(HERE / 'stress_manifest.json', dict(payload, samples=stress_rows,
                selection='Independent stress test: longest training rows and native stage boundaries'))
    train_questions = {normalize_text(r['update']) for r in rows}
    locality_answers = defaultdict(set)
    locality_overlap = 0
    for r in rows:
        if isinstance(r.get('loc'), str) and r['loc'].strip() and isinstance(r.get('loc_ans'), str) and r['loc_ans'].strip():
            question = normalize_text(r['loc'])
            locality_answers[question].add(normalize_text(r['loc_ans']))
            locality_overlap += question in train_questions
    audit = dict(rows=len(rows), selection=selection, selection_mode=opts.selection, sources=sources,
        manifest_sha256=sha(HERE / 'manifest.json'), native_stage_counts=dict(Counter(r['stage'] for r in rows)),
        tag_counts=dict(Counter(str(r.get('tag')) for r in rows)), valid_evaluation_rows=dict(counts),
        missing_evaluation_rows={k: len(rows) - v for k, v in counts.items()}, maximum_tokens=dict(maximum),
        truncations=0, unique_training_qa=len({(r['update'], r['ans']) for r in rows}),
        locality_unique_questions=len(locality_answers),
        locality_multitarget_questions=sum(len(a) > 1 for a in locality_answers.values()),
        locality_training_question_overlap_rows=locality_overlap,
        first_5000_identical_to_latest_development=opts.selection == 'contiguous', stress_rows=len(stress_rows),
        latest_development_qa_overlap=len({(r['update'], r['ans']) for r in rows} & {(r['update'], r['ans']) for r in old_rows}))
    primary_counts = ({a: len(protocol['stage_evaluation_ids'][a] if a == 'locality'
                                else protocol['primary_evaluation_ids'][a]) for a in axes}
                      if protocol else dict(counts))
    audit['primary_evaluation_rows'] = primary_counts
    atomic_json(HERE / 'data_audit.json', audit)
    config = json.loads((reference / 'results/more_learned/full/run_config.json').read_text())
    common_keys = json.loads((reference / 'experiment_plan.json').read_text())['common']
    atomic_json(HERE / 'experiment_plan.json', dict(version='CAMR-K8-150K/v1',
        method='CAMR V5 more_learned (validated 2026-10-08)', events=150000, seed=42,
        initialization='Fresh Qwen3-8B-Base, all 8,190,735,360 parameters trainable',
        common={k: config[k] for k in common_keys}, training_world_size=8,
        fsdp_sharding_strategy='shard_grad_op', activation_checkpointing=False,
        eval_batch_size=512, checkpoint_every_events=25000,
        recovery_validation='Stop after event 128, save all state, and resume into a fresh output directory',
        checkpoint_policy='Retain event128, all 25K checkpoints, and event150000; no automatic deletion',
        checkpoint_root=str(Path('/mnt/yy2607004_hs/engram_checkpoints') / HERE.name),
        evaluation_axes=list(axes), metric='Match accuracy and Exact accuracy on every available paired row',
        locality_definition='Gold-answer Match on paired loc/loc_ans; row-weighted, not pre/post consistency',
        evaluation_denominators=dict(counts), primary_evaluation_denominators=primary_counts,
        primary_evaluation_scope='Historical 100K eligible direct-axis masks and fixed 64-per-stage locality panel; all paired rows also scored and reported',
        selection=selection,
        comparison_caveat='Latest method selected on the first 5K development set; this is not an untouched validation stream.',
        manifest_sha256=sha(HERE / 'manifest.json')))
    files = [p for p in HERE.rglob('*') if p.is_file() and (p.suffix in ['.py', '.json'] or p.name == 'README.md')
             and '__pycache__' not in p.parts and p.name != 'frozen_inputs.json']
    atomic_json(HERE / 'frozen_inputs.json', {str(p.relative_to(HERE)): sha(p) for p in sorted(files)})
    print(json.dumps({'status': 'prepared', **audit}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
