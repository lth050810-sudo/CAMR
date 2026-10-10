"""Audit completed longstream predictions on frozen base-unknown probes.

The panel is selected only from the manifest and base-model predictions. It
does not depend on either trained method's outputs and never changes training.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from baselines.common import answer_scores
from baselines.longstream_protocol import AXIS_FIELDS, atomic


def read_predictions(path, checkpoints):
    result = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['checkpoint'] not in checkpoints:
                continue
            key = (row['checkpoint'], row['axis'], row['sample_id'])
            if key in result:
                raise ValueError(f'Duplicate prediction in {path}: {key}')
            result[key] = row
    return result


def freeze_panel(manifest, base, per_stage, seed):
    records = manifest['records']
    allowed = {axis: set(ids) for axis, ids in manifest['stage_evaluation_ids'].items()}
    panel = {}
    counts = {}
    for axis in AXIS_FIELDS:
        candidates = defaultdict(list)
        for row in records:
            sid = row['sample_id']
            if sid not in allowed[axis]:
                continue
            probe = base.get(('base', axis, sid))
            if probe is None:
                raise ValueError(f'Missing base prediction: {axis} {sid}')
            question, gold = row[AXIS_FIELDS[axis][0]], row[AXIS_FIELDS[axis][1]]
            if (probe['question'], probe['gold']) != (question, gold):
                raise ValueError(f'Base prediction disagrees with manifest: {axis} {sid}')
            # Exclude questions that already contain the answer as a token span.
            if probe['match'] or answer_scores(question, gold)['match']:
                continue
            candidates[int(row['stage'])].append(sid)
        chosen = []
        counts[axis] = {}
        for stage in range(1, 9):
            pool = sorted(candidates[stage], key=lambda sid:
                          hashlib.sha256(f'{seed}:{axis}:{sid}'.encode()).hexdigest())
            counts[axis][str(stage)] = len(pool)
            chosen.extend(pool[:per_stage])
        panel[axis] = chosen
    return panel, counts


def score(rows):
    n = len(rows)
    return {'n': n, 'match': sum(r['match'] for r in rows) / n if n else None,
            'exact': sum(r['exact'] for r in rows) / n if n else None}


def evaluate(manifest, panel, predictions, checkpoints):
    by_id = {row['sample_id']: row for row in manifest['records']}
    summaries = []
    for checkpoint, cursor in checkpoints:
        for axis, ids in panel.items():
            selected = [sid for sid in ids if by_id[sid]['stream_index'] < cursor]
            for method, values in predictions.items():
                rows = []
                for sid in selected:
                    item = values.get((checkpoint, axis, sid))
                    if item is None:
                        raise ValueError(f'Missing {method} prediction: {checkpoint} {axis} {sid}')
                    source = by_id[sid]
                    if (item['question'], item['gold']) != (source[AXIS_FIELDS[axis][0]],
                                                            source[AXIS_FIELDS[axis][1]]):
                        raise ValueError(f'Prediction differs from frozen probe: {method} {axis} {sid}')
                    rows.append(item)
                summaries.append({'checkpoint': checkpoint, 'method': method, 'axis': axis,
                                  **score(rows),
                                  'historical': score([r for r in rows if by_id[r['sample_id']]['stream_index'] <
                                                       max((n for n in (3000, 10000, 50000) if n < cursor), default=0)])})
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--ours', type=Path, required=True)
    parser.add_argument('--cumulative-lora', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--per-stage', type=int, default=256)
    parser.add_argument('--seed', type=int, default=20260926)
    args = parser.parse_args()
    if args.per_stage < 1:
        parser.error('--per-stage must be positive')
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('schema_version') != 'wikibigedit-longstream/v1':
        raise ValueError('Expected the frozen longstream manifest')
    for index, row in enumerate(manifest['records']):
        row['stream_index'] = index
    names = {'base', 'prefix050000', 'prefix100000', 'tau7'}
    sources = {'ours': args.ours, 'cumulative_lora': args.cumulative_lora}
    predictions = {name: read_predictions(path, names) for name, path in sources.items()}
    bases = [{key: value for key, value in predictions[name].items() if key[0] == 'base'}
             for name in sources]
    if bases[0] != bases[1]:
        raise ValueError('Base-model prediction sets disagree between runs')
    panel, counts = freeze_panel(manifest, bases[0], args.per_stage, args.seed)
    for axis, ids in panel.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate ID in {axis} panel')
    checkpoint_cursors = [('prefix050000', 50000), ('tau7', 87500), ('prefix100000', 100000)]
    next_index = {row['sample_id']: index for index, row in enumerate(manifest['records'])}
    available = []
    for name, cursor in checkpoint_cursors:
        if all(all((name, axis, sid) in values for axis, ids in panel.items()
                   for sid in ids if next_index[sid] < cursor)
               for values in predictions.values()):
            available.append((name, cursor))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    frozen = {'schema_version': 'longstream-base-unknown-panel/v1',
              'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'selection': 'base match=false; gold token span absent from question; original eligible stage panel',
              'seed': args.seed, 'per_stage_max': args.per_stage,
              'candidate_counts': counts, 'sample_ids': panel}
    panel_path = output / 'panel.json'
    if panel_path.exists() and json.loads(panel_path.read_text()) != frozen:
        raise ValueError('Frozen panel differs from existing file')
    atomic(panel_path, frozen)
    summaries = evaluate(manifest, panel, predictions, available)
    atomic(output / 'results.json', {'checkpoints': available, 'summary': summaries,
                                     'source_predictions': {name: str(path) for name, path in sources.items()}})
    lines = ['# Base-unknown frozen panel', '',
             'Retrospective stress test on previously generated predictions; no new inference. '
             'Each axis uses original eligible stage-panel IDs where the base model missed and '
             'the question does not contain the gold answer. Selection uses no trained-model output.', '',
             f'Panel: {panel_path.name}; seed {args.seed}; max {args.per_stage} per native stage and axis.', '',
             '| Checkpoint | Axis | Method | n | Match | Exact | Historical n | Historical match |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in summaries:
        hist = row['historical']
        pct = lambda v: f'{v:.2%}' if v is not None else '—'
        lines.append(f"| {row['checkpoint']} | {row['axis']} | {row['method']} | {row['n']} | "
                     f"{pct(row['match'])} | {pct(row['exact'])} | {hist['n']} | {pct(hist['match'])} |")
    lines.extend(['', 'The update axis repeats training questions. Rephrase/persona are held-out '
                  'question forms for trained facts; this panel does not test generalization to unseen facts.'])
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'panel_counts': {axis: len(ids) for axis, ids in panel.items()},
                      'checkpoints': available, 'results': str(output / 'RESULTS.md')}))


if __name__ == '__main__':
    main()
