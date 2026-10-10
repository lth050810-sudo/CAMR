"""Audit a generated 3K random-history control against its policy reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.common import answer_scores, atomic_json, sha256_file


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate(output: Path, reference: Path, expected: int = 3000):
    config = json.loads((output / 'run_config.json').read_text())
    original = json.loads((reference / 'run_config.json').read_text())
    done = json.loads((output / 'completed.json').read_text())
    if done['status'] != 'complete' or done['events'] != expected:
        raise ValueError('Random replay is incomplete')
    if config['mode'] != 'random_replay' or config['parameter_mode'] != 'full':
        raise ValueError('Expected the full-parameter random-history control')
    for key in ('manifest_sha256', 'model_dir', 'seed', 'steps_per_event',
                'learning_rate', 'weight_decay', 'max_grad_norm', 'max_length',
                'max_new_tokens', 'training_views', 'training_views_sha256',
                'training_loss', 'current_loss_weight', 'replay_k', 'add_replay_k'):
        if config[key] != original[key]:
            raise ValueError(f'Control differs from reference in {key}')
    actions = read_jsonl(output / 'actions.jsonl')
    source = read_jsonl(reference / 'actions.jsonl')[:expected]
    if len(actions) != expected or len(source) != expected:
        raise ValueError('Missing action records')
    seen = set()
    changed_selections = 0
    replay_total = updates = 0
    for index, (actual, prior) in enumerate(zip(actions, source)):
        for key in ('event', 'sample_id', 'arrival_stage', 'action', 'replay_count', 'probe_ids'):
            if actual[key] != prior[key]:
                raise ValueError(f'Control budget/schedule mismatch at {index}: {key}')
        if actual['event'] != index or actual['sample_id'] in seen:
            raise ValueError('New facts must be injected once, in order')
        replay = actual['replay_ids']
        if len(replay) != len(set(replay)) or not set(replay) <= seen:
            raise ValueError('Duplicate, current, or future replay')
        if set(replay) & set(actual['probe_ids']):
            raise ValueError('Replay leaked into the probe panel')
        if any(label != 'random_history' for label in actual['replay_sources']):
            raise ValueError('Non-random replay selection')
        for key in ('optimizer_steps', 'training_view_rows'):
            if (actual.get('train') or {}).get(key) != (prior.get('train') or {}).get(key):
                raise ValueError(f'Update/view budget differs at {index}: {key}')
        seen.add(actual['sample_id'])
        replay_total += len(replay)
        updates += (actual.get('train') or {}).get('optimizer_steps', 0)
        changed_selections += set(replay) != set(prior['replay_ids'])
    if done['optimizer_steps'] != updates or done['policy_optimizer_updates'] != 0:
        raise ValueError('Unexpected optimizer accounting')
    if expected == 3000 and not changed_selections:
        raise ValueError('The control reused reference history instead of randomizing it')
    metrics = json.loads((output / 'metrics.json').read_text())[-1]
    predictions = read_jsonl(output / 'predictions.jsonl')
    final = [r for r in predictions if r['checkpoint'] == metrics['checkpoint']]
    if metrics['num_facts'] != expected:
        raise ValueError('Incomplete final evaluation')
    axes = {}
    for axis in ('update', 'rephrase', 'persona', 'locality', 'multihop'):
        records = [r for r in final if r['axis'] == axis]
        if len(records) != metrics[axis]['count']:
            raise ValueError(f'Missing predictions for {axis}')
        if len({r['sample_id'] for r in records}) != len(records):
            raise ValueError('Duplicate predictions')
        recomputed = [answer_scores(r['prediction'], r['gold']) for r in records]
        for kind in ('match', 'exact'):
            count = sum(r[kind] for r in recomputed)
            if count != metrics[axis][kind + '_count']:
                raise ValueError('Stored accuracy disagrees with generated answers')
        axes[axis] = metrics[axis]
    if expected == 3000:
        ref_metrics = json.loads((reference / 'metrics.json').read_text())[-1]
        ref_records = [r for r in read_jsonl(reference / 'predictions.jsonl')
                       if r['checkpoint'] == ref_metrics['checkpoint']]
        panels = lambda records: {(r['axis'], r['sample_id']): (r['question'], r['gold'])
                                  for r in records}
        if panels(final) != panels(ref_records):
            raise ValueError('Final evaluation panels differ')
    result = {'status': 'validated', 'events': expected, 'optimizer_steps': updates,
              'history_selections': replay_total, 'changed_history_sets': changed_selections,
              'new_fact_injections': len(seen), 'axes': axes,
              'reference_actions_sha256': sha256_file(reference / 'actions.jsonl'),
              'random_actions_sha256': sha256_file(output / 'actions.jsonl'),
              'reference_world_size': original['backend']['world_size'],
              'random_world_size': config['backend']['world_size'],
              'comparison': 'Fixed reference actions and replay counts; uniform eligible history; '
                            'same full-parameter optimizer, views and role-weighted objective. '
                            'Sequence lengths and physical FSDP work are reported separately.'}
    atomic_json(output / 'validation.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--expected-events', type=int, default=3000)
    args = parser.parse_args()
    result = validate(args.output_dir, args.reference_dir, args.expected_events)
    print(json.dumps({k: v for k, v in result.items() if k != 'axes'}))


if __name__ == '__main__':
    main()
