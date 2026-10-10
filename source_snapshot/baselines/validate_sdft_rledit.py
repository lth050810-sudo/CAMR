"""Verify generated scores, single-pass facts and trained-editor/EMA evidence."""
import argparse
import json
import math
from pathlib import Path

from baselines.common import (AXES, answer_scores, atomic_json, axis_fields,
                              load_manifest, manifest_panel, sha256_file)
from baselines.editor_dev_selection import check_disjoint
from baselines.adaptation_provenance import verify_identity
from baselines.sdft_qwen_core import teacher_prompt


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate(directory, expected=3000):
    config = json.loads((directory / 'run_config.json').read_text())
    done = json.loads((directory / 'completed.json').read_text())
    verify_identity(config['implementation'])
    for name, digest in config['source']['file_sha256'].items():
        if sha256_file(Path(config['source_dir']) / name) != digest:
            raise ValueError('Pinned upstream source changed: ' + name)
    if done['status'] != 'complete' or done['num_facts'] != expected:
        raise ValueError('Incomplete run')
    if expected == 3000 and done['smoke_only']:
        raise ValueError('A smoke test cannot count as a formal result')
    manifest_path = Path(config['manifest'])
    if sha256_file(manifest_path) != config['manifest_sha256']:
        raise ValueError('Frozen manifest changed')
    if config['manifest_sha256'] != '93b84820bfcb4c7367e6ecc41c3fda40fdf45bd03d2770a6efdd7d8e4a0b63a1':
        raise ValueError('This result uses a different 3K comparison panel')
    rows, manifest = load_manifest(manifest_path)
    rows = rows[:expected]
    expected_ids = [r['sample_id'] for r in rows]
    actions = jsonl(directory / 'actions.jsonl')
    trained_ids = [sample_id for event in actions for sample_id in event['sample_ids']]
    if trained_ids != expected_ids or len(set(trained_ids)) != expected:
        raise ValueError('New facts were skipped, repeated or reordered')
    if config['model_dir'] != '/root/engram/Qwen3-8B-Base':
        raise ValueError('Different base model')
    if config['max_new_tokens'] != 16 or not config['closed_book_evaluation']:
        raise ValueError('Incompatible scoring protocol')
    if config['method'].startswith('RLEdit'):
        check_disjoint(Path(config['meta_manifest']), manifest_path)
        if sha256_file(Path(config['meta_manifest'])) != config['meta_manifest_sha256']:
            raise ValueError('Editor meta-training manifest changed')
        if not done['base_restored_before_3k'] or done['meta_optimizer_steps'] != config['meta_epochs']:
            raise ValueError('Missing trained-editor/base-reset evidence')
        if done['trained_hypernetwork_sha256'] == done['initial_hypernetwork_sha256']:
            raise ValueError('Untrained hypernetwork')
        if not Path(done['editor_checkpoint']).is_file() or not Path(done['edit_checkpoint']).is_file():
            raise ValueError('Missing reusable editor or final edit checkpoint')
        meta_events = json.loads((directory / 'meta_training_log.json').read_text())
        meta_rows, _ = load_manifest(Path(config['meta_manifest']))
        meta_rows = meta_rows[:done['meta_num_facts']]
        expected_meta_ids = [r['sample_id'] for r in meta_rows]
        for epoch in range(1, done['meta_epochs'] + 1):
            actual_meta_ids = [sample_id for event in meta_events if event['epoch'] == epoch
                               for sample_id in event['sample_ids']]
            if actual_meta_ids != expected_meta_ids:
                raise ValueError('Editor training did not make its declared pass through dev data')
        meta_ids = {sample_id for event in meta_events for sample_id in event['sample_ids']}
        if meta_ids & set(expected_ids):
            raise ValueError('Final facts used in editor meta-training')
        if len(meta_ids) != done['meta_num_facts'] or not any(a['backtracked_batches'] > 1 for a in meta_events):
            raise ValueError('Incomplete sequential editor training')
        if not any(a['parameter_changed'] for a in actions):
            raise ValueError('The edit phase never changed model weights')
    else:
        if config['kl_direction'] != 'teacher || student' or config['epochs'] != 1:
            raise ValueError('Wrong SDFT loss direction or epoch budget')
        if done['costs']['ema_updates'] != done['optimizer_steps']:
            raise ValueError('Missing EMA teacher updates')
        if done['optimizer_steps'] != config['total_optimizer_steps'] or len(actions) != done['optimizer_steps']:
            raise ValueError('Wrong SDFT optimizer budget')
        if done['student_parameter_fingerprint_before'] == done['student_parameter_fingerprint_after']:
            raise ValueError('No trained student')
        trajectories = jsonl(directory / 'trajectories.jsonl')
        if [r['sample_id'] for r in trajectories] != expected_ids:
            raise ValueError('Trajectories do not cover each new fact exactly once')
        for row, trajectory in zip(rows, trajectories):
            if trajectory['generation_source'] != 'current_student' or trajectory['loss_tokens'] < 1:
                raise ValueError('Invalid on-policy trajectory')
            if trajectory['teacher_prompt'] != teacher_prompt(row['update'], row['ans']):
                raise ValueError('Teacher demonstration differs from the current training fact')
        if expected == 3000:
            checkpoint = Path(done['checkpoint_dir'])
            if not (checkpoint / 'student_final').is_dir():
                raise ValueError('Missing student checkpoint')
            if any(not (checkpoint / f'teacher_rank{rank}.pt').is_file() for rank in range(done['world_size'])):
                raise ValueError('Missing EMA teacher shards')
    metrics = json.loads((directory / 'metrics.json').read_text())[-1]
    records = [r for r in jsonl(directory / 'predictions.jsonl') if r['checkpoint'] == metrics['checkpoint']]
    for axis in AXES:
        actual = [r for r in records if r['axis'] == axis]
        panel_name = 'mhop' if axis == 'multihop' else axis
        panel = manifest_panel(manifest, panel_name, limit=512) if axis in ('locality', 'multihop') else rows
        if expected != 3000 and axis in ('locality', 'multihop'):
            panel = panel[:2]
        q, answer = axis_fields(axis)
        required = {r['sample_id']: (r[q], r[answer]) for r in panel if r.get(q) and r.get(answer)}
        if len(actual) != len(required) or {r['sample_id']: (r['question'], r['gold']) for r in actual} != required:
            raise ValueError('Final generated evaluation panel differs: ' + axis)
        scores = [answer_scores(r['prediction'], r['gold']) for r in actual]
        for kind in ('exact', 'match'):
            count = sum(s[kind] for s in scores)
            if count != metrics[axis][kind + '_count']:
                raise ValueError('Metric does not match generated answers')
            if not math.isclose(metrics[axis][kind + '_accuracy'], count / len(required), abs_tol=1e-12):
                raise ValueError('Reported accuracy differs from the generated answers')
        if metrics[axis]['count'] != len(required):
            raise ValueError('Wrong evaluation denominator')
    result = {'status': 'validated', 'method': config['method'], 'num_facts': expected,
              'source_commit': config['source']['commit'], 'axes': {axis: metrics[axis] for axis in AXES},
              'manifest_sha256': config['manifest_sha256'], 'predictions_sha256': sha256_file(directory / 'predictions.jsonl')}
    atomic_json(directory / 'validation.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--expected-facts', type=int, default=3000)
    a = p.parse_args()
    result = validate(a.output_dir, a.expected_facts)
    print(json.dumps(result))
