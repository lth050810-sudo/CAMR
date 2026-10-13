import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKPOINT_ROOT = Path(os.environ.get('CAMR_CHECKPOINT_ROOT', HERE / 'checkpoints'))
AXES = {'update': ('update', 'ans'), 'rephrase': ('rephrase', 'ans'),
        'persona': ('personas', 'ans'), 'locality': ('loc', 'loc_ans')}
LABELS = {'update': 'Original', 'rephrase': 'Rephrase', 'persona': 'Persona', 'locality': 'Locality'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def checkpoint_inventory(path, expected_events=None, expected_steps=None):
    path = Path(path)
    meta = read(path / 'checkpoint_meta.json')
    if not meta.get('complete') or meta['format'] != 'torch.distributed.checkpoint' or meta['world_size'] != 8:
        raise ValueError(f'Incomplete or incompatible checkpoint: {path}')
    try:
        event_cursor = int(path.name.removeprefix('event'))
    except ValueError as exc:
        raise ValueError(f'Checkpoint directory must be named eventNNNNNN: {path}') from exc
    if expected_events is not None and event_cursor != expected_events:
        raise ValueError('Checkpoint event-cursor mismatch')
    if expected_steps is not None and meta['total_optimizer_steps'] != expected_steps:
        raise ValueError('Checkpoint optimizer-step mismatch')
    required = [path / 'extra_state.pt', path / 'distributed/.metadata']
    required += [path / f'rng-rank-{rank}.pt' for rank in range(8)]
    shards = sorted((path / 'distributed').glob('*.distcp'))
    if len(shards) != 8 or any(not p.is_file() or p.stat().st_size == 0 for p in required + shards):
        raise ValueError(f'Missing checkpoint weights or resume state: {path}')
    return dict(path=str(path), event_cursor=event_cursor,
                optimizer_steps=meta['total_optimizer_steps'], weight_shards=len(shards),
                bytes=sum(p.stat().st_size for p in path.rglob('*') if p.is_file()),
                files={str(p.relative_to(path)): p.stat().st_size for p in required + shards})


def audit_actions(path, expected, expected_rows=None, max_replay=4, max_probe=4):
    count, peak, maximum = 0, 0.0, 0
    optimizer_steps = current_presentations = ignore_events = 0
    last_train = None
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row['event'] != count:
                raise ValueError('Event order mismatch')
            if expected_rows is not None and row['sample_id'] != expected_rows[count]['sample_id']:
                raise ValueError('Source arrival order changed')
            replay, probes = row['replay_ids'], row['probe_ids']
            if (len(replay) != len(set(replay)) or len(replay) != row['replay_count']
                    or len(replay) > max_replay or len(probes) > max_probe
                    or set(replay) & set(probes)):
                raise ValueError('Invalid replay/probe selection')
            train = row.get('train')
            if row['action'] == 'ignore':
                ignore_events += 1
                if (not row.get('allow_ignore') or row.get('ignore_behavior') != 'original'
                        or row.get('current_fact_trained') or row.get('training_role') not in ('history_only', 'none')):
                    raise ValueError('Invalid IGNORE action semantics')
                if replay:
                    if (train is None or train['optimizer_steps'] != 1
                            or train['current_fact_presentations'] != 0
                            or train.get('has_current') is not False
                            or not train['parameter_changed']):
                        raise ValueError('IGNORE maintenance must train history only')
                elif train is not None or not row.get('optimizer_step_skipped'):
                    raise ValueError('Replay-free IGNORE must skip the optimizer step')
            else:
                if (train is None or train['optimizer_steps'] != 1
                        or train['current_fact_presentations'] != 1
                        or not train['parameter_changed'] or not row.get('current_fact_trained')):
                    raise ValueError('ADD/MERGE must train the current fact exactly once')
            if train is not None:
                optimizer_steps += train['optimizer_steps']
                current_presentations += train['current_fact_presentations']
                peak = max(peak, train.get('peak_vram_gb', 0))
                last_train = train
            maximum = max(maximum, len(replay))
            count += 1
    if count != expected or maximum > 8 or last_train is None:
        raise ValueError(f'Action trace count/budget mismatch: {count}, {maximum}')
    return dict(events=count, max_replay_count=maximum, peak_vram_gib=peak,
                optimizer_steps=optimizer_steps, current_fact_presentations=current_presentations,
                ignore_events=ignore_events,
                last_parameter_fingerprint=last_train['parameter_fingerprint_after'])
