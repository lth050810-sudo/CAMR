import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKPOINT_ROOT = Path('/mnt/yy2607004_hs/engram_checkpoints') / HERE.name
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


def checkpoint_inventory(path, expected_steps=None):
    path = Path(path)
    meta = read(path / 'checkpoint_meta.json')
    if not meta.get('complete') or meta['format'] != 'torch.distributed.checkpoint' or meta['world_size'] != 8:
        raise ValueError(f'Incomplete or incompatible checkpoint: {path}')
    if expected_steps is not None and meta['total_optimizer_steps'] != expected_steps:
        raise ValueError('Checkpoint optimizer-step mismatch')
    required = [path / 'extra_state.pt', path / 'distributed/.metadata']
    required += [path / f'rng-rank-{rank}.pt' for rank in range(8)]
    shards = sorted((path / 'distributed').glob('*.distcp'))
    if len(shards) != 8 or any(not p.is_file() or p.stat().st_size == 0 for p in required + shards):
        raise ValueError(f'Missing checkpoint weights or resume state: {path}')
    return dict(path=str(path), optimizer_steps=meta['total_optimizer_steps'], weight_shards=len(shards),
                bytes=sum(p.stat().st_size for p in path.rglob('*') if p.is_file()),
                files={str(p.relative_to(path)): p.stat().st_size for p in required + shards})


def audit_actions(path, expected, expected_rows=None):
    count, peak, maximum, last = 0, 0.0, 0, None
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row['event'] != count or row['action'] == 'ignore':
                raise ValueError('Event order or current-fact omission')
            if expected_rows is not None and row['sample_id'] != expected_rows[count]['sample_id']:
                raise ValueError('Source arrival order changed')
            replay, probes = row['replay_ids'], row['probe_ids']
            if len(replay) != len(set(replay)) or len(replay) != row['replay_count'] or len(replay) > 8 or set(replay) & set(probes):
                raise ValueError('Invalid replay/probe selection')
            train = row['train']
            if train['optimizer_steps'] != 1 or train['current_fact_presentations'] != 1 or not train['parameter_changed']:
                raise ValueError('Expected exactly one full-parameter update per arrival')
            peak = max(peak, train.get('peak_vram_gb', 0))
            maximum = max(maximum, len(replay))
            last = row
            count += 1
    if count != expected or maximum != 8:
        raise ValueError(f'Action trace count/budget mismatch: {count}, {maximum}')
    return dict(events=count, max_replay_count=maximum, peak_vram_gib=peak,
                last_parameter_fingerprint=last['train']['parameter_fingerprint_after'])
