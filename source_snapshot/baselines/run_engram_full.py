#!/usr/bin/env python3
"""Full-parameter Qwen3-8B continual learning with an online memory writer.

Launch with ``python -m torch.distributed.run --standalone --nproc-per-node=8
-m baselines.run_engram_full ...``. Rank zero owns the frozen sentence encoder,
memory and small actor/critic; all ranks execute the same FSDP operations. ADD
learns the current fact with coverage/uniform history, MERGE adds compatible retrieved history, and
IGNORE skips the current fact and maintains safe history only. Formal evaluation remains closed-book.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist

from baselines.common import (
    answer_scores, append_jsonl, atomic_json, axis_fields, environment_metadata,
    fixed_panel, group_by_stage, load_manifest, manifest_panel, sha256_file,
)
from baselines.engram_continual_core import FactMemory
from baselines.engram_full_backend import FullParameterBackend
from baselines.engram_full_safety import (
    choose_safe_probe_and_replay_pools, safe_query, strict_mastery_match,
)
from baselines.engram_frozen_encoder import HFMeanPoolEncoder, fact_document
from baselines.engram_full_stream import make_events, read_schedule, select_rows
from baselines.engram_online_policy import OnlineWriterPolicy


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.environ.get('CAMR_MODEL_DIR', REPOSITORY_ROOT / 'models/Qwen3-8B-Base'))
ENCODER_DIR = os.environ.get('CAMR_ENCODER_DIR', str(REPOSITORY_ROOT / 'models/all-mpnet-base-v2'))
AXES = ('update', 'rephrase', 'persona', 'locality', 'multihop')
STATE_VERSION = 'camr-online/v1'
WRITER_TRAIN_FIELDS = (
    'sample_id', 'stage', 'update', 'ans', 'subject', 'subject_id',
    'relation', 'relation_id', 'object', 'object_id', 'tag', 'scope',
)


def training_row(row):
    """Freeze this runner's input boundary independently of legacy callers.

    Authoritative replacement metadata is handled explicitly before querying;
    it is not forwarded into another memory implementation's mutation hooks.
    """
    return {key: copy.deepcopy(row[key]) for key in WRITER_TRAIN_FIELDS if key in row}


def source_fingerprints():
    return {name: sha256_file(Path(__file__).parent / name) for name in
            ('run_engram_full.py', 'engram_full_backend.py', 'engram_online_policy.py',
             'engram_continual_core.py', 'common.py', 'engram_full_safety.py',
             'engram_grad_norm.py', 'engram_frozen_encoder.py', 'engram_full_stream.py',
             'engram_training_views.py', 'longstream_protocol.py',
             'fullstream_scale_review.py')}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, default=MODEL_DIR)
    p.add_argument('--embedding-model', default=ENCODER_DIR)
    p.add_argument('--checkpoint-dir', type=Path,
                   help='New checkpoint parent; full FP32 model + Adam needs about 100 GB per save.')
    p.add_argument('--resume-checkpoint', type=Path,
                   help='Resume into a NEW output directory, preserving the previous run.')
    p.add_argument('--save-every-events', type=int, default=0)
    p.add_argument('--save-final-checkpoint', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--stop-after-events', type=int, default=0,
                   help='Stop after this absolute event count, saving pending policy state without flushing it.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mode', choices=['policy', 'rule', 'always_add', 'random_replay', 'hash_replay'], default='policy')
    p.add_argument('--fact-mode', choices=['structured', 'text'], default='structured')
    p.add_argument('--max-stages', type=int, default=0)
    p.add_argument('--examples-per-stage', type=int, default=0)
    p.add_argument('--repeat-every', type=int, default=0,
                   help='Training-only duplicate diagnostic; never reuse held-out paraphrases.')
    p.add_argument('--steps-per-event', type=int, default=2)
    p.add_argument('--learning-rate', type=float, default=2e-6)
    p.add_argument('--parameter-mode', choices=['full', 'lora'], default='full')
    p.add_argument('--fsdp-sharding-strategy', choices=['full_shard', 'shard_grad_op'], default='full_shard')
    p.add_argument('--lora-rank', type=int, default=16)
    p.add_argument('--lora-alpha', type=int, default=32)
    p.add_argument('--lora-dropout', type=float, default=0.0)
    p.add_argument('--training-views', type=int, choices=[1, 2], default=None)
    p.add_argument('--training-view-policy', choices=['accepted_only', 'legacy_wrapper'], default='accepted_only')
    p.add_argument('--history-only-loss-policy', choices=['follow_training_loss', 'legacy_fact_mean'], default='follow_training_loss')
    p.add_argument('--training-views-file', type=Path)
    p.add_argument('--training-schedule', type=Path,
                   help='Replay a completed policy run action and exact history IDs; policy learning is disabled.')
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--max-new-tokens', type=int, default=16)
    p.add_argument('--eval-batch-size', type=int, default=16)
    p.add_argument('--activation-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--flash-attention', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--replay-k', type=int, default=4)
    p.add_argument('--add-replay-k', type=int, default=None,
                   help='Safe-history rows for ADD; defaults to --replay-k. Zero gives a current-only ADD ablation.')
    p.add_argument('--ignore-replay-k', type=int, default=None,
                   help='Safe-history-only maintenance for IGNORE; defaults to --replay-k. Zero disables maintenance.')
    p.add_argument('--ignore-behavior', choices=['original', 'disabled', 'current_only', 'skip_step'],
                   default='original', help='IGNORE control: original skips current; disabled masks IGNORE; '
                   'current_only trains current without replay; skip_step trains neither.')
    p.add_argument('--merge-similar-k', type=int, default=4,
                   help='Maximum learned compatible targets inside the total MERGE replay budget.')
    p.add_argument('--uniform-replay-fraction', type=float, default=0.5)
    p.add_argument('--training-loss', choices=['token_mean', 'role_mean'], default=None)
    p.add_argument('--matched-random-loss', action='store_true',
                   help='Explicitly label random replay as an Ours-loss control, not the standard CE baseline.')
    p.add_argument('--current-loss-weight', type=float, default=0.5,
                   help='Current-fact loss mass in role_mean; replay receives the remaining mass.')
    p.add_argument('--replay-schedule', type=Path)
    p.add_argument('--same-task-replay-only', action='store_true',
                   help='Restrict replay and probes to the current task block in multi-task streams.')
    p.add_argument('--probe-size', type=int, default=None)
    p.add_argument('--similarity-threshold', type=float, default=0.65)
    p.add_argument('--mastery-nll', type=float, default=0.25)
    p.add_argument('--hash-tables', type=int, default=8)
    p.add_argument('--hash-bits', type=int, default=8)
    p.add_argument('--max-candidates', type=int, default=64)
    p.add_argument('--policy-lr', type=float, default=3e-4)
    p.add_argument('--policy-hidden-dim', type=int, default=32)
    p.add_argument('--rollout-steps', type=int, default=4)
    p.add_argument('--gamma', type=float, default=0.95)
    p.add_argument('--entropy-coef', type=float, default=0.01)
    p.add_argument('--value-coef', type=float, default=0.5)
    p.add_argument('--retention-weight', type=float, default=1.0)
    p.add_argument('--cost-weight', type=float, default=0.01)
    p.add_argument('--locality-panel-size', type=int, default=16)
    p.add_argument('--multihop-panel-size', type=int, default=8)
    p.add_argument('--eval-axes', nargs='+', choices=AXES, default=list(AXES))
    p.add_argument('--skip-base-eval', action='store_true')
    p.add_argument('--no-stage-eval', action='store_true')
    p.add_argument('--log-every', type=int, default=1)
    p.add_argument('--longstream', action='store_true', help='Use frozen 3K/10K/50K/100K prefix evaluation and native-stage checkpoints.')
    p.add_argument('--longstream-control', type=Path, help='Shared target-events control; may only shorten the frozen stream at 50K.')
    args = p.parse_args(argv)
    optimized = args.mode == 'policy'
    if args.training_loss is None:
        args.training_loss = 'role_mean' if optimized or args.matched_random_loss else 'token_mean'
    if args.training_views is None:
        args.training_views = 2 if optimized and args.training_loss == 'role_mean' else 1
    if args.probe_size is None:
        args.probe_size = 4
    if args.matched_random_loss and args.mode != 'random_replay':
        p.error('--matched-random-loss is only for explicitly named random-replay controls')
    if args.mode == 'random_replay' and not args.matched_random_loss:
        if args.training_loss != 'token_mean' or args.training_views != 1 or args.history_only_loss_policy != 'follow_training_loss':
            p.error('Standard random replay requires token_mean, one view, and follow_training_loss; use --matched-random-loss for a separate control')
    if not math.isfinite(args.uniform_replay_fraction) or not 0 < args.uniform_replay_fraction <= 1:
        p.error('--uniform-replay-fraction must be in (0,1]')
    if args.add_replay_k is None:
        args.add_replay_k = args.replay_k
    if args.ignore_replay_k is None:
        args.ignore_replay_k = args.replay_k
    if args.ignore_behavior != 'original' and (args.mode not in ('policy', 'rule') or args.training_schedule):
        p.error('IGNORE controls require policy/rule mode without a frozen training schedule')
    if args.ignore_behavior in ('current_only', 'skip_step'):
        args.ignore_replay_k = 0
    positive = ('steps_per_event', 'max_length', 'max_new_tokens', 'eval_batch_size',
                'replay_k', 'hash_tables', 'hash_bits', 'max_candidates',
                'policy_hidden_dim', 'rollout_steps', 'log_every', 'lora_rank', 'lora_alpha')
    nonnegative = ('save_every_events', 'stop_after_events', 'max_stages',
                   'examples_per_stage', 'repeat_every', 'probe_size',
                   'locality_panel_size', 'multihop_panel_size', 'add_replay_k', 'ignore_replay_k', 'merge_similar_k')
    if any(getattr(args, key) <= 0 for key in positive):
        p.error('Step, batch, policy and hash sizes must be positive.')
    if any(getattr(args, key) < 0 for key in nonnegative):
        p.error('Selection and diagnostic limits cannot be negative.')
    for key in ('learning_rate', 'policy_lr', 'max_grad_norm'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            p.error(f'{key} must be finite and positive')
    for key in ('weight_decay', 'mastery_nll', 'retention_weight', 'cost_weight', 'entropy_coef', 'value_coef'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            p.error(f'{key} must be finite and nonnegative')
    if not 0 <= args.gamma <= 1 or not -1 <= args.similarity_threshold <= 1 or args.hash_bits > 63:
        p.error('Invalid discount, similarity threshold or hash bits.')
    if not math.isfinite(args.current_loss_weight) or not 0 < args.current_loss_weight < 1:
        p.error('--current-loss-weight must be finite and strictly between zero and one')
    if args.merge_similar_k < 1:
        p.error('MERGE must reserve at least one compatible target')
    if args.model_dir.resolve() != MODEL_DIR.resolve():
        p.error(f'This run is fixed to CAMR_MODEL_DIR={MODEL_DIR}.')
    if args.replay_schedule and args.mode != 'random_replay':
        p.error('--replay-schedule is only supported by random_replay.')
    if not math.isfinite(args.lora_dropout) or not 0 <= args.lora_dropout < 1:
        p.error('--lora-dropout must be in [0,1)')
    if args.training_schedule and (args.mode != 'policy' or args.replay_schedule):
        p.error('--training-schedule requires policy mode and excludes --replay-schedule')
    if args.training_views > 1 and args.training_loss != 'role_mean':
        p.error('Multiple training views require --training-loss role_mean')
    if args.longstream and (args.steps_per_event != 1 or args.repeat_every):
        p.error('Longstream single-current-pass requires steps_per_event=1 and no diagnostic repeats')
    if args.checkpoint_dir is None:
        root = Path(os.environ.get('CAMR_CHECKPOINT_ROOT', args.output_dir.parent / 'checkpoints'))
        args.checkpoint_dir = root / args.output_dir.name
    return args


class BackendCoordinator:
    """Keep collective model calls in exactly the same order on every rank."""

    def __init__(self, backend: FullParameterBackend):
        self.backend = backend

    def _send(self, op: str, **kwargs: Any) -> None:
        if self.backend.world_size > 1:
            dist.broadcast_object_list([{'op': op, **kwargs}], src=0)

    def answer_nll(self, rows):
        self._send('nll', rows=rows)
        return self.backend.answer_nll(rows)

    def generate(self, questions):
        self._send('generate', questions=questions)
        return self.backend.generate(questions)

    def train(self, rows, steps, *, has_current=True):
        self._send('train', rows=rows, steps=steps, has_current=has_current)
        return self.backend.train(rows, steps, has_current=has_current)

    def save_checkpoint(self, path, state):
        self._send('save', path=str(path))
        return self.backend.save_checkpoint(path, state)

    def load_checkpoint(self, path):
        self._send('load', path=str(path))
        return self.backend.load_checkpoint(path)

    def shutdown(self):
        self._send('shutdown')


def serve_backend(backend: FullParameterBackend) -> None:
    while True:
        message = [None]
        dist.broadcast_object_list(message, src=0)
        command = message[0]
        op = command['op']
        if op == 'shutdown':
            return
        if op == 'nll':
            backend.answer_nll(command['rows'])
        elif op == 'generate':
            backend.generate(command['questions'])
        elif op == 'train':
            backend.train(command['rows'], command['steps'], has_current=command['has_current'])
        elif op == 'save':
            backend.save_checkpoint(Path(command['path']), None)
        elif op == 'load':
            backend.load_checkpoint(Path(command['path']))
        else:
            raise ValueError(f'Unknown distributed operation: {op}')


def memory_state(memory: FactMemory) -> dict[str, Any]:
    if hasattr(memory, 'state_dict'):
        return memory.state_dict()
    return {
        'config': {name: getattr(memory, name) for name in (
            'dim', 'seed', 'tables', 'bits', 'max_candidates', 'similarity_threshold', 'fact_mode')},
        'hyperplanes': memory.hyperplanes.clone(),
        'rows': copy.deepcopy(memory.rows),
        'vectors': [vector.clone() for vector in memory._vectors],
    }


def restore_memory(state: dict[str, Any]) -> FactMemory:
    if 'version' in state and hasattr(FactMemory, 'from_state_dict'):
        return FactMemory.from_state_dict(state)
    memory = FactMemory(**state['config'])
    if state['hyperplanes'].shape != memory.hyperplanes.shape or len(state['rows']) != len(state['vectors']):
        raise ValueError('Invalid memory checkpoint shapes')
    memory.hyperplanes = state['hyperplanes'].clone()
    for row, vector in zip(state['rows'], state['vectors']):
        memory.insert(row, vector)
    return memory


def choose_probe_and_replay_pools(memory, query, row, probe_rng, probe_size, retired_ids=()):
    """Fix the audit sample before choosing an action; never train it this step."""
    return choose_safe_probe_and_replay_pools(memory, query, row, probe_rng, probe_size, retired_ids)


def select_camr_replay(action, available, gate_candidates, budget, rng,
                       policy, decision, similar_k=4, uniform_fraction=0.5):
    """Sample the paper's uniform reserve and learned MERGE targets."""
    if action not in {'add', 'merge', 'ignore'}:
        raise ValueError('Unsupported action')
    if budget < 0 or similar_k < 0 or not math.isfinite(uniform_fraction) or not 0 <= uniform_fraction <= 1:
        raise ValueError('Invalid replay budget/fraction')
    budget = min(budget, len(available))
    target_slots = min(similar_k, budget - math.ceil(budget * uniform_fraction)) if action == 'merge' else 0
    uniform = rng.sample(available, budget - target_slots)
    selected_ids = {str(row['sample_id']) for row in uniform}
    available_ids = {str(row['sample_id']) for row in gate_candidates}
    if len(available_ids) != len(gate_candidates):
        raise ValueError('Duplicate target candidate IDs')
    eligible = [i for i, row in enumerate(gate_candidates) if str(row['sample_id']) not in selected_ids]
    chosen = policy.select_targets(decision, eligible, min(target_slots, len(eligible))) if target_slots else []
    learned = [gate_candidates[i] for i in chosen]
    selected_ids.update(str(row['sample_id']) for row in learned)
    missing = budget - len(uniform) - len(learned)
    fill = []
    if missing and len(available) <= 2 * budget:
        pool = [row for row in available if str(row['sample_id']) not in selected_ids]
        fill = rng.sample(pool, missing)
    else:
        while len(fill) < missing:
            row = available[rng.randrange(len(available))]
            identifier = str(row['sample_id'])
            if identifier not in selected_ids:
                fill.append(row)
                selected_ids.add(identifier)
    rows = uniform + learned + fill
    sources = (['random_history'] * len(uniform) + ['learned_history'] * len(learned)
               + ['random_history'] * len(fill))
    return rows, sources


def compatible_config(args, manifest_hash, encoder_hash):
    ignored = {'output_dir', 'checkpoint_dir', 'resume_checkpoint', 'save_every_events',
               'save_final_checkpoint', 'stop_after_events', 'log_every'}
    values = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key not in ignored}
    values.update(manifest_sha256=manifest_hash, encoder_sha256=encoder_hash,
                  source_sha256=source_fingerprints())
    if args.replay_schedule:
        values['replay_schedule_sha256'] = sha256_file(args.replay_schedule)
    if args.training_views_file:
        values['training_views_sha256'] = sha256_file(args.training_views_file)
    if args.training_schedule:
        values['training_schedule_sha256'] = sha256_file(args.training_schedule)
    return values


def stream_digest(events):
    minimal = [{'row': training_row(e['row']), 'arrival_stage': e['arrival_stage'],
                'diagnostic_repeat': e['diagnostic_repeat'],
                'replacement_metadata': {key: e['row'].get(key) for key in
                                         ('supersedes_sample_ids', 'supersedes',
                                          'single_valued', 'authoritative_update')}}
               for e in events]
    return hashlib.sha256(json.dumps(minimal, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def explicit_retirements(event, memory, retired_ids, aliases=None):
    """Only explicit authoritative replacement metadata may retire old targets."""
    raw = event['row']
    if ('supersedes_sample_ids' in raw and 'supersedes' in raw
            and raw['supersedes_sample_ids'] != raw['supersedes']):
        raise ValueError('Conflicting explicit replacement fields')
    replacements = raw.get('supersedes_sample_ids', raw.get('supersedes', []))
    if not isinstance(replacements, list) or any(not isinstance(v, str) for v in replacements):
        raise ValueError('supersedes_sample_ids must be a list of observed sample IDs')
    known = {str(old['sample_id']): old for old in memory.rows}
    aliases = aliases or {}
    current = training_row(raw)
    if raw.get('single_valued') is True and raw.get('authoritative_update') is True:
        replacements = list(dict.fromkeys(replacements + [
            str(old['sample_id']) for old in memory.rows
            if str(old['sample_id']) not in retired_ids
            and memory._slot_key(current) is not None
            and memory._slot_key(old) == memory._slot_key(current)
            and (memory._fact_key(old) != memory._fact_key(current)
                 or not strict_mastery_match(old.get('ans'), current.get('ans')))
        ]))
    canonical_ids = []
    for sample_id in replacements:
        canonical = aliases.get(sample_id, sample_id)
        if canonical not in known or sample_id == current['sample_id']:
            raise ValueError(f'Replacement references unknown/self fact: {sample_id}')
        # Explicit IDs authorize retirement; ordinary differing objects do not.
        if memory._slot_key(known[canonical]) != memory._slot_key(current):
            raise ValueError('Explicit replacement must refer to the same fact slot')
        canonical_ids.append(canonical)
    retired_now = set(canonical_ids) | set(replacements)
    retired_now.update(alias for alias, canonical in aliases.items() if canonical in canonical_ids)
    retired_ids.update(retired_now)
    return sorted(retired_now)


def filtered_query(memory, row, vector, retired_ids):
    return safe_query(memory, row, vector, retired_ids)


def evaluate(driver, rows, locality, multihop, axes, checkpoint, stage):
    """Evaluation-only questions never enter writer state, reward, or memory."""
    metrics = {'checkpoint': checkpoint, 'num_facts': len(rows), 'current_stage': stage}
    records = []
    for axis in axes:
        pool = locality if axis == 'locality' else multihop if axis == 'multihop' else rows
        question_key, answer_key = axis_fields(axis)
        valid = [row for row in pool if isinstance(row.get(question_key), str) and row[question_key].strip()
                 and isinstance(row.get(answer_key), str) and row[answer_key].strip()]
        predictions = driver.generate([row[question_key] for row in valid]) if valid else []
        if len(predictions) != len(valid):
            raise ValueError('Generation count does not match evaluation inputs')
        axis_records = [{'checkpoint': checkpoint, 'sample_id': row['sample_id'],
                         'fact_stage': row['stage'], 'axis': axis,
                         'question': row[question_key], 'gold': row[answer_key],
                         'prediction': prediction, **answer_scores(prediction, row[answer_key])}
                        for row, prediction in zip(valid, predictions)]
        records.extend(axis_records)
        count = len(axis_records)
        exact = sum(record['exact'] for record in axis_records)
        match = sum(record['match'] for record in axis_records)
        metrics[axis] = {'count': count, 'exact_count': exact, 'match_count': match,
                         'exact_accuracy': exact / count if count else None,
                         'match_accuracy': match / count if count else None}
        metrics[f'{axis}_accuracy'] = metrics[axis]['match_accuracy']
        if stage is not None and axis in ('update', 'rephrase', 'persona'):
            for name, selected in (
                (f'current_{axis}_accuracy', [r for r in axis_records if r['fact_stage'] == stage]),
                (f'past_{axis}_retention', [r for r in axis_records if r['fact_stage'] < stage]),
            ):
                metrics[name] = sum(r['match'] for r in selected) / len(selected) if selected else None
    metrics['per_fact_stage'] = {
        str(fact_stage): {
            axis: (sum(r['match'] for r in records if r['fact_stage'] == fact_stage and r['axis'] == axis)
                   / sum(1 for r in records if r['fact_stage'] == fact_stage and r['axis'] == axis))
            if any(r['fact_stage'] == fact_stage and r['axis'] == axis for r in records) else None
            for axis in ('update', 'rephrase', 'persona')}
        for fact_stage in sorted({r['stage'] for r in rows})}
    return metrics, records


def read_training_schedule(path, events, steps_per_event):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) != len(events):
        raise ValueError('Exact training schedule length differs from stream')
    for index, (record, event) in enumerate(zip(records, events)):
        replay_ids = record.get('replay_ids')
        if (record.get('event') != index or record.get('sample_id') != event['row']['sample_id']
                or record.get('arrival_stage') != event['arrival_stage']
                or record.get('diagnostic_repeat') != event['diagnostic_repeat']
                or record.get('action') not in ('add', 'merge', 'ignore')
                or not isinstance(replay_ids, list) or len(set(replay_ids)) != len(replay_ids)
                or record.get('replay_count') != len(replay_ids)):
            raise ValueError(f'Invalid exact training schedule event {index}')
        if record['action'] == 'ignore' and replay_ids and record.get('training_role') != 'history_only':
            raise ValueError('IGNORE replay schedule must declare history_only training role')
        if record['action'] != 'ignore' and record.get('training_role', 'current_and_history') != 'current_and_history':
            raise ValueError('ADD/MERGE schedule must train the current fact')
        if record['action'] == 'merge' and not replay_ids:
            raise ValueError('MERGE schedule requires history')
        if (record['action'] != 'ignore' or replay_ids) and (record.get('train') or {}).get('optimizer_steps') != steps_per_event:
            raise ValueError('Exact schedule optimizer-step budget differs')
        if record['action'] == 'ignore' and replay_ids and (record.get('train') or {}).get('has_current') is not False:
            raise ValueError('IGNORE replay source must certify has_current=False')
    return records


def run(args, driver: BackendCoordinator) -> dict[str, Any]:
    """Rank-zero experiment controller; separated for meaningful CPU tests."""
    if args.mode == 'random_replay' and not args.matched_random_loss and (
            args.training_loss != 'token_mean' or args.training_views != 1):
        raise ValueError('Standard random replay must use ordinary answer-token CE and one view')
    for key in ('training_loss', 'training_views', 'training_view_policy', 'history_only_loss_policy'):
        actual = driver.backend.metadata.get(key)
        if actual is not None and actual != getattr(args, key):
            raise ValueError(f'Backend objective differs from requested configuration: {key}')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty run: {args.output_dir}')
    all_rows, metadata = load_manifest(args.manifest)
    rows = select_rows(all_rows, args.max_stages, args.examples_per_stage)
    if not rows:
        raise ValueError('Empty knowledge stream')
    events = make_events(rows, args.repeat_every)
    if args.longstream:
        # The indexed safety adapter preserves the original candidate/pool
        # order and RNG consumption while avoiding whole-history deep copies.
        import sys
        from baselines.fullstream_scale_review import install
        install(sys.modules[__name__])
        if metadata.get('schema_version') != 'wikibigedit-longstream/v1':
            raise ValueError('Longstream mode requires the audited frozen longstream manifest')
        if [r['sample_id'] for r in rows] != metadata['selection']['sample_ids']:
            raise ValueError('Runner changed the frozen source order')
    schedule = read_schedule(args.replay_schedule, events,
                                 allow_ignore_replay=args.ignore_replay_k > 0,
                                 random_replay=args.mode == 'random_replay') if args.replay_schedule else None
    exact_schedule = read_training_schedule(args.training_schedule, events, args.steps_per_event) if args.training_schedule else None
    encoder = HFMeanPoolEncoder(args.embedding_model, device=torch.device('cpu'), batch_size=32,
                               max_length=args.max_length, query_prefix='', document_prefix='', local_files_only=True)
    dim = int(encoder.model.config.hidden_size)
    memory = FactMemory(dim, seed=args.seed, tables=args.hash_tables, bits=args.hash_bits,
                        max_candidates=args.max_candidates, similarity_threshold=args.similarity_threshold,
                        fact_mode=args.fact_mode)
    policy = OnlineWriterPolicy(dim, feature_dim=8, hidden_dim=args.policy_hidden_dim,
                                lr=args.policy_lr, gamma=args.gamma, entropy_coef=args.entropy_coef,
                                value_coef=args.value_coef, seed=args.seed)
    replay_rng, probe_rng = random.Random(args.seed + 9001), random.Random(args.seed + 9002)
    probe_sampler = probe_rng
    signature = compatible_config(args, sha256_file(args.manifest), encoder.provenance.get('provenance_sha256'))
    digest = stream_digest(events)
    retired_ids, seen_ids, embeddings, aliases = set(), set(), {}, {}
    replay_history = {}
    event_cursor = 0
    actions, metrics, predictions, updates = [], [], [], []
    action_counts, totals, stages = Counter(), Counter(), {}
    resumed = None
    if args.resume_checkpoint:
        resumed = driver.load_checkpoint(args.resume_checkpoint)
        if resumed['version'] != STATE_VERSION or resumed['stream_digest'] != digest:
            raise ValueError('Checkpoint stream/version mismatch')
        if resumed['compatible_config'] != signature:
            differences = [k for k in signature if signature.get(k) != resumed['compatible_config'].get(k)]
            raise ValueError(f'Checkpoint configuration mismatch: {differences}')
        memory = restore_memory(resumed['memory'])
        policy.load_training_state_dict(resumed['policy'])
        replay_rng.setstate(resumed['replay_rng'])
        probe_rng.setstate(resumed['probe_rng'])
        retired_ids, seen_ids = set(resumed['retired_ids']), set(resumed['seen_ids'])
        embeddings = resumed['embeddings']
        aliases = resumed['aliases']
        replay_history = resumed['replay_history']
        event_cursor = resumed['event_cursor']
        actions, metrics = resumed['actions'], resumed['metrics']
        predictions, updates = resumed['predictions'], resumed['policy_updates']
        action_counts, totals = Counter(resumed['action_counts']), Counter(resumed['totals'])
        stages = resumed['stages']
        if event_cursor < 0 or event_cursor > len(events) or len(actions) != event_cursor:
            raise ValueError('Invalid checkpoint event cursor')
    if args.stop_after_events and args.stop_after_events <= event_cursor:
        raise ValueError('--stop-after-events must exceed the resumed event cursor')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({
        'schema_version': STATE_VERSION, 'training_method': args.parameter_mode + '_bf16_fsdp',
        'full_parameter_update': args.parameter_mode == 'full', 'base_weights_frozen': args.parameter_mode == 'lora',
        'trainable_fraction': driver.backend.metadata.get('trainable_fraction', 1.0),
        'trainable_parameters': driver.backend.metadata.get('trainable_parameters'),
        'closed_book_evaluation': True, 'internal_engram_hook': False,
        'encoder': encoder.provenance, 'backend': driver.backend.metadata,
        'num_facts': len(rows), 'num_events': len(events), 'stream_digest': digest,
        'manifest_sha256': sha256_file(args.manifest), 'protocol_id': metadata.get('protocol_id'),
        'training_views_sha256': signature.get('training_views_sha256'),
        'training_schedule_sha256': signature.get('training_schedule_sha256'),
        'split_audit': metadata.get('split_audit'), 'sample_ids': [r['sample_id'] for r in rows],
        'oracle_structure_access': args.fact_mode == 'structured',
        'equivalence_scope': memory.equivalence_method, 'stream_is_diagnostic': bool(args.repeat_every),
        'mastery_metric': 'current answer-token NLL threshold only (for an active equivalent record)',
        'ignore_effect': {
            'original': 'skip certified redundant current; train selected history',
            'disabled': 'mask IGNORE; every current enters training',
            'current_only': 'certified IGNORE trains current at full weight; no replay',
            'skip_step': 'certified IGNORE performs no optimizer step and no replay',
        }[args.ignore_behavior],
        'baseline_objective': ('matched_role_loss_control' if args.matched_random_loss else
                               'standard_answer_token_ce' if args.mode == 'random_replay' else 'ours'),
        'policy_scope': ('frozen action and exact replay IDs from reference policy; no policy learning'
                         if exact_schedule else 'online actor-critic; learned QK action representation and conditional MERGE target sampling'),
        'action_definition': (
            'random_replay=ADD current+uniform safe non-probe history; action is never MERGE or IGNORE'
            if args.mode == 'random_replay' else (
            'ADD=current+uniform safe history; MERGE=current+uniform reserve+learned safe targets; '
            'IGNORE follows ignore_effect and ignore_replay_k; '
            'MERGE denotes joint consolidation, not structural memory deduplication'
        )),
        'probe_policy': 'uniform; pre-action training history only; disjoint from this event replay',
        'reward': 'current gain + retention_weight * mean probe gain '
                  '- cost_weight * logical training tokens / 128',
        'retirement_policy': 'explicit supersedes_sample_ids only; ambiguous differing values are never auto-retired',
        'budget_note': 'matched replay counts/steps do not necessarily match tokens; replicated FSDP physical tokens reported separately',
        'resumed_event_cursor': event_cursor,
        'source_sha256': source_fingerprints(),
        **environment_metadata(),
    })
    atomic_json(args.output_dir / 'run_config.json', config)
    atomic_json(args.output_dir / 'selected_stream.json', rows)
    # A resumed run uses a fresh directory and retains its parent's trace prefix.
    for filename, records in (('actions.jsonl', actions), ('predictions.jsonl', predictions),
                              ('policy_updates.jsonl', updates)):
        if records:
            append_jsonl(args.output_dir / filename, records)
    atomic_json(args.output_dir / 'metrics.json', metrics)
    locality = manifest_panel(metadata, 'locality', limit=args.locality_panel_size)
    multihop = manifest_panel(metadata, 'mhop', limit=args.multihop_panel_size)
    if locality is None:
        locality = fixed_panel(rows, 'loc', 'loc_ans', args.locality_panel_size, args.seed + 1001)
    if multihop is None:
        multihop = fixed_panel(rows, 'mhop', 'mhop_ans', args.multihop_panel_size, args.seed + 2001)
    started = time.perf_counter()
    checkpoint_seconds = 0.0

    def target_events():
        if args.longstream_control:
            target = int(json.loads(args.longstream_control.read_text())['target_events'])
            if target not in (50000, 100000):
                raise ValueError('Shared target must be 50000 or 100000')
            return min(len(events), target)
        return len(events)

    def checkpoint(cursor):
        nonlocal checkpoint_seconds
        state = {'version': STATE_VERSION, 'stream_digest': digest, 'compatible_config': signature,
                 'memory': memory_state(memory), 'policy': policy.training_state_dict(),
                 'replay_rng': replay_rng.getstate(), 'probe_rng': probe_rng.getstate(),
                 'retired_ids': sorted(retired_ids), 'seen_ids': sorted(seen_ids), 'embeddings': embeddings,
                 'aliases': aliases, 'replay_history': replay_history,
                 'event_cursor': cursor, 'actions': actions, 'metrics': metrics, 'predictions': predictions,
                 'policy_updates': updates, 'action_counts': dict(action_counts), 'totals': dict(totals), 'stages': stages}
        path = args.checkpoint_dir / f'event{cursor:06d}'
        save_started = time.perf_counter()
        driver.save_checkpoint(path, state)
        duration = time.perf_counter() - save_started
        checkpoint_seconds += duration
        if args.longstream:
            checkpoint_bytes = sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
            append_jsonl(args.output_dir / 'checkpoint_costs.jsonl', [{'events': cursor, 'path': str(path),
                'seconds': duration, 'bytes': checkpoint_bytes}])
            if metrics and metrics[-1].get('events') == cursor:
                metrics[-1]['cumulative_cost'].update(elapsed_seconds=time.perf_counter() - started,
                    gpu_hours=(time.perf_counter() - started) * driver.backend.world_size / 3600,
                    checkpoint_seconds=checkpoint_seconds, latest_checkpoint_bytes=checkpoint_bytes)
                atomic_json(args.output_dir / 'metrics.json', metrics)
        atomic_json(args.output_dir / 'latest_checkpoint.json', {'path': str(path), 'event_cursor': cursor})
        return str(path)

    def log_update(update, before_event, terminal=False):
        record = {'before_event': before_event, 'terminal': terminal, **update}
        updates.append(record)
        append_jsonl(args.output_dir / 'policy_updates.jsonl', [record])
        return record

    if not resumed and not args.skip_base_eval:
        if args.longstream:
            from baselines.longstream_protocol import evaluate as evaluate_longstream
            result, records = evaluate_longstream(driver, rows, metadata, 0, 'base', base=True)
        else:
            result, records = evaluate(driver, rows, locality, multihop, args.eval_axes, 'base', None)
        metrics.append(result)
        predictions.extend(records)
        append_jsonl(args.output_dir / 'predictions.jsonl', records)
        atomic_json(args.output_dir / 'metrics.json', metrics)
    final_checkpoint = None
    last_saved_cursor = None
    for index in range(event_cursor, len(events)):
        event_started = time.perf_counter()
        event = events[index]
        row = training_row(event['row'])
        stage = int(event['arrival_stage'])
        if event['diagnostic_repeat'] and row['sample_id'] in retired_ids:
            raise ValueError('Diagnostic repeat would reintroduce a retired target; use a version-aware diagnostic stream.')
        retired_now = explicit_retirements(event, memory, retired_ids, aliases)
        if row['sample_id'] not in embeddings:
            encoding_started = time.perf_counter()
            embeddings[row['sample_id']] = encoder.encode([fact_document(row)])[0].clone()
            totals['encoder_seconds'] += time.perf_counter() - encoding_started
        vector = embeddings[row['sample_id']]
        retrieval_started = time.perf_counter()
        query = filtered_query(memory, row, vector, retired_ids)
        retrieval_seconds = time.perf_counter() - retrieval_started
        probes, available, related = choose_probe_and_replay_pools(memory, query, row, probe_sampler,
                                                                 args.probe_size, retired_ids)
        if args.same_task_replay_only:
            # TRACE continual streams encode each dataset as five consecutive
            # stages.  A replay row is eligible only when it belongs to the
            # same five-stage task block as the current arrival.
            task_block = (stage - 1) // 5
            def same_task(old):
                old_stage = int(old.get('stage', old.get('arrival_stage', -1)))
                return old_stage >= 0 and (old_stage - 1) // 5 == task_block
            probes = [old for old in probes if same_task(old)]
            available = [old for old in available if same_task(old)]
            related = [old for old in related if same_task(old)]
        before, probe_tokens = driver.answer_nll([row] + probes)
        allow_ignore = (args.ignore_behavior != 'disabled' and bool(query['duplicate_rows'])
                        and before[0] <= args.mastery_nll and not retired_now)
        # Section 4.2: once an active equivalent record exists and no
        # retirement is requested, IGNORE mastery is determined solely by
        # the current answer-token NLL threshold. Generation is evaluation,
        # not an additional writer-policy gate.
        mastery_prediction = None
        mastery_tokens = 0
        allow_merge = bool(related)
        features = [query['top_similarity'], float(bool(query['duplicate_rows'])),
                    min(before[0] / 10, 2.0), min(sum(before[1:]) / max(1, len(probes)) / 10, 2.0),
                    sum(bool(item.get('ambiguous')) for item in query['candidates']) / max(1, len(query['candidates'])),
                    min(len(related) / args.max_candidates, 1.0),
                    index / len(events), totals['optimizer_steps'] / max(1, index * args.steps_per_event)]
        related_ids = {old['sample_id'] for old in related}
        gate_candidates = [item for item in query['candidates'] if item['row']['sample_id'] in related_ids]
        candidate_vectors = torch.stack([memory._vectors[item['memory_index']] for item in gate_candidates]) \
            if gate_candidates else torch.empty((0, dim))
        policy_update = None
        if args.mode == 'policy' and not exact_schedule and policy.pending_count >= args.rollout_steps:
            bootstrap = policy.choose(features, vector, candidate_vectors, allow_ignore, allow_merge, mode='bootstrap')
            policy_update = log_update(policy.finish_rollout(bootstrap_decision=bootstrap, terminal=False), index)
        if exact_schedule:
            action = exact_schedule[index]['action']
            if (action == 'ignore' and not allow_ignore) or (action == 'merge' and not allow_merge):
                raise ValueError(f'Exact schedule action is unsafe at event {index}')
            decision = {'action': action, 'probabilities': {name: float(name == action)
                         for name in ('ignore', 'merge', 'add')}, 'gate_scores': []}
        elif args.mode == 'policy':
            decision = policy.choose(features, vector, candidate_vectors, allow_ignore, allow_merge)
        else:
            action = 'add'
            if args.mode == 'rule':
                action = 'ignore' if allow_ignore else 'merge' if allow_merge else 'add'
            elif args.mode == 'hash_replay':
                action = 'merge' if allow_merge else 'add'
            elif args.mode == 'random_replay':
                # Random history replay is a control for ADD.  It has no
                # semantic MERGE or IGNORE action, even when history exists.
                # MERGE is reserved for the learned/rule writer policy.
                allow_merge = False
                count = (schedule[index]['replay_count'] if schedule else
                         min(args.add_replay_k, len(available)))
                if count > len(available):
                    raise ValueError('Not enough non-probe history to match replay schedule')
                action = 'add'
            decision = {'action': action, 'probabilities': {name: float(name == action) for name in ('ignore', 'merge', 'add')},
                        'gate_scores': []}
        replay = []
        replay_sources = []
        replay_source = 'none'
        if decision['action'] != 'ignore' or args.ignore_replay_k or exact_schedule or schedule:
            if exact_schedule:
                by_id = {old['sample_id']: old for old in available}
                ids = exact_schedule[index]['replay_ids']
                if any(identifier not in by_id for identifier in ids):
                    raise ValueError(f'Exact schedule history is outside safe non-probe pool at event {index}')
                budget = (args.ignore_replay_k if decision['action'] == 'ignore' else
                          args.replay_k if decision['action'] == 'merge' else args.add_replay_k)
                if len(ids) > budget:
                    raise ValueError('Exact schedule exceeds configured replay budget')
                replay = [by_id[identifier] for identifier in ids]
                replay_sources = exact_schedule[index].get('replay_sources', ['scheduled_history'] * len(ids))
                if len(replay_sources) != len(replay):
                    raise ValueError('Exact schedule source count mismatch')
                replay_source = 'exact_reference_history' if replay else 'none'
            elif args.mode == 'random_replay':
                count = (schedule[index]['replay_count'] if schedule else
                         min(args.add_replay_k, len(available)))
                replay = replay_rng.sample(available, count)
                replay_source = 'random_history' if replay else 'none'
                replay_sources = ['random_history'] * len(replay)
            elif args.mode == 'policy':
                budget = (args.ignore_replay_k if decision['action'] == 'ignore' else
                          args.replay_k if decision['action'] == 'merge' else args.add_replay_k)
                replay, replay_sources = select_camr_replay(
                    decision['action'], available, [item['row'] for item in gate_candidates],
                    budget, replay_rng, policy, decision, args.merge_similar_k, args.uniform_replay_fraction)
                replay_source = 'learned_uniform_history' if replay else 'none'
            elif decision['action'] == 'merge':
                replay = related[:args.replay_k]
                replay_source = 'similar_history'
                replay_sources = ['similar_history'] * len(replay)
            elif args.mode != 'always_add':
                budget = args.ignore_replay_k if decision['action'] == 'ignore' else args.add_replay_k
                replay = replay_rng.sample(available, min(budget, len(available)))
                replay_source = 'random_history' if replay else 'none'
                replay_sources = ['random_history'] * len(replay)
        if decision['action'] == 'merge' and not replay:
            raise RuntimeError('MERGE requires at least one eligible historical sample')
        train_stats = None
        replay_history_before = {old['sample_id']: copy.deepcopy(replay_history[old['sample_id']]) for old in replay}
        has_current = decision['action'] != 'ignore' or args.ignore_behavior == 'current_only'
        training_rows = [row] + replay if has_current else replay
        training_role = ('current_and_history' if has_current else 'history_only') if training_rows else 'none'
        if training_rows:
            if has_current:
                train_stats = driver.train(training_rows, args.steps_per_event)
            else:
                train_stats = driver.train(training_rows, args.steps_per_event, has_current=False)
            for old in replay:
                history = replay_history[old['sample_id']]
                history['last_replayed_event'] = index
                history['replay_count'] += 1
        after, tokens = driver.answer_nll([row] + probes)
        probe_tokens += tokens
        post_ignore_prediction = None
        post_ignore_mastered = None
        if not has_current:
            # Maintenance can disturb a fact that was mastered before the step.
            # Observe it without adding this current fact to the gradient batch.
            if train_stats:
                post_ignore_mastered = after[0] <= args.mastery_nll
            else:
                post_ignore_prediction, post_ignore_mastered = mastery_prediction, allow_ignore
        current_gain = before[0] - after[0]
        retention_gain = sum(a - b for a, b in zip(before[1:], after[1:])) / max(1, len(probes))
        trained_tokens = int(train_stats['sequence_tokens']) if train_stats else 0
        reward = current_gain + args.retention_weight * retention_gain - args.cost_weight * trained_tokens / 128
        reward_components = {'reward': reward, 'mode': 'raw'}
        if not math.isfinite(reward):
            raise ValueError('Nonfinite policy reward')
        if args.mode == 'policy' and not exact_schedule:
            policy.observe(decision, reward)
        if not query['duplicate_rows']:
            memory.insert(row, vector)
            aliases[row['sample_id']] = row['sample_id']
            replay_history[row['sample_id']] = {
                'inserted_event': index, 'last_replayed_event': None, 'replay_count': 0}
        else:
            aliases[row['sample_id']] = query['duplicate_rows'][0]['sample_id']
        seen_ids.add(row['sample_id'])
        action_counts[decision['action']] += 1
        trace = {
            'event': index, 'sample_id': row['sample_id'], 'arrival_stage': stage,
            'diagnostic_repeat': event['diagnostic_repeat'], 'action': decision['action'],
            'probabilities': decision['probabilities'], 'gate_scores': decision.get('gate_scores', []),
            'gate_candidate_ids': [item['row']['sample_id'] for item in gate_candidates],
            'target_selection': copy.deepcopy(decision.get('state', {}).get('target_selection')),
            'features': features, 'allow_ignore': allow_ignore, 'allow_merge': allow_merge,
            'forced': not allow_ignore and not allow_merge, 'mastery_prediction': mastery_prediction,
            'mastery_nll': before[0], 'candidate_count': query['candidate_count'],
            'post_ignore_prediction': post_ignore_prediction, 'post_ignore_mastered': post_ignore_mastered,
            'training_role': training_role, 'current_fact_trained': has_current,
            'ignore_behavior': args.ignore_behavior,
            'optimizer_step_skipped': train_stats is None,
            'candidate_ids': [item['row']['sample_id'] for item in query['candidates']],
            'candidate_similarities': [item['similarity'] for item in query['candidates']],
            'duplicate_ids': [old['sample_id'] for old in query['duplicate_rows']],
            'ambiguous_ids': [old['sample_id'] for old in query['ambiguous_rows']],
            'retired_now': retired_now, 'replay_count': len(replay), 'replay_source': replay_source,
            'replay_sources': replay_sources, 'replay_history_before': replay_history_before,
            'replay_ids': [old['sample_id'] for old in replay], 'probe_ids': [old['sample_id'] for old in probes],
            'current_gain': current_gain, 'retention_gain': retention_gain, 'reward': reward,
            'reward_components': reward_components,
            'before_nll': before, 'after_nll': after, 'train': train_stats,
            'policy_update': policy_update, 'reward_forward_tokens': probe_tokens,
            'physical_reward_forward_tokens': probe_tokens * driver.backend.world_size,
            'mastery_generation_tokens': mastery_tokens, 'retrieval_seconds': retrieval_seconds,
            'event_seconds': time.perf_counter() - event_started,
        }
        actions.append(trace)
        append_jsonl(args.output_dir / 'actions.jsonl', [trace])
        totals['events'] += 1
        totals['current_events'] += int(has_current)
        totals['replay_count'] += len(replay)
        totals['reward_forward_tokens'] += probe_tokens
        totals['physical_reward_forward_tokens'] += probe_tokens * driver.backend.world_size
        totals['mastery_generation_tokens'] += mastery_tokens
        if training_role == 'history_only':
            totals['ignore_maintenance_events'] += 1
            totals['ignore_maintenance_steps'] += train_stats['optimizer_steps']
            totals['ignore_maintenance_tokens'] += train_stats['sequence_tokens']
            totals['ignore_post_mastery_failures'] += int(not post_ignore_mastered)
        cost_fields = ('optimizer_steps', 'sequence_tokens', 'answer_tokens', 'physical_sequence_tokens', 'wall_seconds',
                       'physical_answer_tokens', 'physical_padded_sequence_tokens', 'current_sequence_tokens',
                       'replay_sequence_tokens', 'current_answer_tokens', 'replay_answer_tokens',
                       'current_fact_presentations', 'replay_fact_presentations', 'current_view_presentations', 'replay_view_presentations')
        for key in cost_fields:
            if train_stats:
                totals[key] += train_stats.get(key, 0)
        stats = stages.setdefault(str(stage), {'stage': stage, 'events': 0, 'optimizer_steps': 0,
                                              'sequence_tokens': 0, 'physical_sequence_tokens': 0, 'action_counts': {}})
        stats['events'] += 1
        stats['action_counts'][decision['action']] = stats['action_counts'].get(decision['action'], 0) + 1
        for key in cost_fields:
            stats[key] = stats.get(key, 0) + (train_stats.get(key, 0) if train_stats else 0)
        stats['replay_selections'] = stats.get('replay_selections', 0) + len(replay)
        event_cursor = index + 1
        stage_done = event_cursor == len(events) or events[event_cursor]['arrival_stage'] != stage
        prefix_done = args.longstream and event_cursor in (3000, 10000, 50000, 100000)
        if args.longstream and (prefix_done or stage_done):
            from baselines.longstream_protocol import evaluate as evaluate_longstream
            result, records = evaluate_longstream(driver, rows, metadata, event_cursor,
                f'prefix{event_cursor:06d}' if prefix_done else f'tau{stage}', full_prefix=prefix_done)
            result['native_stage_end'] = stage_done
            result['train'] = copy.deepcopy(stats)
            result['cumulative_cost'] = dict(totals)
            result['cumulative_cost'].update(unique_new_events=len(seen_ids),
                elapsed_seconds=time.perf_counter() - started,
                gpu_hours=(time.perf_counter() - started) * driver.backend.world_size / 3600,
                checkpoint_seconds=checkpoint_seconds,
                forward_accounting=copy.deepcopy(getattr(driver.backend, 'forward_accounting', {})))
            metrics.append(result)
            predictions.extend(records)
            append_jsonl(args.output_dir / 'predictions.jsonl', records)
            atomic_json(args.output_dir / 'metrics.json', metrics)
        elif stage_done and (not args.no_stage_eval or event_cursor == len(events)) and not getattr(args, "external_final_evaluation", False):
            current_rows = [r for r in rows if r['sample_id'] in seen_ids and r['sample_id'] not in retired_ids]
            result, records = evaluate(driver, current_rows, locality, multihop, args.eval_axes, f'tau{stage}', stage)
            result['train'] = copy.deepcopy(stats)
            metrics.append(result)
            predictions.extend(records)
            append_jsonl(args.output_dir / 'predictions.jsonl', records)
            atomic_json(args.output_dir / 'metrics.json', metrics)
        atomic_json(args.output_dir / 'training_log.json', list(stages.values()))
        if event_cursor % args.log_every == 0:
            print(json.dumps({'events_done': event_cursor, 'total_events': len(events),
                              'actions': dict(action_counts), 'reward': reward,
                              'optimizer_steps': totals['optimizer_steps']}, ensure_ascii=False), flush=True)
        stopped = bool((args.stop_after_events and event_cursor >= args.stop_after_events or event_cursor >= target_events()) and event_cursor < len(events))
        if event_cursor == target_events() and args.mode == 'policy' and not exact_schedule and policy.pending_count:
            log_update(policy.finish_rollout(terminal=True), event_cursor, terminal=True)
        if stopped or (args.longstream and (stage_done or prefix_done)) or (args.save_every_events and event_cursor % args.save_every_events == 0 and event_cursor < len(events)):
            final_checkpoint = checkpoint(event_cursor)
            last_saved_cursor = event_cursor
        if stopped:
            break
        if args.longstream and event_cursor % args.log_every == 0:
            atomic_json(args.output_dir / 'progress.json', {'status': 'running', 'events': event_cursor,
                'unique_new_events': len(seen_ids), 'target_events': target_events(), 'totals': dict(totals),
                'elapsed_seconds': time.perf_counter() - started, 'checkpoint_seconds': checkpoint_seconds,
                'world_size': driver.backend.world_size})
    complete = event_cursor == target_events()
    if args.longstream and complete and totals['current_fact_presentations'] != totals['current_events']:
        raise ValueError('Single-current-pass audit failed: current presentations differ from current-training events')
    if complete and args.mode == 'policy' and not exact_schedule and policy.pending_count:
        log_update(policy.finish_rollout(terminal=True), event_cursor, terminal=True)
    if complete and args.save_final_checkpoint and last_saved_cursor != event_cursor:
        final_checkpoint = checkpoint(event_cursor)
    torch.save({'memory': memory_state(memory), 'retired_ids': sorted(retired_ids), 'aliases': aliases,
                'replay_history': replay_history}, args.output_dir / 'memory.pt')
    torch.save(policy.training_state_dict(), args.output_dir / 'policy.pt')
    completion = {
        'status': 'complete' if complete else 'stopped', 'method': 'engram_' + args.parameter_mode + '_' + args.mode,
        'events': event_cursor, 'expected_events': target_events(), 'source_events': len(events), 'num_facts': len(seen_ids),
        'final_stage': events[event_cursor - 1]['arrival_stage'] if event_cursor else None,
        'elapsed_seconds': time.perf_counter() - started, 'action_counts': dict(action_counts),
        'optimizer_steps': totals['optimizer_steps'], 'trained_sequence_tokens': totals['sequence_tokens'],
        'physical_sequence_tokens': totals['physical_sequence_tokens'],
        'reward_forward_tokens': totals['reward_forward_tokens'],
        'physical_reward_forward_tokens': totals['physical_reward_forward_tokens'],
        'policy_optimizer_updates': sum(bool(u.get('updated')) for u in updates),
        'actor_optimizer_updates': sum(bool(u.get('actor_updated')) for u in updates),
        'selector_optimizer_updates': sum(bool(u.get('selector_updated')) for u in updates),
        'policy_pending_transitions': policy.pending_count,
        'replay_coverage': {
            'stored_facts': len(replay_history),
            'ever_replayed_facts': sum(h['replay_count'] > 0 for h in replay_history.values()),
            'max_replay_count': max((h['replay_count'] for h in replay_history.values()), default=0),
        },
        'final_checkpoint': final_checkpoint, 'totals': dict(totals),
        'backend': driver.backend.metadata,
        'model_forward_accounting': getattr(driver.backend, 'forward_accounting', {}),
    }
    atomic_json(args.output_dir / 'completed.json', completion)
    print(json.dumps(completion, ensure_ascii=False), flush=True)
    return completion


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    # Check paths before expensive model initialization on every rank.
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty run: {args.output_dir}')
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    torch.set_num_threads(4)
    backend = FullParameterBackend(args)
    if backend.rank:
        serve_backend(backend)
    else:
        driver = BackendCoordinator(backend)
        try:
            run(args, driver)
        finally:
            driver.shutdown()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
