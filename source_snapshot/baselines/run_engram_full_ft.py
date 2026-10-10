#!/usr/bin/env python3
"""External hash writer with online gate/policy learning and full Qwen FSDP updates.

ADD trains only the current fact; MERGE adds compatible historical examples;
IGNORE needs an equivalent active fact, low answer NLL and exact generation.
No internal memory hooks, LoRA, test-label rewards, or rule pretraining.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch

from baselines.common import (
    _manifest_rows, append_jsonl, atomic_json, environment_metadata, fixed_panel,
    group_by_stage, load_manifest, manifest_panel, sha256_file,
)
from baselines.engram_continual_core import ContextAwarePolicy, FactMemory, normalized_text, training_row
from baselines.run_engram_continual import make_events, read_schedule, select_rows
from baselines.run_rag import HFMeanPoolEncoder, fact_document


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, default=Path('/root/engram/Qwen3-8B-Base'))
    p.add_argument('--embedding-model', default='/mnt/yy2607004_hs/embedding_models/all-mpnet-base-v2')
    p.add_argument('--mode', choices=['policy', 'rule', 'always_add', 'hash_replay', 'random_replay'], default='policy')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--fact-mode', choices=['structured', 'text'], default='structured')
    p.add_argument('--max-stages', type=int, default=0)
    p.add_argument('--examples-per-stage', type=int, default=0)
    p.add_argument('--max-events', type=int, default=0, help='Stop after this total event count; resumable, not a completed stream.')
    p.add_argument('--repeat-every', type=int, default=0, help='Diagnostic only: repeat an observed training QA.')
    p.add_argument('--update-steps', type=int, default=1)
    p.add_argument('--per-device-batch-size', type=int, default=1)
    p.add_argument('--eval-batch-size', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=2e-6)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--max-new-tokens', type=int, default=16)
    p.add_argument('--activation-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--flash-attention', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--replay-k', type=int, default=1)
    p.add_argument('--replay-schedule', type=Path, help='Count/step matched random comparison; actual tokens audited separately.')
    p.add_argument('--probe-size', type=int, default=4)
    p.add_argument('--similarity-threshold', type=float, default=0.65)
    p.add_argument('--mastery-nll', type=float, default=0.25)
    p.add_argument('--hash-tables', type=int, default=8)
    p.add_argument('--hash-bits', type=int, default=8)
    p.add_argument('--max-candidates', type=int, default=64)
    p.add_argument('--policy-lr', type=float, default=1e-3)
    p.add_argument('--gate-dim', type=int, default=64)
    p.add_argument('--retention-weight', type=float, default=1.0)
    p.add_argument('--cost-weight', type=float, default=0.01)
    p.add_argument('--locality-panel-size', type=int, default=16)
    p.add_argument('--multihop-panel-size', type=int, default=8)
    p.add_argument('--skip-base-eval', action='store_true')
    p.add_argument('--no-stage-eval', action='store_true', help='Still evaluate the final seen prefix.')
    p.add_argument('--skip-eval', action='store_true', help='Infrastructure smoke only: no quality claims.')
    p.add_argument('--checkpoint-dir', type=Path)
    p.add_argument('--checkpoint-every-events', type=int, default=0)
    p.add_argument('--save-final-checkpoint', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume-checkpoint', type=Path)
    p.add_argument('--local-rank', '--local_rank', type=int, default=None)
    p.add_argument('--distributed-timeout-hours', type=float, default=1.0)
    a = p.parse_args(argv)
    for key in ('update_steps', 'per_device_batch_size', 'eval_batch_size', 'max_length',
                'max_new_tokens', 'replay_k', 'hash_tables', 'hash_bits', 'max_candidates', 'gate_dim'):
        if getattr(a, key) <= 0:
            p.error(f'{key} must be positive')
    for key in ('max_stages', 'examples_per_stage', 'max_events', 'repeat_every', 'probe_size',
                'locality_panel_size', 'multihop_panel_size', 'checkpoint_every_events'):
        if getattr(a, key) < 0:
            p.error(f'{key} must be nonnegative')
    for key in ('learning_rate', 'policy_lr', 'max_grad_norm', 'distributed_timeout_hours'):
        if not math.isfinite(getattr(a, key)) or getattr(a, key) <= 0:
            p.error(f'{key} must be positive and finite')
    for key in ('retention_weight', 'cost_weight', 'weight_decay', 'mastery_nll'):
        if not math.isfinite(getattr(a, key)) or getattr(a, key) < 0:
            p.error(f'{key} must be nonnegative and finite')
    if not -1 <= a.similarity_threshold <= 1 or not 1 <= a.hash_bits <= 63:
        p.error('invalid similarity threshold or hash bit count')
    if a.model_dir.resolve() != Path('/root/engram/Qwen3-8B-Base').resolve():
        p.error('This experiment fixes /root/engram/Qwen3-8B-Base')
    if a.replay_schedule and a.mode != 'random_replay':
        p.error('a replay schedule requires random_replay mode')
    if a.checkpoint_dir is None:
        a.checkpoint_dir = Path('/mnt/yy2607004_hs/engram_checkpoints') / a.output_dir.name
    return a


def mean(values):
    return sum(values) / len(values) if values else 0.0


def normalize_row_ids(row):
    """Use the core's string ID convention at every runner input boundary."""
    row = dict(row)
    row['sample_id'] = str(row['sample_id'])
    if 'supersedes' in row:
        references = row['supersedes']
        if not isinstance(references, (list, tuple)) or any(
                not isinstance(sid, (str, int)) or isinstance(sid, bool) for sid in references):
            raise ValueError('supersedes must be a list of previous string/integer sample IDs')
        row['supersedes'] = [str(sid) for sid in references]
    return row


def normalize_aliases(aliases):
    result = {}
    for observed, canonical in aliases.items():
        observed, canonical = str(observed), str(canonical)
        if observed in result and result[observed] != canonical:
            raise ValueError(f'conflicting canonical IDs for observed sample {observed}')
        result[observed] = canonical
    return result


def restore_memory(memory_state):
    # Normalize legacy integer IDs without changing vector/index/retirement
    # positions. The source/config identity check still precedes restoration;
    # this does not permit continuing a checkpoint from different source code.
    memory_state = dict(memory_state)
    memory_state['rows'] = [normalize_row_ids(row) for row in memory_state['rows']]
    return FactMemory.from_state_dict(memory_state)


def select_stream(path, max_stages, examples_per_stage):
    rows, metadata = load_manifest(path)
    rows = [normalize_row_ids(row) for row in rows]
    if any(row.get('supersedes') or row.get('authoritative_update') for row in rows):
        original = _manifest_rows(json.loads(path.read_text()))
        positions = {str(row['sample_id']): index for index, row in enumerate(original)}
        rows.sort(key=lambda row: (row['stage'], positions[row['sample_id']]))
        stages = list(group_by_stage(rows).values())[:max_stages or None]
        return [row for stage in stages for row in stage[:examples_per_stage or None]], metadata, 'stage_then_manifest_order'
    return select_rows(rows, max_stages, examples_per_stage), metadata, 'stage_then_sample_id'


def predecision_history(memory, query, row, rng, probe_size, replay_k):
    """Fix reward probes before sampling; reserve a bounded replay opportunity.

    Selection depends on the observed prefix, never the chosen action. Exclude
    equivalent/same-slot-ambiguous/retiring targets from both history pools.
    """
    blocked = {row['sample_id']}
    for key in ('duplicate_rows', 'ambiguous_rows', 'replacement_rows'):
        blocked.update(old['sample_id'] for old in query.get(key, []))
    active = memory.active_rows
    # Unresolved same-slot targets remain auditable, but neither becomes safe
    # replay merely because the next input happens to concern another subject.
    slots = {}
    for old in active:
        key = memory._slot_key(old)
        if key is not None:
            slots.setdefault(key, []).append(old)
    for members in slots.values():
        if len({memory._fact_key(old) for old in members}) > 1:
            blocked.update(old['sample_id'] for old in members)
    questions = {}
    for old in active + [row]:
        questions.setdefault(normalized_text(old['update']), []).append(old)
    for members in questions.values():
        if len({normalized_text(old['ans']) for old in members}) > 1:
            blocked.update(old['sample_id'] for old in members)
    eligible = [old for old in active if old['sample_id'] not in blocked]
    probe_count = min(probe_size, max(0, len(eligible) - replay_k))
    probes = rng.sample(eligible, probe_count)
    probe_ids = {old['sample_id'] for old in probes}
    replay_pool = [old for old in eligible if old['sample_id'] not in probe_ids]
    replay_ids = {old['sample_id'] for old in replay_pool}
    related = [old for old in query['similar_rows'] if old['sample_id'] in replay_ids]
    return probes, replay_pool, related


def mastery_allowed(query, nll, prediction, gold, threshold):
    return (bool(query['duplicate_rows']) and not query.get('replacement_rows')
            and nll <= threshold and prediction is not None
            and bool(normalized_text(gold))
            and normalized_text(prediction) == normalized_text(gold))


def resolve_supersedes(row, observed_aliases):
    """An equivalent arrival is still a valid observed ID for future updates."""
    row = normalize_row_ids(training_row(row))
    aliases = normalize_aliases(observed_aliases)
    if 'supersedes' in row:
        if row['sample_id'] in row['supersedes']:
            raise ValueError('an incoming sample cannot supersede its own sample ID')
        row['supersedes'] = list(dict.fromkeys(aliases.get(sid, sid) for sid in row['supersedes']))
    return row


def commit_observation(memory, row, vector, query, observed_aliases):
    """Commit retirement and retain one active canonical for the new fact.

    A confirmed replacement may also equal an already-active record. In that
    case all equivalent canonicals advance to this new observation, preserving
    the core's strictly forward retirement indices. Their aliases follow the
    new canonical; aliases for different, obsolete targets remain retired.
    """
    row = normalize_row_ids(training_row(row))
    normalized_aliases = normalize_aliases(observed_aliases)
    observed_aliases.clear()
    observed_aliases.update(normalized_aliases)
    canonical = row['sample_id']
    replacements, duplicates = query.get('replacement_rows', []), query['duplicate_rows']
    superseded = [str(old['sample_id']) for old in replacements]
    coalesced = [str(old['sample_id']) for old in duplicates] if replacements else []
    if not duplicates or replacements:
        if coalesced:
            row['supersedes'] = list(dict.fromkeys(row.get('supersedes', []) + superseded + coalesced))
        memory.insert(row, vector)
        if coalesced:
            for observed, previous in list(observed_aliases.items()):
                if previous in coalesced:
                    observed_aliases[observed] = canonical
        observed_aliases[canonical] = canonical
    else:
        canonical = str(duplicates[0]['sample_id'])
        observed_aliases[row['sample_id']] = canonical
    return {
        'canonical_sample_id': canonical,
        'superseded_sample_ids': list(dict.fromkeys(superseded + coalesced)),
        'coalesced_duplicate_ids': coalesced,
    }


def active_multihop_panel(panel, active_ids, aliases):
    """Conservatively drop probes whose observed source fact was retired.

    A source's equivalent canonical may move without changing its fact; aliases
    preserve those probes. Unobserved fixed-panel sources stay in the panel.
    Locality probes have no demonstrated dependency and are handled separately.
    """
    return [row for row in panel if row['sample_id'] not in aliases
            or aliases[row['sample_id']] in active_ids]


def choose_replay(decision, related, random_pool, rng, k, schedule_record=None):
    """A scheduled IGNORE cannot override this model's own mastery evidence."""
    if schedule_record is not None:
        action = schedule_record['action']
        if action == 'ignore' and not decision['allow_ignore']:
            raise ValueError('Cannot copy scheduled ignore: this model has not mastered the fact')
        count = schedule_record['replay_count']
        if count > len(random_pool):
            raise ValueError('Insufficient safe non-probe history for scheduled replay')
        replay = rng.sample(random_pool, count)
        decision.update(action=action, log_prob=None,
                        probabilities={name: float(name == action) for name in ('ignore', 'merge', 'add')})
    elif decision['mode'] == 'random_replay':
        replay = rng.sample(random_pool, min(k, len(random_pool)))
        action = 'merge' if replay else 'add'
        decision.update(action=action, log_prob=None,
                        probabilities={name: float(name == action) for name in ('ignore', 'merge', 'add')})
    else:
        replay = related[:k] if decision['action'] == 'merge' else []
    return replay


def resume_identity(args, manifest_digest, rows, encoder_provenance):
    excluded = {'output_dir', 'checkpoint_dir', 'resume_checkpoint', 'max_events',
                'checkpoint_every_events', 'save_final_checkpoint', 'skip_eval',
                'skip_base_eval', 'no_stage_eval', 'local_rank', 'distributed_timeout_hours'}
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key not in excluded}
    config.update(manifest_sha256=manifest_digest,
                  sample_ids=[row['sample_id'] for row in rows], encoder=encoder_provenance)
    # Changing local implementation invalidates an exact continuation claim.
    config['sources'] = {name: sha256_file(Path(__file__).parent / name)
                         for name in ('run_engram_full_ft.py', 'engram_full_ft.py', 'engram_continual_core.py',
                                      'engram_grad_norm.py', 'common.py', 'run_rag.py',
                                      'run_engram_continual.py', 'run_full_ft_fsdp.py')}
    if args.replay_schedule:
        config['schedule_sha256'] = sha256_file(args.replay_schedule)
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def run(args, backend, encoder=None):
    rank = backend.rank
    output_error = None
    if rank == 0:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            output_error = f'Refusing to overwrite nonempty run: {args.output_dir}'
        else:
            try:
                args.output_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                output_error = str(exc)
    output_error = backend.broadcast_object(output_error)
    if output_error:
        raise FileExistsError(output_error)
    rows, metadata, stream_order = select_stream(args.manifest, args.max_stages, args.examples_per_stage)
    if not rows:
        raise ValueError('empty training selection')
    if args.repeat_every and any(row.get('supersedes') or row.get('authoritative_update') for row in rows):
        raise ValueError('Automatic repeat diagnostics exclude update streams; supply explicit currently valid repeats instead')
    events = make_events(rows, args.repeat_every)
    schedule = read_schedule(args.replay_schedule, events) if args.replay_schedule else None
    locality = manifest_panel(metadata, 'locality', limit=args.locality_panel_size)
    multihop = manifest_panel(metadata, 'mhop', limit=args.multihop_panel_size)
    if locality is None:
        locality = fixed_panel(rows, 'loc', 'loc_ans', args.locality_panel_size, args.seed + 1001)
    if multihop is None:
        multihop = fixed_panel(rows, 'mhop', 'mhop_ans', args.multihop_panel_size, args.seed + 2001)
    locality = [normalize_row_ids(row) for row in locality]
    multihop = [normalize_row_ids(row) for row in multihop]
    started = time.perf_counter()
    encoder_info = backend.broadcast_object(encoder.provenance if rank == 0 else None)
    dim = int(encoder_info['hidden_size'])
    manifest_digest = sha256_file(args.manifest)
    identity = resume_identity(args, manifest_digest, rows, encoder_info)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(
        schema_version='engram-full-ft-writer/v1', manifest_sha256=manifest_digest,
        stream_order=stream_order,
        protocol_id=metadata.get('protocol_id'), split_audit=metadata.get('split_audit'),
        sample_ids=[row['sample_id'] for row in rows], num_facts=len(rows), num_events=len(events),
        encoder=encoder_info, base_weights_frozen=False, lora=False, internal_memory_hook=False,
        model_numel=backend.model_numel, trainable_numel=backend.trainable_numel,
        parameter_dtype='float32', optimizer_state_dtype='float32', compute_dtype='bfloat16',
        parameter_update_measurement='max over up to 64 evenly spaced elements per parameter shard',
        gradient_clipping='ordered parameter global FULL_SHARD norm',
        world_size=backend.world_size, closed_book_evaluation=True,
        policy_training='joint online event loop; immediate REINFORCE + EMA baseline; no rule pretraining',
        query_source='independent frozen encoder, not Qwen hidden state',
        action_scope='three modes; fixed update_steps and replay_k, no learned compute budget yet',
        fact_scope='structured exact identity or normalized exact QA; not general semantic equivalence',
        sample_id_type='string; integer IDs are normalized at input and restore boundaries',
        multihop_retirement='exclude observed sources whose canonical fact is retired; preserve equivalent alias moves',
        reward_scope='current training answer and preselected non-replayed past training QA NLL changes',
        budget_scope='counts/steps may match; actual distributed tokens and dummy work are logged separately',
        add_definition='current fact only', merge_definition='current fact plus compatible non-probe replay',
        stream_is_diagnostic=bool(args.repeat_every), resume_identity=identity,
        **environment_metadata(),
    )
    memory = policy = replay_rng = probe_rng = None
    history, metrics, seen_rows, observed_aliases = [], [], [], {}
    start_event = 0
    if rank == 0:
        memory = FactMemory(dim, seed=args.seed, tables=args.hash_tables, bits=args.hash_bits,
                            max_candidates=args.max_candidates, similarity_threshold=args.similarity_threshold,
                            fact_mode=args.fact_mode)
        policy = ContextAwarePolicy(dim, feature_dim=8, gate_dim=args.gate_dim, seed=args.seed, lr=args.policy_lr)
        replay_rng, probe_rng = random.Random(args.seed + 9001), random.Random(args.seed + 9002)
    if args.resume_checkpoint:
        restored = backend.load_checkpoint(args.resume_checkpoint, manifest_digest)
        restored = backend.broadcast_object(restored if rank == 0 else None)
        if restored['resume_identity'] != identity:
            raise ValueError('resume training configuration, source, encoder or stream changed')
        start_event = restored['next_event']
        history, metrics = restored['history'], restored['metrics']
        seen_rows = [normalize_row_ids(row) for row in restored['seen_rows']]
        observed_aliases = normalize_aliases(restored.get('observed_aliases', {}))
        if rank == 0:
            memory = restore_memory(restored['memory'])
            for old in memory.rows:
                observed_aliases.setdefault(old['sample_id'], old['sample_id'])
            policy.load_checkpoint_state(restored['policy'])
            replay_rng.setstate(restored['replay_rng'])
            probe_rng.setstate(restored['probe_rng'])
    if rank == 0:
        atomic_json(args.output_dir / 'run_config.json', config)
        atomic_json(args.output_dir / 'selected_stream.json', rows)
        if history:
            append_jsonl(args.output_dir / 'actions.jsonl', history)
        atomic_json(args.output_dir / 'metrics.json', metrics)

    def evaluate(checkpoint, stage):
        active_ids = {old['sample_id'] for old in memory.active_rows} if rank == 0 else None
        active_ids = set(backend.broadcast_object(sorted(active_ids) if rank == 0 else None))
        aliases = backend.broadcast_object(observed_aliases if rank == 0 else None)
        active_evaluation = [old for old in seen_rows if aliases.get(old['sample_id'], old['sample_id']) in active_ids]
        active_multihop = active_multihop_panel(multihop, active_ids, aliases)
        result, predictions = backend.evaluate_checkpoint(active_evaluation, locality, active_multihop, checkpoint, stage)
        result['superseded_evaluation_rows_excluded'] = len(seen_rows) - len(active_evaluation)
        result['superseded_multihop_rows_excluded'] = len(multihop) - len(active_multihop)
        metrics.append(result)
        if rank == 0:
            append_jsonl(args.output_dir / 'predictions.jsonl', predictions)
            atomic_json(args.output_dir / 'metrics.json', metrics)
            print(json.dumps({'checkpoint': checkpoint, 'metrics': result}, ensure_ascii=False), flush=True)

    if not start_event and not args.skip_base_eval and not args.skip_eval:
        # Evaluation-only: this reference does not enter writer state or rewards.
        base_metrics, predictions = backend.evaluate_checkpoint(rows, locality, multihop, 'base', None)
        metrics.append(base_metrics)
        if rank == 0:
            append_jsonl(args.output_dir / 'predictions.jsonl', predictions)
            atomic_json(args.output_dir / 'metrics.json', metrics)

    def checkpoint(next_event):
        state = None
        if rank == 0:
            state = dict(resume_identity=identity, next_event=next_event, history=history,
                         metrics=metrics, seen_rows=seen_rows, memory=memory.state_dict(),
                         policy=policy.checkpoint_state(), replay_rng=replay_rng.getstate(),
                         probe_rng=probe_rng.getstate(), observed_aliases=observed_aliases)
        destination = args.checkpoint_dir / f'event_{next_event:06d}'
        backend.save_checkpoint(destination, state, manifest_digest)
        if rank == 0:
            atomic_json(args.output_dir / 'latest_checkpoint.json', {'path': str(destination), 'next_event': next_event})
        return destination

    stop_event = min(len(events), args.max_events or len(events))
    if not start_event <= stop_event:
        raise ValueError('max-events is before the resumed cursor')
    last_checkpoint_event = None
    for event_index in range(start_event, stop_event):
        event = events[event_index]
        row = training_row(event['row'])
        event_started = time.perf_counter()
        preplan = None
        if rank == 0:
            row = resolve_supersedes(row, observed_aliases)
            encoding_started = time.perf_counter()
            vector = encoder.encode([fact_document(row)])[0]
            encoder_seconds = time.perf_counter() - encoding_started
            query = memory.query(row, vector)
            probes, random_pool, related = predecision_history(
                memory, query, row, probe_rng, args.probe_size, args.replay_k)
            preplan = {'row': row, 'probes': probes, 'has_duplicate': bool(query['duplicate_rows']),
                       'has_replacement': bool(query.get('replacement_rows'))}
        preplan = backend.broadcast_object(preplan)
        row = preplan['row']
        probes = preplan['probes']
        before, reward_tokens = backend.answer_nll([row] + probes)
        mastery_prediction = None
        mastery_tokens = 0
        generation_started = time.perf_counter()
        if preplan['has_duplicate'] and not preplan['has_replacement'] and before[0] <= args.mastery_nll:
            mastery_prediction = backend.generate_answers([row['update']])[0]
            mastery_tokens = getattr(backend, 'last_generation_tokens', 0)
        mastery_seconds = time.perf_counter() - generation_started
        plan = None
        if rank == 0:
            allow_ignore = mastery_allowed(query, before[0], mastery_prediction, row['ans'], args.mastery_nll)
            features = [query['top_similarity'], float(preplan['has_duplicate']), min(before[0] / 10, 1),
                        min(len(memory.active_rows) / 1000, 1), query['candidate_count'] / args.max_candidates,
                        min(len(query['ambiguous_rows']) / args.max_candidates, 1),
                        min(mean(before[1:]) / 10, 1), min(event_index / 1000, 1)]
            choose_mode = 'rule' if args.mode == 'hash_replay' else args.mode
            decision = policy.choose(vector, memory.candidate_embeddings(query), features,
                                     allow_ignore=allow_ignore and args.mode != 'hash_replay',
                                     allow_merge=bool(related), mode=choose_mode)
            decision.update(allow_ignore=allow_ignore, mode=args.mode)
            replay = choose_replay(decision, related, random_pool, replay_rng, args.replay_k,
                                   schedule[event_index] if schedule else None)
            plan = {'action': decision['action'], 'replay': replay}
        plan = backend.broadcast_object(plan)
        stats = None
        if plan['action'] != 'ignore':
            stats = backend.train([row] + plan['replay'], seed=args.seed + event_index, steps=args.update_steps)
        after, tokens = backend.answer_nll([row] + probes)
        reward_tokens += tokens
        if rank == 0:
            stat = asdict(stats) if stats is not None else None
            trained_tokens = stat['sequence_tokens'] if stat else 0
            current_gain, retention_gain = before[0] - after[0], mean(before[1:]) - mean(after[1:])
            reward = current_gain + args.retention_weight * retention_gain - args.cost_weight * trained_tokens / 128
            policy_update = policy.learn(decision, reward) if args.mode == 'policy' else None
            replacements = query.get('replacement_rows', [])
            observation = commit_observation(memory, row, vector, query, observed_aliases)
            subtype = ('UPDATE_REPLACE' if replacements else
                       'REFRESH' if query['duplicate_rows'] and plan['action'] != 'ignore' else plan['action'].upper())
            trace = dict(
                event=event_index, sample_id=row['sample_id'], arrival_stage=event['arrival_stage'],
                diagnostic_repeat=event['diagnostic_repeat'], action=plan['action'], internal_relation=subtype,
                probabilities=decision['probabilities'], gate_values=decision.get('gate_values', []),
                candidate_count=query['candidate_count'], hash_recalled_count=query['hash_recalled_count'],
                top_similarity=query['top_similarity'], duplicate_count=len(query['duplicate_rows']),
                ambiguous_count=len(query['ambiguous_rows']),
                requested_supersedes=list(event['row'].get('supersedes', [])),
                **observation,
                allow_ignore=allow_ignore, allow_merge=bool(related), mastery_prediction=mastery_prediction,
                current_nll_before=before[0], current_nll_after=after[0],
                refresh_unmastered_duplicate=bool(query['duplicate_rows']) and not allow_ignore,
                replay_count=len(plan['replay']), replay_sample_ids=[old['sample_id'] for old in plan['replay']],
                probe_sample_ids=[old['sample_id'] for old in probes], probes_selected_before_action=True,
                current_gain=current_gain, retention_gain=retention_gain, reward=reward,
                policy_update=policy_update, reward_forward_tokens=reward_tokens,
                mastery_seconds=mastery_seconds, mastery_forward_tokens=mastery_tokens,
                encoder_seconds=encoder_seconds,
                train=stat, event_seconds=time.perf_counter() - event_started,
                memory_count=len(memory.rows), active_memory_count=len(memory.active_rows),
            )
            history.append(trace)
            append_jsonl(args.output_dir / 'actions.jsonl', [trace])
            print(json.dumps({'event': event_index, 'action': plan['action'], 'reward': reward,
                              'policy_updated': bool(policy_update and policy_update['updated']),
                              'train': stat}, ensure_ascii=False), flush=True)
        if not event['diagnostic_repeat']:
            seen_rows.append(event['row'])
        stage_end = event_index + 1 == len(events) or events[event_index + 1]['arrival_stage'] != event['arrival_stage']
        end = event_index + 1 == stop_event
        if not args.skip_eval and (end or (stage_end and not args.no_stage_eval)):
            evaluate(f'event_{event_index + 1}', event['arrival_stage'])
        if args.checkpoint_every_events and (event_index + 1) % args.checkpoint_every_events == 0:
            checkpoint(event_index + 1)
            last_checkpoint_event = event_index + 1
    if args.save_final_checkpoint and last_checkpoint_event != stop_event:
        checkpoint(stop_event)
    if rank == 0:
        torch.save(policy.checkpoint_state(), args.output_dir / 'policy.pt')
        torch.save(memory.state_dict(), args.output_dir / 'memory.pt')
        complete = stop_event == len(events)
        stats = [record['train'] for record in history if record['train']]
        result = dict(status='complete' if complete else 'bounded_stop', events=stop_event,
                      total_events=len(events), final_stage=events[stop_event - 1]['arrival_stage'],
                      method='engram_full_ft_' + args.mode, elapsed_seconds=time.perf_counter() - started,
                      action_counts=dict(Counter(record['action'] for record in history)),
                      trained_sequence_tokens=sum(stat['sequence_tokens'] for stat in stats),
                      effective_sequence_tokens=sum(stat.get('effective_sequence_tokens', stat['sequence_tokens']) for stat in stats),
                      optimizer_steps=sum(stat['optimizer_steps'] for stat in stats),
                      reward_forward_tokens=sum(record['reward_forward_tokens'] for record in history),
                      mastery_forward_tokens=sum(record.get('mastery_forward_tokens', 0) for record in history),
                      policy_updates=sum(bool(record.get('policy_update', {}).get('updated')) for record in history if record.get('policy_update')),
                      parameter_update_max_abs=max((stat.get('parameter_update_max_abs', 0) for stat in stats), default=0),
                      resumed_from=str(args.resume_checkpoint) if args.resume_checkpoint else None)
        atomic_json(args.output_dir / ('completed.json' if complete else 'bounded_stop.json'), result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    backend.barrier()


def main(argv=None):
    args = parse_args(argv)
    from baselines.engram_full_ft import create_backend
    torch.set_num_threads(4)
    backend = create_backend(args)
    encoder = None
    if backend.rank == 0:
        encoder = HFMeanPoolEncoder(args.embedding_model, device=torch.device('cpu'), batch_size=1,
                                   max_length=args.max_length, query_prefix='', document_prefix='', local_files_only=True)
    run(args, backend, encoder)


if __name__ == '__main__':
    main()
