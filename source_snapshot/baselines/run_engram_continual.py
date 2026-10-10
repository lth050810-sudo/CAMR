#!/usr/bin/env python3
"""Engram-inspired online replay router; this is not an Engram layer port.

Frozen Qwen base + LoRA, independent frozen MPNet, hash candidate retrieval,
and a constrained stochastic writer policy. Evaluation remains closed-book.
Run on the separate development protocol before making benchmark claims.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from baselines.common import (
    AnswerOnlyDataset, answer_scores, append_jsonl, atomic_json,
    environment_metadata, fixed_panel, generate_answers, group_by_stage,
    load_manifest, load_model, load_tokenizer, make_collator, manifest_panel,
    sha256_file, train_answer_only,
)
from baselines.engram_continual_core import ActionPolicy, FactMemory
from baselines.run_lora_family import evaluate_and_write, lora_model, set_seed
from baselines.run_rag import HFMeanPoolEncoder, fact_document


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, default=Path('Qwen3-8B-Base'))
    p.add_argument('--embedding-model', default='/mnt/yy2607004_hs/embedding_models/all-mpnet-base-v2')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mode', choices=['policy', 'rule', 'always_add', 'random_replay'], default='policy')
    p.add_argument('--fact-mode', choices=['structured', 'text'], default='structured')
    p.add_argument('--max-stages', type=int, default=0)
    p.add_argument('--examples-per-stage', type=int, default=0)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--eval-batch-size', type=int, default=32)
    p.add_argument('--learning-rate', type=float, default=2e-4)
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--max-new-tokens', type=int, default=16)
    p.add_argument('--lora-r', type=int, default=16)
    p.add_argument('--lora-alpha', type=int, default=32)
    p.add_argument('--lora-dropout', type=float, default=0.0)
    p.add_argument('--target-modules', default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')
    p.add_argument('--replay-k', type=int, default=1)
    p.add_argument('--replay-schedule', type=Path, help='Random baseline: match policy actions.jsonl replay count at every event.')
    p.add_argument('--probe-size', type=int, default=4)
    p.add_argument('--similarity-threshold', type=float, default=0.65)
    p.add_argument('--mastery-nll', type=float, default=0.25)
    p.add_argument('--hash-tables', type=int, default=8)
    p.add_argument('--hash-bits', type=int, default=8)
    p.add_argument('--max-candidates', type=int, default=64)
    p.add_argument('--policy-lr', type=float, default=0.01)
    p.add_argument('--retention-weight', type=float, default=1.0)
    p.add_argument('--cost-weight', type=float, default=0.01, help='Penalty per 128 trained sequence tokens.')
    p.add_argument('--locality-panel-size', type=int, default=32)
    p.add_argument('--multihop-panel-size', type=int, default=16)
    p.add_argument('--skip-base-eval', action='store_true')
    p.add_argument('--no-stage-eval', action='store_true')
    p.add_argument('--repeat-every', type=int, default=0, help='Diagnostic only: repeat a past training QA after N original arrivals; never uses rephrase.')
    args = p.parse_args()
    positive = ['epochs', 'batch_size', 'eval_batch_size', 'max_length', 'max_new_tokens',
                'replay_k', 'hash_tables', 'hash_bits', 'max_candidates']
    if any(getattr(args, key) <= 0 for key in positive):
        p.error('Training, batch, replay and hash sizes must be positive.')
    if any(getattr(args, key) < 0 for key in ['max_stages', 'examples_per_stage', 'probe_size',
                                             'locality_panel_size', 'multihop_panel_size', 'repeat_every']):
        p.error('Limits cannot be negative.')
    if args.replay_schedule and args.mode != 'random_replay':
        p.error('--replay-schedule is only valid for random_replay')
    return args


def select_rows(rows, max_stages=0, examples_per_stage=0):
    stages = group_by_stage(sorted(rows, key=lambda row: (int(row['stage']), str(row['sample_id']))))
    keys = list(stages)[:max_stages or None]
    return [row for key in keys for row in stages[key][:examples_per_stage or None]]


def training_view(row):
    """Explicit firewall between training information and held-out probes."""
    keys = ('sample_id', 'stage', 'update', 'ans', 'subject', 'subject_id',
            'relation', 'relation_id', 'object', 'object_id', 'tag')
    return {key: row[key] for key in keys if key in row}


def make_events(rows, repeat_every=0):
    """Inject only old training QA, separately marked from natural arrivals."""
    events = []
    for index, row in enumerate(rows):
        events.append({'row': row, 'arrival_stage': row['stage'], 'diagnostic_repeat': False})
        if repeat_every and (index + 1) % repeat_every == 0:
            old = rows[max(0, index + 1 - repeat_every)]
            events.append({'row': old, 'arrival_stage': row['stage'], 'diagnostic_repeat': True})
    return events


def read_schedule(path, events):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) != len(events):
        raise ValueError('replay schedule length differs from this event stream')
    for index, (record, event) in enumerate(zip(records, events)):
        if (record['event'] != index or record['sample_id'] != event['row']['sample_id']
                or record['arrival_stage'] != event['arrival_stage']
                or record['diagnostic_repeat'] != event['diagnostic_repeat']):
            raise ValueError(f'replay schedule stream mismatch at event {index}')
        count = record['replay_count']
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError('scheduled replay count must be a nonnegative integer')
        action = record['action']
        if action not in {'add', 'merge', 'ignore'}:
            raise ValueError('invalid scheduled action')
        if (action == 'merge') != (count > 0):
            raise ValueError('scheduled merge requires replay; add/ignore must have none')
    return records


@torch.inference_mode()
def answer_nll(model, tokenizer, rows, device, max_length):
    if not rows:
        return [], 0
    dataset = AnswerOnlyDataset(rows, tokenizer, max_length)
    if len(dataset) != len(rows):
        raise ValueError('A policy reward example has no answer tokens after truncation')
    batch = {key: value.to(device) for key, value in make_collator(tokenizer)(list(dataset)).items()}
    model.eval()
    logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False).logits
    labels = batch['labels'][:, 1:]
    losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), labels, reduction='none', ignore_index=-100)
    nll = losses.sum(1) / (labels != -100).sum(1).clamp_min(1)
    values = nll.cpu().tolist()
    if not all(math.isfinite(value) for value in values):
        raise ValueError('nonfinite reward loss')
    return values, int(batch['attention_mask'].sum())


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty run: {args.output_dir}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the Qwen training pilot')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    torch.set_num_threads(4)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    all_rows, metadata = load_manifest(args.manifest)
    rows = select_rows(all_rows, args.max_stages, args.examples_per_stage)
    if not rows:
        raise ValueError('empty training selection')
    events = make_events(rows, args.repeat_every)
    schedule = read_schedule(args.replay_schedule, events) if args.replay_schedule else None
    # Panels may include future facts; they are evaluation-only and never passed to the writer.
    locality = manifest_panel(metadata, 'locality', limit=args.locality_panel_size)
    multihop = manifest_panel(metadata, 'mhop', limit=args.multihop_panel_size)
    if locality is None:
        locality = fixed_panel(rows, 'loc', 'loc_ans', args.locality_panel_size, args.seed + 1001)
    if multihop is None:
        multihop = fixed_panel(rows, 'mhop', 'mhop_ans', args.multihop_panel_size, args.seed + 2001)
    started = time.perf_counter()
    encoder = HFMeanPoolEncoder(args.embedding_model, device=device, batch_size=32,
                               max_length=args.max_length, query_prefix='', document_prefix='', local_files_only=True)
    tokenizer = load_tokenizer(args.model_dir)
    model = load_model(args.model_dir, device)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({
        'manifest_sha256': sha256_file(args.manifest), 'protocol_id': metadata.get('protocol_id'),
        'sample_ids': [row['sample_id'] for row in rows], 'num_facts': len(rows), 'num_events': len(events),
        'closed_book_evaluation': True, 'base_weights_frozen': True,
        'encoder': encoder.provenance, 'oracle_structure_access': args.fact_mode == 'structured',
        'equivalence_scope': 'oracle subject/relation/object identity + exact normalized answer; aliases conservatively unresolved' if args.fact_mode == 'structured' else 'exact normalized training QA only',
        'source_sha256': {name: sha256_file(Path(__file__).parent / name)
                          for name in ['engram_continual_core.py', 'run_engram_continual.py']},
        'replication_scope': 'Engram-inspired external replay writer, not original Engram or a Qwen intermediate-layer module',
        'reward': 'current answer NLL decrease + retention_weight * non-replayed history answer NLL decrease - cost_weight * trained sequence tokens / 128',
        'policy_scope': 'online contextual bandit on training-visible QA; immediate surrogate reward, no long-horizon RL claim',
        'memory_policy': 'full seen unique memory; no eviction; same-slot unequal facts excluded from replay for this arrival, not automatically superseded',
        'budget_note': 'matched replay schedule matches counts and optimizer steps; actual tokens may differ and are logged',
        'stream_is_diagnostic': bool(args.repeat_every), 'split_audit': metadata.get('split_audit'),
        **environment_metadata(),
    })
    atomic_json(args.output_dir / 'run_config.json', config)
    atomic_json(args.output_dir / 'selected_stream.json', rows)
    metrics = []
    if not args.skip_base_eval:
        metrics.append(evaluate_and_write(model, tokenizer, rows, locality, multihop, device, args, 'base', None))
        atomic_json(args.output_dir / 'metrics.json', metrics)
    model = lora_model(model, args)
    bad = [name for name, parameter in model.named_parameters() if parameter.requires_grad and 'lora_' not in name]
    if bad:
        raise RuntimeError(f'Unexpected trainable base parameters: {bad[:3]}')
    memory = FactMemory(int(encoder.model.config.hidden_size), seed=args.seed,
                        tables=args.hash_tables, bits=args.hash_bits, max_candidates=args.max_candidates,
                        similarity_threshold=args.similarity_threshold, fact_mode=args.fact_mode)
    policy = ActionPolicy(seed=args.seed, lr=args.policy_lr)
    replay_rng = random.Random(args.seed + 9001)
    probe_rng = random.Random(args.seed + 9002)
    optimizer = None
    train_log, action_counts, seen_rows = [], Counter(), []
    embeddings_by_id = {}
    event_index = 0
    for stage, current_rows in group_by_stage(rows).items():
        encoder_started = time.perf_counter()
        # Pure frozen encoding; only insert each row after its learning decision completes.
        vectors = encoder.encode([fact_document(training_view(row)) for row in current_rows])
        embeddings_by_id.update({row['sample_id']: vector for row, vector in zip(current_rows, vectors)})
        stage_events = [event for event in events if event['arrival_stage'] == stage]
        stage_stats = {'stage': stage, 'current_count': len(current_rows), 'events': len(stage_events),
                       'encoder_seconds': time.perf_counter() - encoder_started, 'optimizer_steps': 0,
                       'sequence_tokens': 0, 'answer_tokens': 0, 'reward_forward_tokens': 0,
                       'train_seconds': 0.0, 'router_reward_seconds': 0.0, 'replay_count': 0}
        stage_counts = Counter()
        for event in stage_events:
            event_started = time.perf_counter()
            row = training_view(event['row'])
            vector = embeddings_by_id[row['sample_id']]
            query = memory.query(row, vector)
            before, reward_tokens = answer_nll(model, tokenizer, [row], device, args.max_length)
            allow_ignore = bool(query['duplicate_rows']) and before[0] <= args.mastery_nll
            mastery_prediction = None
            if allow_ignore:
                mastery_prediction = generate_answers(model, tokenizer, [row['update']], device,
                                                       batch_size=1, max_length=args.max_length,
                                                       max_new_tokens=args.max_new_tokens)[0]
                allow_ignore = answer_scores(mastery_prediction, row['ans'])['match']
            ambiguous = {r['sample_id'] for r in query.get('ambiguous_rows', [])}
            eligible_random = [old for old in memory.rows
                               if old['sample_id'] != row['sample_id'] and old['sample_id'] not in ambiguous]
            allow_merge = bool(query['similar_rows'])
            decision = policy.choose([query['top_similarity'], float(bool(query['duplicate_rows'])),
                                      min(before[0] / 10, 1.0), min(len(memory.rows) / 1000, 1.0)],
                                     allow_ignore=allow_ignore, allow_merge=allow_merge, mode=args.mode)
            replay = []
            if args.mode == 'random_replay':
                count = schedule[event_index]['replay_count'] if schedule else min(args.replay_k, len(eligible_random))
                if count > len(eligible_random):
                    raise ValueError('Insufficient eligible history to match scheduled replay count')
                action = 'merge' if count else 'add'
                if schedule and schedule[event_index]['action'] == 'ignore':
                    if not allow_ignore:
                        raise ValueError('Cannot copy a scheduled ignore when this model has not mastered the fact')
                    action = 'ignore'
                decision = {'action': action, 'probabilities': {a: float(a == action) for a in ['ignore', 'merge', 'add']}, 'log_prob': None}
                replay = replay_rng.sample(eligible_random, count)
            elif decision['action'] == 'merge':
                replay = query['similar_rows'][:args.replay_k]
            replay_ids = {old['sample_id'] for old in replay}
            # Probes are past training QA disjoint from replay; no held-out evaluation targets.
            probe_pool = [old for old in eligible_random if old['sample_id'] not in replay_ids]
            probes = probe_rng.sample(probe_pool, min(args.probe_size, len(probe_pool)))
            old_before, tokens = answer_nll(model, tokenizer, probes, device, args.max_length)
            reward_tokens += tokens
            stats = None
            if decision['action'] != 'ignore':
                stats, optimizer = train_answer_only(
                    model, tokenizer, [row] + replay, device, seed=args.seed + event_index,
                    epochs=args.epochs, batch_size=args.batch_size, gradient_accumulation_steps=1,
                    learning_rate=args.learning_rate, weight_decay=0.0, max_grad_norm=1.0,
                    max_length=args.max_length, optimizer=optimizer)
            after, tokens = answer_nll(model, tokenizer, [row] + probes, device, args.max_length)
            reward_tokens += tokens
            trained_tokens = stats.sequence_tokens if stats else 0
            current_gain, retention_gain = before[0] - after[0], mean(old_before) - mean(after[1:])
            reward = current_gain + args.retention_weight * retention_gain - args.cost_weight * trained_tokens / 128
            policy_update = policy.learn(decision, reward) if args.mode == 'policy' else None
            # Consolidate duplicate facts; they are refreshed in LoRA when mastery fails.
            if not query['duplicate_rows']:
                memory.insert(row, vector)
            action_counts[decision['action']] += 1
            stage_counts[decision['action']] += 1
            trace = {
                'event': event_index, 'sample_id': row['sample_id'], 'arrival_stage': stage,
                'diagnostic_repeat': event['diagnostic_repeat'], 'action': decision['action'],
                'probabilities': decision['probabilities'], 'candidate_count': query['candidate_count'],
                'top_similarity': query['top_similarity'], 'duplicate_count': len(query['duplicate_rows']),
                'ambiguous_count': len(ambiguous), 'allow_ignore': allow_ignore, 'allow_merge': allow_merge,
                'mastery_prediction': mastery_prediction, 'current_nll_before': before[0], 'current_nll_after': after[0],
                'refresh_unmastered_duplicate': bool(query['duplicate_rows']) and not allow_ignore,
                'replay_count': len(replay), 'replay_sample_ids': [old['sample_id'] for old in replay],
                'probe_sample_ids': [old['sample_id'] for old in probes],
                'current_gain': current_gain, 'retention_gain': retention_gain, 'reward': reward,
                'policy_update': policy_update, 'reward_forward_tokens': reward_tokens,
                'train': asdict(stats) if stats else None,
                'event_seconds': time.perf_counter() - event_started,
            }
            append_jsonl(args.output_dir / 'actions.jsonl', [trace])
            stage_stats['reward_forward_tokens'] += reward_tokens
            stage_stats['replay_count'] += len(replay)
            if stats:
                for key in ['optimizer_steps', 'sequence_tokens', 'answer_tokens']:
                    stage_stats[key] += getattr(stats, key)
                stage_stats['train_seconds'] += stats.wall_seconds
            stage_stats['router_reward_seconds'] += trace['event_seconds'] - (stats.wall_seconds if stats else 0)
            event_index += 1
            if event_index % 8 == 0:
                print(json.dumps({'events_done': event_index, 'total_events': len(events), 'actions': dict(action_counts)}, ensure_ascii=False), flush=True)
        seen_rows.extend(current_rows)
        stage_stats['action_counts'] = dict(stage_counts)
        stage_stats['memory_count'] = len(memory.rows)
        train_log.append(stage_stats)
        atomic_json(args.output_dir / 'training_log.json', train_log)
        if not args.no_stage_eval or stage == max(group_by_stage(rows)):
            result = evaluate_and_write(model, tokenizer, seen_rows, locality, multihop, device, args, f'tau{stage}', stage)
            result['train'] = stage_stats
            metrics.append(result)
            atomic_json(args.output_dir / 'metrics.json', metrics)
            print(json.dumps({'stage': stage, 'rephrase_accuracy': result['rephrase_accuracy'], 'past_rephrase_retention': result.get('past_rephrase_retention')}, ensure_ascii=False), flush=True)
    adapter_dir = args.output_dir / 'adapter'
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    torch.save(policy.state_dict(), args.output_dir / 'policy.pt')
    torch.save({'rows': memory.rows, 'embeddings': torch.stack([embeddings_by_id[row['sample_id']] for row in memory.rows]),
                'seed': args.seed, 'config': config}, args.output_dir / 'memory.pt')
    atomic_json(args.output_dir / 'completed.json', {
        'status': 'complete', 'method': 'engram_inspired_' + args.mode, 'events': event_index,
        'final_stage': max(group_by_stage(rows)), 'elapsed_seconds': time.perf_counter() - started,
        'action_counts': dict(action_counts), 'trained_sequence_tokens': sum(r['sequence_tokens'] for r in train_log),
        'optimizer_steps': sum(r['optimizer_steps'] for r in train_log),
    })


if __name__ == '__main__':
    main()
