"""Eight-rank FSDP, single-current-pass baselines on one frozen long stream."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import time

from baselines.longstream_protocol import PREFIXES, atomic, blocks, digest, evaluate

FIELDS = ('optimizer_steps', 'sequence_tokens', 'answer_tokens', 'physical_sequence_tokens',
          'physical_answer_tokens', 'physical_padded_sequence_tokens', 'current_sequence_tokens',
          'replay_sequence_tokens', 'current_answer_tokens', 'replay_answer_tokens',
          'current_fact_presentations', 'replay_fact_presentations', 'current_view_presentations',
          'replay_view_presentations', 'wall_seconds')


def packed_batches(indices, rows, max_batch, max_padded_tokens):
    batch, width = [], 0
    for i in indices:
        length = rows[i]['original_training_sequence_tokens']
        new_width = max(width, length)
        if batch and (len(batch) >= max_batch or new_width * (len(batch) + 1) > max_padded_tokens):
            yield batch
            batch, width = [], 0
        batch.append(i)
        width = max(width, length)
    if batch:
        yield batch


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, default=Path('/root/engram/Qwen3-8B-Base'))
    p.add_argument('--method', choices=('cumulative_lora', 'random_lora', 'sequential_full', 'joint_full'), required=True)
    p.add_argument('--fsdp-sharding-strategy', choices=['full_shard', 'shard_grad_op'], default='full_shard')
    p.add_argument('--control', type=Path)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, choices=(1,), default=1)
    p.add_argument('--global-batch-size', type=int, default=8)
    p.add_argument('--max-padded-tokens', type=int, default=4096)
    p.add_argument('--block-size', type=int, default=375)
    p.add_argument('--memory-capacity', type=int, default=1000)
    p.add_argument('--target', type=int, default=100000)
    p.add_argument('--benchmark-steps', type=int, default=0)
    p.add_argument('--benchmark-online', action='store_true')
    p.add_argument('--benchmark-checkpoint', action='store_true')
    p.add_argument('--eval-batch-size', type=int, default=64)
    p.add_argument('--learning-rate', type=float)
    p.add_argument('--training-loss', choices=('token_mean',), default='token_mean',
                   help='Formal baselines use ordinary mean CE over all answer tokens in each batch.')
    p.add_argument('--training-views', type=int, choices=(1,), default=1,
                   help='Formal baselines train only the original question.')
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--max-new-tokens', type=int, default=16)
    p.add_argument('--activation-checkpointing', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--flash-attention', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume-checkpoint', type=Path)
    p.add_argument('--skip-base-eval', action='store_true')
    a = p.parse_args(argv)
    if a.method == 'cumulative_lora' and a.control:
        control = json.loads(a.control.read_text())
        if 'cumulative_lora_block_size' in control:
            a.block_size = control['cumulative_lora_block_size']
            if not isinstance(a.block_size, int) or isinstance(a.block_size, bool):
                p.error('Cumulative LoRA block size must be an integer')
    a.parameter_mode = 'lora' if a.method.endswith('lora') else 'full'
    a.lora_rank, a.lora_alpha, a.lora_dropout = 16, 32, 0.05
    a.training_views_file = None
    # Kept for backend metadata compatibility; token_mean does not use this weight.
    a.current_loss_weight = 0.5
    if a.benchmark_online:
        if a.benchmark_steps <= 0:
            p.error('--benchmark-online requires positive --benchmark-steps; it is not a formal baseline objective')
        a.training_views, a.training_loss = 2, 'role_mean'
    a.learning_rate = a.learning_rate or (2e-4 if a.parameter_mode == 'lora' else 1e-5)
    a.distributed_timeout_minutes = 120
    if min(a.global_batch_size, a.block_size, a.target, a.memory_capacity) < 1:
        p.error('Batch, block, target and memory sizes must be positive')
    if a.benchmark_steps < 0:
        p.error('--benchmark-steps must be nonnegative')
    return a


def validate_baseline_objective(a, backend_metadata):
    """Fail closed if a formal baseline inherits an online-method objective."""
    online_benchmark = a.benchmark_online and a.benchmark_steps > 0
    expected_loss, expected_views = ('role_mean', 2) if online_benchmark else ('token_mean', 1)
    if (a.training_loss, a.training_views) != (expected_loss, expected_views):
        raise ValueError('Formal block baselines require token_mean and one original training view')
    if (backend_metadata.get('training_loss'), backend_metadata.get('training_views')) != (expected_loss, expected_views):
        raise ValueError('Backend objective differs from the explicit baseline configuration')


def run(a, driver):
    from baselines.common import append_jsonl, sha256_file
    validate_baseline_objective(a, driver.backend.metadata)
    payload = json.loads(a.manifest.read_text())
    if payload.get('schema_version') != 'wikibigedit-longstream/v1':
        raise ValueError('Expected frozen longstream manifest')
    rows = payload['records']
    if [r['sample_id'] for r in rows] != payload['selection']['sample_ids']:
        raise ValueError('Frozen stream IDs differ')
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(manifest_sha256=sha256_file(a.manifest), backend=driver.backend.metadata,
                  objective_scope=('synthetic_online_benchmark' if a.benchmark_online
                                   else 'formal_answer_token_ce_baseline'),
                  unique_current_passes=1, optimizer_continues_across_blocks=True,
                  batch_strategy='replicated logical global batch across FSDP ranks; physical cost counts all replicas',
                  temporal_order=f'{a.block_size}-new-fact blocks split at native stages and exact prefixes; shuffling only within current block + eligible past',
                  source_sha256=sha256_file(Path(__file__)))
    atomic(a.output_dir / 'run_config.json', config)
    totals, metrics, memory = Counter(), [], []
    start_block, seen, elapsed_prior = 0, 0, 0.0
    rng = random.Random(a.seed + 3001)
    if a.resume_checkpoint:
        state = driver.load_checkpoint(a.resume_checkpoint)
        if state['signature'] != digest({k: v for k, v in config.items() if k not in ('output_dir', 'checkpoint_dir', 'resume_checkpoint', 'backend')}):
            raise ValueError('Resume configuration differs')
        start_block, seen, elapsed_prior = state['next_block'], state['seen'], state['elapsed_seconds']
        totals, metrics, memory = Counter(state['totals']), state['metrics'], state['memory']
        rng.setstate(state['rng'])
    started = time.monotonic()
    def elapsed():
        return elapsed_prior + time.monotonic() - started
    def target():
        shared = int(json.loads(a.control.read_text())['target_events']) if a.control else len(rows)
        if shared not in (50000, 100000):
            raise ValueError('Invalid shared target')
        return min(shared, a.target, len(rows))
    def evaluate_at(cursor, name, full=False, base=False):
        summary, predictions = evaluate(driver, rows, payload, cursor, name, full_prefix=full, base=base)
        summary['cumulative_cost'] = {**dict(totals), 'unique_new_events': seen, 'elapsed_seconds': elapsed(),
                                      'gpu_hours': elapsed() * driver.backend.world_size / 3600,
                                      'forward_accounting': dict(driver.backend.forward_accounting)}
        metrics.append(summary)
        append_jsonl(a.output_dir / 'predictions.jsonl', predictions)
        atomic(a.output_dir / 'metrics.json', metrics)
    if a.benchmark_steps:
        measured, batch_sizes, peaks = [], [], []
        for step in range(a.benchmark_steps + 5):
            # Stratified lengths sampled across the full frozen stream.
            indices = random.Random(a.seed + step).sample(range(len(rows)), a.global_batch_size)
            if not a.benchmark_online:
                if step == 0:
                    indices = sorted(range(len(rows)), key=lambda i: rows[i]['original_training_sequence_tokens'], reverse=True)[:a.global_batch_size]
                indices = next(packed_batches(indices, rows, a.global_batch_size, a.max_padded_tokens))
            batch = [{k: rows[i][k] for k in ('sample_id', 'update', 'ans')} for i in indices]
            for i, r in enumerate(batch):
                r['_stream_role'] = 'current' if not a.benchmark_online or i == 0 else 'replay'
            t = time.monotonic()
            if a.benchmark_online:
                driver.answer_nll(batch[:5])
            stat = driver.train(batch, 1)
            if a.benchmark_online:
                driver.answer_nll(batch[:5])
            if step >= 5:
                measured.append(time.monotonic() - t)
                batch_sizes.append(len(batch))
            peaks.append(stat['peak_vram_gb'])
        sample = rows[:a.eval_batch_size]
        t = time.monotonic()
        driver.generate([r['update'] for r in sample])
        eval_seconds = time.monotonic() - t
        save_seconds, checkpoint_bytes = 0.0, 0
        if a.benchmark_checkpoint:
            t = time.monotonic()
            path = a.checkpoint_dir / 'benchmark'
            driver.save_checkpoint(path, {'benchmark': True})
            save_seconds = time.monotonic() - t
            checkpoint_bytes = sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
        atomic(a.output_dir / 'benchmark.json', {'steps': len(measured), 'step_seconds_mean': sum(measured) / len(measured),
            'step_seconds_samples': measured, 'eval_seconds_per_question': eval_seconds / len(sample),
            'checkpoint_seconds': save_seconds, 'checkpoint_bytes': checkpoint_bytes,
            'world_size': driver.backend.world_size, 'global_batch_size': a.global_batch_size,
            'mean_examples_per_step': sum(batch_sizes) / len(batch_sizes),
            'peak_train_vram_gib': max(peaks), 'max_padded_tokens': a.max_padded_tokens,
            'last_train': stat, 'elapsed_seconds': elapsed()})
        return
    if not a.skip_base_eval and not a.resume_checkpoint:
        evaluate_at(0, 'base', base=True)
    if a.method == 'joint_full':
        plan = [{'index': 0, 'stage': rows[target() - 1]['stage'], 'start': 0, 'end': target(),
                 'stage_end': True, 'prefix_end': True}]
    else:
        plan = list(blocks(rows, a.block_size))
    for b in plan[start_block:]:
        if b['end'] > target():
            break
        if b['start'] != seen:
            raise ValueError('Noncontiguous current-data block')
        current = list(range(b['start'], b['end']))
        if a.method == 'cumulative_lora':
            replay = list(range(b['start']))
        elif a.method == 'random_lora':
            replay = rng.sample(memory, min(len(current), len(memory)))
        else:
            replay = []
        if set(current) & set(replay) or any(i >= b['start'] for i in replay):
            raise ValueError('Replay contains current/future data')
        examples = current + replay
        random.Random(a.seed + b['index'] + 1).shuffle(examples)
        block_cost = Counter()
        block_start = time.monotonic()
        offset = 0
        for batch_index, indices in enumerate(packed_batches(examples, rows, a.global_batch_size, a.max_padded_tokens)):
            batch = [{**{k: rows[i][k] for k in ('sample_id', 'update', 'ans')},
                      '_stream_role': 'current' if i >= b['start'] else 'replay'} for i in indices]
            stat = driver.train(batch, 1)
            for key in FIELDS:
                block_cost[key] += stat.get(key, 0)
            totals['peak_vram_gb'] = max(totals['peak_vram_gb'], stat.get('peak_vram_gb', 0))
            if batch_index % 100 == 0:
                atomic(a.output_dir / 'progress.json', {'status': 'running', 'events': seen,
                    'current_block_end': b['end'], 'block_examples_processed': offset + len(indices),
                    'block_examples_total': len(examples), 'totals': dict(totals + block_cost),
                    'elapsed_seconds': elapsed(), 'target_events': target(), 'world_size': driver.backend.world_size})
            offset += len(indices)
        if block_cost['current_fact_presentations'] != len(current):
            raise ValueError('New facts were not presented exactly once')
        if block_cost['replay_fact_presentations'] != len(replay):
            raise ValueError('Replay accounting differs')
        for i in current:
            seen += 1
            if a.method == 'random_lora':
                if len(memory) < a.memory_capacity:
                    memory.append(i)
                else:
                    replace = rng.randrange(seen)
                    if replace < a.memory_capacity:
                        memory[replace] = i
        totals.update(block_cost)
        totals['replay_selections'] += len(replay)
        log = {**b, 'current_count': len(current), 'replay_count': len(replay),
               'current_ids_sha256': digest([rows[i]['sample_id'] for i in current]),
               'replay_ids_sha256': digest([rows[i]['sample_id'] for i in replay]),
               'replay_indices': replay if a.method == 'random_lora' else None,
               'replay_interval': [0, b['start']] if a.method == 'cumulative_lora' else None,
               'block_seconds': time.monotonic() - block_start, **dict(block_cost)}
        append_jsonl(a.output_dir / 'training_log.jsonl', [log])
        if b['stage_end'] or b['prefix_end']:
            name = f'prefix{seen:06d}' if seen in PREFIXES else f'tau{b["stage"]}'
            evaluate_at(seen, name, full=seen in PREFIXES)
            state = {'signature': digest({k: v for k, v in config.items() if k not in ('output_dir', 'checkpoint_dir', 'resume_checkpoint', 'backend')}),
                     'seen': seen, 'next_block': b['index'] + 1, 'totals': dict(totals),
                     'memory': memory, 'rng': rng.getstate(), 'metrics': metrics, 'elapsed_seconds': elapsed()}
            path = a.checkpoint_dir / f'event{seen:06d}'
            t = time.monotonic()
            driver.save_checkpoint(path, state)
            checkpoint_seconds = time.monotonic() - t
            checkpoint_bytes = sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
            append_jsonl(a.output_dir / 'checkpoint_costs.jsonl', [{'events': seen, 'path': str(path),
                'seconds': checkpoint_seconds, 'bytes': checkpoint_bytes}])
            totals['checkpoint_seconds'] += checkpoint_seconds
            totals['checkpoint_bytes'] += checkpoint_bytes
            metrics[-1]['cumulative_cost'].update(elapsed_seconds=elapsed(),
                gpu_hours=elapsed() * driver.backend.world_size / 3600,
                checkpoint_seconds=totals['checkpoint_seconds'], checkpoint_bytes=totals['checkpoint_bytes'])
            atomic(a.output_dir / 'metrics.json', metrics)
            atomic(a.output_dir / 'latest_checkpoint.json', {'path': str(path), 'event_cursor': seen})
        print(json.dumps({'method': a.method, 'events': seen, 'target': target(), 'block': b['index'],
                          'current': len(current), 'replay': len(replay), 'elapsed_seconds': elapsed()}), flush=True)
    if seen != target() or totals['current_fact_presentations'] != seen:
        raise ValueError('Incomplete stream or current-data epoch violation')
    atomic(a.output_dir / 'completed.json', {'status': 'complete', 'events': seen, 'expected_events': target(),
        'num_facts': seen, 'unique_new_events': seen, 'epochs': 1, 'totals': dict(totals),
        'elapsed_seconds': elapsed(), 'gpu_hours': elapsed() * driver.backend.world_size / 3600,
        'backend': driver.backend.metadata, 'forward_accounting': driver.backend.forward_accounting})


def main(argv=None):
    import torch
    import torch.distributed as dist
    from baselines.engram_full_backend import FullParameterBackend
    from baselines.run_engram_full import BackendCoordinator, serve_backend
    a = parse_args(argv)
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(a.output_dir)
    torch.set_num_threads(4)
    backend = FullParameterBackend(a)
    if backend.world_size != 8:
        raise ValueError('Formal/benchmark jobs require exactly eight ranks')
    if backend.rank:
        serve_backend(backend)
    else:
        driver = BackendCoordinator(backend)
        try:
            run(a, driver)
        finally:
            driver.shutdown()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
