"""Full-parameter on-policy SDFT with an EMA teacher and frozen WikiBigEdit QA.

Ports the released forward KL, sequence reduction, on-policy trajectories and
EMA update into the existing FP32-master FSDP runtime. vLLM is unnecessary for
short QA trajectories; generation and learning use the same student weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

from baselines.common import (AXES, answer_scores, append_jsonl, atomic_json, axis_fields,
                              group_by_stage, load_manifest, manifest_panel, sha256_file)
from baselines.adaptation_provenance import implementation_identity
from baselines.engram_full_backend import FullParameterBackend, _build_model
from baselines.engram_grad_norm import clip_full_shard_grad_norm_
from baselines.sdft_qwen_core import (completion_logits, encode_prompts, forward_kl_per_sequence,
                                      teacher_prompt, update_ema)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, default=Path('/root/engram/Qwen3-8B-Base'))
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--source-dir', type=Path,
                   default=Path('/root/engram/external/recent_extra/SDFT-source'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--warmup-ratio', type=float, default=0.1)
    p.add_argument('--prompts-per-update', type=int, default=32)
    p.add_argument('--micro-batch-size', type=int, default=8)
    p.add_argument('--ema-alpha', type=float, default=0.01)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--rollout-tokens', type=int, default=16)
    p.add_argument('--loss-skip-tokens', type=int, default=0)
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--teacher-max-length', type=int, default=256)
    p.add_argument('--max-new-tokens', type=int, default=16)
    p.add_argument('--eval-batch-size', type=int, default=32)
    p.add_argument('--activation-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--flash-attention', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--save-checkpoint', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--max-facts', type=int, default=0)
    p.add_argument('--distributed-timeout-minutes', type=int, default=30)
    a = p.parse_args()
    if min(a.prompts_per_update, a.micro_batch_size, a.rollout_tokens) < 1:
        p.error('Positive sampling/training budgets required')
    if not 0 < a.ema_alpha <= 1 or not 0 <= a.warmup_ratio < 1 or a.temperature <= 0:
        p.error('Invalid EMA, warmup or temperature')
    a.parameter_mode = 'full'
    a.training_views = 1
    a.training_loss = 'token_mean'  # Backend SFT loss is not used; see forward_kl_per_sequence.
    a.current_loss_weight = 0.5
    return a


@torch.no_grad()
def sample_student(backend, prompts, tokens, temperature, counter):
    backend.model.eval()
    input_ids, attention = prompts['input_ids'], prompts['attention_mask']
    active = torch.ones(input_ids.shape[0], device=backend.device, dtype=torch.bool)
    outputs, masks = [], []
    eos = backend.tokenizer.eos_token_id
    for _ in range(tokens):
        logits = backend.model(input_ids=input_ids, attention_mask=attention,
            position_ids=(attention.cumsum(-1) - 1).clamp(min=0),
            logits_to_keep=1, use_cache=False).logits[:, -1].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite rollout logits')
        next_token = torch.empty((input_ids.shape[0],), dtype=torch.long, device=backend.device)
        if backend.rank == 0:
            next_token.copy_(torch.multinomial(F.softmax(logits / temperature, -1), 1).squeeze(-1))
        dist.broadcast(next_token, 0)
        next_token = torch.where(active, next_token, torch.full_like(next_token, eos))
        masks.append(active.long())
        outputs.append(next_token)
        counter['student_rollout_forwards'] += 1
        counter['student_rollout_sequence_tokens'] += int(attention.sum())
        input_ids = torch.cat((input_ids, next_token[:, None]), -1)
        attention = torch.cat((attention, active.long()[:, None]), -1)
        active = active & (next_token != eos)
        if not bool(active.any()):
            break
    return torch.stack(outputs, -1), torch.stack(masks, -1)


def evaluate(backend, rows, locality, multihop):
    stage = max(r['stage'] for r in rows)
    checkpoint = f'tau{stage}'
    result = {'checkpoint': checkpoint, 'current_stage': stage, 'num_facts': len(rows)}
    predictions = []
    for axis in AXES:
        panel = locality if axis == 'locality' else multihop if axis == 'multihop' else rows
        q, answer = axis_fields(axis)
        valid = [r for r in panel if r.get(q) and r.get(answer)]
        values = backend.generate([r[q] for r in valid])
        records = [{'checkpoint': checkpoint, 'sample_id': r['sample_id'], 'fact_stage': r['stage'],
                    'axis': axis, 'question': r[q], 'gold': r[answer], 'prediction': value,
                    **answer_scores(value, r[answer])} for r, value in zip(valid, values)]
        predictions.extend(records)
        exact, match = sum(r['exact'] for r in records), sum(r['match'] for r in records)
        result[axis] = {'count': len(records), 'exact_count': exact, 'match_count': match,
                        'exact_accuracy': exact / len(records), 'match_accuracy': match / len(records)}
        result[axis + '_accuracy'] = result[axis]['match_accuracy']
    return result, predictions


def main():
    a = parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError('Refusing to overwrite an earlier run')
    rows, manifest = load_manifest(a.manifest)
    if a.max_facts:
        rows = rows[:a.max_facts]
    locality = manifest_panel(manifest, 'locality', limit=512)
    multihop = manifest_panel(manifest, 'mhop', limit=512)
    if a.max_facts:
        locality, multihop = locality[:2], multihop[:2]
    source_path = a.source_dir / 'SOURCE.json'
    source = json.loads(source_path.read_text())
    for name, expected in source['file_sha256'].items():
        if sha256_file(source_path.parent / name) != expected:
            raise ValueError('Pinned SDFT source changed: ' + name)
    torch.set_num_threads(4)
    backend = FullParameterBackend(a)
    teacher, _, _ = _build_model(a, backend.device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    # Both independent FSDP models start from the identical original base checkpoint.
    update_ema(backend.model, teacher, 1.0)
    torch.manual_seed(a.seed + backend.rank)
    torch.cuda.manual_seed(a.seed + backend.rank)
    stages = group_by_stage(rows)
    total_updates = sum(math.ceil(len(group) / a.prompts_per_update) for group in stages.values())
    scheduler = get_cosine_schedule_with_warmup(backend.optimizer,
        num_warmup_steps=math.ceil(total_updates * a.warmup_ratio), num_training_steps=total_updates)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(method='SDFT-forward-KL-Qwen3-Base', source=source, num_facts=len(rows),
                  implementation=implementation_identity('sdft'),
                  manifest_sha256=sha256_file(a.manifest), backend=backend.metadata,
                  epochs=1, num_generations=1, num_iterations=1, total_optimizer_steps=total_updates,
                  teacher='EMA student, conditioned on the current training question and demonstration answer',
                  generation_source='student without demonstration', kl_direction='teacher || student',
                  alpha=0.0, beta=0.0, top_p=1.0, temperature=a.temperature,
                  training_fields=['update', 'ans'], history_replay=False, closed_book_evaluation=True,
                  training_objective='Per-sequence mean forward KL on sampled student completion tokens, then batch mean',
                  teacher_prompt_template=teacher_prompt('<question>', '<training answer>'),
                  adaptations=[f'FP32-master FSDP full-parameter student and sharded EMA teacher on {backend.world_size} GPUs.',
                               'Single chronological pass through the fixed 3K; no history or held-out training fields.',
                               'Base QA completion prompts; short demonstration prompt instead of chat/thinking template.',
                               '16 sampled completion tokens; no skipped initial tokens because Base has no chat/thinking prefix.',
                               'Exact student sampling instead of vLLM; no off-engine importance correction is needed.',
                               'Same official LR2e-5, batch32, warmup10%, cosine schedule, EMA alpha0.01 and one update per trajectory.'])
    if backend.rank == 0:
        a.output_dir.mkdir(parents=True)
        atomic_json(a.output_dir / 'run_config.json', config)
    dist.barrier()
    counter = Counter()
    done_facts = 0
    logs = []
    initial_fingerprint = backend.parameter_fingerprint()
    started = time.time()
    for stage, stage_rows in stages.items():
        for start in range(0, len(stage_rows), a.prompts_per_update):
            group = stage_rows[start:start + a.prompts_per_update]
            backend.optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            trajectory_records = []
            microbatches = []
            for sub in range(0, len(group), a.micro_batch_size):
                part = group[sub:sub + a.micro_batch_size]
                prompt = {k: v.to(backend.device) for k, v in
                          encode_prompts(backend.tokenizer, part, max_length=a.max_length).items()}
                demonstration = {k: v.to(backend.device) for k, v in encode_prompts(
                    backend.tokenizer, part, teacher=True, max_length=a.teacher_max_length).items()}
                completion, mask = sample_student(backend, prompt, a.rollout_tokens, a.temperature, counter)
                with torch.no_grad():
                    teacher_logits = completion_logits(teacher, demonstration, completion, mask)
                    teacher_log_probs = F.log_softmax(teacher_logits / a.temperature, -1)
                microbatches.append((prompt, completion, mask, teacher_log_probs))
                counter['teacher_forwards'] += 1
                counter['completion_tokens'] += int(mask.sum())
                counter['teacher_sequence_tokens'] += int(demonstration['attention_mask'].sum() + mask.sum())
                counter['student_learning_sequence_tokens'] += int(prompt['attention_mask'].sum() + mask.sum())
                for index, row in enumerate(part):
                    length = int(mask[index].sum())
                    token_ids = completion[index, :length].tolist()
                    trajectory_records.append({'sample_id': row['sample_id'], 'stage': stage,
                        'optimizer_step': backend.total_optimizer_steps + 1,
                        'student_prompt': backend.tokenizer.decode(prompt['input_ids'][index][
                            prompt['attention_mask'][index].bool()].tolist(), skip_special_tokens=False),
                        'teacher_prompt': teacher_prompt(row['update'], row['ans']),
                        'completion_ids': token_ids, 'completion': backend.tokenizer.decode(token_ids),
                        'generation_source': 'current_student', 'loss_tokens': max(0, length - a.loss_skip_tokens)})
                del teacher_logits
            # All trajectories use exactly the pre-update student. Finish generation
            # before accumulating FSDP gradients, so evaluation never intervenes in
            # a partially accumulated training update.
            backend.model.train()
            for prompt, completion, mask, teacher_log_probs in microbatches:
                logits = completion_logits(backend.model, prompt, completion, mask)
                losses = forward_kl_per_sequence(logits / a.temperature, teacher_log_probs,
                                                 mask, a.loss_skip_tokens)
                loss = losses.sum() / len(group)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite SDFT loss')
                loss.backward()
                loss_value += float(loss.detach())
                counter['student_backward_forwards'] += 1
                del logits, loss, losses
            del microbatches, teacher_log_probs
            grad_norm = float(clip_full_shard_grad_norm_(backend.model, a.max_grad_norm))
            if not math.isfinite(grad_norm):
                raise FloatingPointError('Nonfinite student gradient')
            lr = backend.optimizer.param_groups[0]['lr']
            backend.optimizer.step()
            backend.total_optimizer_steps += 1
            scheduler.step()
            backend.optimizer.zero_grad(set_to_none=True)
            update_ema(backend.model, teacher, a.ema_alpha)
            counter['ema_updates'] += 1
            done_facts += len(group)
            log = {'stage': stage, 'optimizer_step': backend.total_optimizer_steps,
                   'sample_ids': [r['sample_id'] for r in group], 'new_facts': len(group),
                   'facts_completed': done_facts, 'loss': loss_value, 'gradient_norm': grad_norm,
                   'learning_rate': lr, 'ema_alpha': a.ema_alpha, 'costs': dict(counter)}
            logs.append(log)
            if backend.rank == 0:
                append_jsonl(a.output_dir / 'trajectories.jsonl', trajectory_records)
                append_jsonl(a.output_dir / 'actions.jsonl', [log])
                atomic_json(a.output_dir / 'progress.json', {'status': 'training', 'stage': stage,
                    'facts_completed': done_facts, 'num_facts': len(rows),
                    'optimizer_steps': backend.total_optimizer_steps, 'total_optimizer_steps': total_updates,
                    'elapsed_seconds': time.time() - started})
                print(json.dumps({k: v for k, v in log.items() if k not in ('sample_ids', 'costs')}), flush=True)
    final_fingerprint = backend.parameter_fingerprint()
    if initial_fingerprint == final_fingerprint:
        raise RuntimeError('SDFT did not change the student parameters')
    if backend.rank == 0:
        atomic_json(a.output_dir / 'progress.json', {'status': 'evaluating', 'facts_completed': done_facts})
    result, predictions = evaluate(backend, rows, locality, multihop)
    if backend.rank == 0:
        atomic_json(a.output_dir / 'metrics.json', [result])
        append_jsonl(a.output_dir / 'predictions.jsonl', predictions)
    if a.save_checkpoint:
        backend.save_checkpoint(a.checkpoint_dir / 'student_final', {'configuration': config, 'costs': dict(counter)})
        # Persist local FP32 EMA shards; final completion waits for every rank.
        a.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save({name: p.detach().cpu() for name, p in teacher.named_parameters()},
                   a.checkpoint_dir / f'teacher_rank{backend.rank}.pt')
    dist.barrier()
    if backend.rank == 0:
        atomic_json(a.output_dir / 'completed.json', {'status': 'complete', 'method': config['method'],
            'num_facts': done_facts, 'epochs': 1, 'optimizer_steps': backend.total_optimizer_steps,
            'student_parameter_fingerprint_before': initial_fingerprint,
            'student_parameter_fingerprint_after': final_fingerprint,
            'costs': dict(counter), 'world_size': backend.world_size,
            'replicated_physical_training_costs': {
                key: value * backend.world_size for key, value in counter.items()
                if key.endswith('_tokens') or key.endswith('_forwards')},
            'evaluation_costs': backend.forward_accounting,
            'checkpoint_dir': str(a.checkpoint_dir) if a.save_checkpoint else None,
            'smoke_only': bool(a.max_facts), 'elapsed_seconds': time.time() - started,
            'peak_vram_gib_rank0': torch.cuda.max_memory_allocated() / 2**30})
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
