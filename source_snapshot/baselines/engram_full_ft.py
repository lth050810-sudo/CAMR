"""Full-parameter BF16 FSDP execution for the external Engram writer.

All public execution/checkpoint methods are collective: every rank calls them
with the same rows and configuration.  Model forwards always enter the root
FSDP wrapper, including greedy decoding.  FP32 parameter shards and AdamW
moments are retained while FSDP casts forward/backward work to BF16.

The ordinary-module constructor exists for small CPU tests.  Production model
loading is deliberately restricted to the project's Qwen3-8B-Base directory.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.checkpoint import (
    FileSystemReader, FileSystemWriter, load_sharded_optimizer_state_dict,
    load_state_dict, save_state_dict,
)
from torch.distributed.fsdp import (
    BackwardPrefetch, FullyShardedDataParallel as FSDP, MixedPrecision,
    ShardedOptimStateDictConfig, ShardedStateDictConfig, ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from baselines.common import (
    AXES, AnswerOnlyDataset, answer_scores, atomic_json, axis_fields,
    load_model, load_tokenizer, make_collator, qa_prompt,
)
from baselines.run_full_ft_fsdp import setup_distributed, summarize_checkpoint_records
from baselines.engram_grad_norm import clip_full_shard_grad_norm_


MODEL_DIR = Path('/root/engram/Qwen3-8B-Base')
CHECKPOINT_SCHEMA = 'engram-writer-full-ft/v1'


@dataclass
class FullFTTrainStats:
    """Update costs; gradient_norm_before_clip is the last optimizer step's norm."""
    loss: float
    optimizer_steps: int
    sequence_tokens: int
    answer_tokens: int
    wall_seconds: float
    peak_vram_gb: float
    effective_sequence_tokens: int
    effective_answer_tokens: int
    padding_examples: int
    parameter_update_max_abs: float
    gradient_norm_before_clip: float


def validate_model_dir(model_dir: Path | str) -> Path:
    path = Path(model_dir).resolve()
    if path != MODEL_DIR.resolve():
        raise ValueError(f'full-parameter writer requires {MODEL_DIR}; got {path}')
    return path


def _get(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _rng_state() -> dict[str, Any]:
    result = {'python': random.getstate(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        result['cuda'] = torch.cuda.get_rng_state()
    # NumPy's global RNG is distinct from writer-owned Generator instances.
    try:
        import numpy as np
        result['numpy'] = np.random.get_state()
    except ImportError:
        pass
    return result


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state:
        torch.cuda.set_rng_state(state['cuda'])
    if 'numpy' in state:
        import numpy as np
        np.random.set_state(state['numpy'])


def _torch_load(path: Path) -> Any:
    # These are locally written execution checkpoints, including Python RNGs.
    return torch.load(path, map_location='cpu', weights_only=False)


class FullFTBackend:
    def __init__(
        self, model: torch.nn.Module, tokenizer: Any, args: Any,
        device: torch.device, *, rank: int = 0, world_size: int = 1,
        model_numel: int | None = None,
    ) -> None:
        self.model, self.tokenizer, self.args = model, tokenizer, args
        self.device, self.rank, self.world_size = torch.device(device), rank, world_size
        self.distributed = dist.is_available() and dist.is_initialized()
        if world_size > 1 and not self.distributed:
            raise RuntimeError('multiple ranks require an initialized process group')
        self.is_fsdp = isinstance(model, FSDP)
        if self.device.type != 'cpu' and not self.is_fsdp:
            raise ValueError('production CUDA writer must use FSDP')
        parameters = list(model.parameters())
        if not parameters or any(not p.requires_grad for p in parameters):
            raise ValueError('all Qwen parameters must be trainable')
        if any(p.is_floating_point() and p.dtype != torch.float32 for p in parameters):
            raise ValueError('persistent full-FT parameters must be FP32 for small updates')
        self.model_numel = model_numel if model_numel is not None else sum(p.numel() for p in parameters)
        self.trainable_numel = self.model_numel
        self.optimizer = torch.optim.AdamW(
            parameters, lr=_get(args, 'learning_rate', 2e-6),
            weight_decay=_get(args, 'weight_decay', 0.0),
            **({'fused': True} if self.device.type == 'cuda' else {}),
        )
        self.optimizer_steps = 0
        self.max_length = _get(args, 'max_length', 128)
        self.eval_batch_size = _get(args, 'eval_batch_size', 8)
        self.last_generation_tokens = 0

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def broadcast_object(self, value: Any, src: int = 0) -> Any:
        if not self.distributed:
            return value
        payload = [value if self.rank == src else None]
        dist.broadcast_object_list(payload, src=src)
        return payload[0]

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()

    def _reduce(self, value: float | int, op: Any = dist.ReduceOp.SUM) -> float:
        tensor = torch.tensor(float(value), dtype=torch.float64, device=self.device)
        if self.distributed:
            dist.all_reduce(tensor, op=op)
        return float(tensor.item())

    def _gather(self, value: list[Any]) -> list[Any]:
        if not self.distributed:
            return value
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        return [item for rank_items in gathered for item in rank_items]

    def _primary_action(self, callback: Any) -> Any:
        """Broadcast rank-0 I/O errors before peers enter another collective."""
        result, error = None, None
        if self.is_primary:
            try:
                result = callback()
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
        result, error = self.broadcast_object((result, error))
        if error:
            raise RuntimeError(error)
        return result

    def _dataset(self, rows: Sequence[dict[str, Any]]) -> AnswerOnlyDataset:
        dataset = AnswerOnlyDataset(rows, self.tokenizer, self.max_length)
        if len(dataset) != len(rows):
            raise ValueError('a writer example has no answer tokens after truncation')
        if any(int((item['labels'][1:] != -100).sum()) == 0 for item in dataset):
            raise ValueError('a writer example has no shifted answer targets')
        return dataset

    def _batch(self, examples: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in make_collator(self.tokenizer)(examples).items()}

    def _rank_indices(self, count: int, batch_size: int):
        for start in range(0, count, batch_size * self.world_size):
            local_start = start + self.rank * batch_size
            yield list(range(local_start, min(local_start + batch_size, count)))

    def _parameter_samples(self) -> list[tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor]]:
        """Bounded update evidence; the reported maximum is over these samples."""
        samples = []
        for parameter in self.model.parameters():
            if parameter.numel():
                # FP32 rounds large shard endpoints up to numel, producing an
                # out-of-bounds index. FP64 exactly represents these indices.
                positions = torch.linspace(0, parameter.numel() - 1, min(64, parameter.numel()),
                                           device=parameter.device, dtype=torch.float64).long()
                samples.append((parameter, positions, parameter.detach().view(-1)[positions].clone()))
        return samples

    def train(self, rows: Sequence[dict[str, Any]], seed: int, steps: int | None = None) -> FullFTTrainStats:
        steps = _get(self.args, 'update_steps', 1) if steps is None else steps
        batch_size = _get(self.args, 'per_device_batch_size', 1)
        if steps <= 0 or batch_size <= 0:
            raise ValueError('update steps and per-device batch size must be positive')
        dataset = self._dataset(rows)
        examples = list(dataset)
        global_answers = sum(int((item['labels'][1:] != -100).sum()) for item in examples)
        self.model.train()
        config = getattr(self.model, 'config', None)
        if config is not None:
            config.use_cache = False
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        samples = self._parameter_samples()
        loss_sum = sequence_tokens = answer_tokens = effective_sequence_tokens = 0
        effective_answer_tokens = padding_examples = 0
        last_grad_norm = 0.0
        rng = random.Random(seed)
        for _ in range(steps):
            indices = list(range(len(examples)))
            rng.shuffle(indices)
            self.optimizer.zero_grad(set_to_none=True)
            chunks = list(self._rank_indices(len(indices), batch_size))
            for chunk_index, positions in enumerate(chunks):
                selected = [examples[indices[position]] for position in positions]
                real = bool(selected)
                batch = self._batch(selected if real else [examples[indices[0]]])
                actual_answers = int((batch['labels'][:, 1:] != -100).sum())
                actual_sequence = int(batch['attention_mask'].sum())
                boundary = chunk_index + 1 == len(chunks)
                context = self.model.no_sync() if self.is_fsdp and not boundary else nullcontext()
                with context:
                    output = self.model(**batch, use_cache=False)
                    loss = output.loss
                    if not torch.isfinite(loss.detach()).all():
                        raise ValueError('nonfinite full-parameter training loss')
                    # FSDP averages across ranks. Counteract this averaging and
                    # normalize the entire logical update by its target tokens.
                    weight = actual_answers * self.world_size / global_answers if real else 0.0
                    (loss * weight).backward()
                sequence_tokens += actual_sequence
                answer_tokens += actual_answers
                if real:
                    loss_sum += float(loss.detach()) * actual_answers
                    effective_sequence_tokens += actual_sequence
                    effective_answer_tokens += actual_answers
                else:
                    padding_examples += 1
            if self.is_fsdp:
                grad_norm = clip_full_shard_grad_norm_(self.model, _get(self.args, 'max_grad_norm', 1.0))
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), _get(self.args, 'max_grad_norm', 1.0))
            last_grad_norm = float(grad_norm)
            if not math.isfinite(last_grad_norm):
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError('nonfinite global gradient norm; optimizer step was not executed')
            self.optimizer.step()
            self.optimizer_steps += 1
            self.optimizer.zero_grad(set_to_none=True)
        delta = max((float((parameter.detach().view(-1)[positions] - before).abs().max())
                     for parameter, positions, before in samples), default=0.0)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(self.device) / 1024 ** 3 if self.device.type == 'cuda' else 0.0
        total_loss = self._reduce(loss_sum)
        real_answer_count = int(self._reduce(effective_answer_tokens))
        return FullFTTrainStats(
            loss=total_loss / max(real_answer_count, 1), optimizer_steps=steps,
            sequence_tokens=int(self._reduce(sequence_tokens)), answer_tokens=int(self._reduce(answer_tokens)),
            wall_seconds=self._reduce(elapsed, dist.ReduceOp.MAX), peak_vram_gb=self._reduce(peak, dist.ReduceOp.MAX),
            effective_sequence_tokens=int(self._reduce(effective_sequence_tokens)), effective_answer_tokens=real_answer_count,
            padding_examples=int(self._reduce(padding_examples)), parameter_update_max_abs=self._reduce(delta, dist.ReduceOp.MAX),
            gradient_norm_before_clip=last_grad_norm,
        )

    @torch.no_grad()
    def answer_nll(self, rows: Sequence[dict[str, Any]]) -> tuple[list[float], int]:
        if not rows:
            return [], 0
        dataset = self._dataset(rows)
        previous_training = self.model.training
        self.model.eval()
        scored, tokens = [], 0
        try:
            for indices in self._rank_indices(len(dataset), self.eval_batch_size):
                batch = self._batch([dataset[index] for index in indices] if indices else [dataset[0]])
                logits = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False).logits
                labels = batch['labels'][:, 1:]
                losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), labels, reduction='none', ignore_index=-100)
                nll = losses.sum(1) / (labels != -100).sum(1).clamp_min(1)
                values = nll.cpu().tolist()
                if not all(math.isfinite(value) for value in values):
                    raise ValueError('nonfinite reward loss')
                scored.extend(zip(indices, values))
                tokens += int(batch['attention_mask'].sum())
        finally:
            self.model.train(previous_training)
        results = sorted(self._gather(scored))
        if [index for index, _ in results] != list(range(len(rows))):
            raise RuntimeError('distributed probe result coverage mismatch')
        return [value for _, value in results], int(self._reduce(tokens))

    @torch.no_grad()
    def generate_answers(
        self, questions: Sequence[str], *, batch_size: int | None = None,
        max_length: int | None = None, max_new_tokens: int | None = None,
    ) -> list[str]:
        """Greedy closed-book decoding with every token entering root FSDP.

        The no-cache path is deliberately simple and avoids calling a wrapped
        module's HF generate method, which bypasses root FSDP hooks.
        """
        if not questions:
            self.last_generation_tokens = 0
            return []
        batch_size = self.eval_batch_size if batch_size is None else batch_size
        max_length = self.max_length if max_length is None else max_length
        max_new_tokens = _get(self.args, 'max_new_tokens', 16) if max_new_tokens is None else max_new_tokens
        if min(batch_size, max_length, max_new_tokens) <= 0:
            raise ValueError('generation sizes must be positive')
        previous_training, old_padding = self.model.training, self.tokenizer.padding_side
        self.model.eval()
        self.tokenizer.padding_side = 'left'
        predictions, tokens = [], 0
        eos = self.tokenizer.eos_token_id
        eos_ids = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = eos_ids[0] if eos_ids else 0
        try:
            for indices in self._rank_indices(len(questions), batch_size):
                prompts = [qa_prompt(questions[index]) for index in indices] if indices else [qa_prompt(questions[0])]
                encoded = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True,
                                         max_length=max_length, add_special_tokens=True)
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=self.device)
                generated = []
                for _ in range(max_new_tokens):
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 0)
                    output = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                        position_ids=position_ids, use_cache=False)
                    next_ids = output.logits[:, -1].argmax(-1)
                    next_ids = torch.where(done, torch.full_like(next_ids, pad_id), next_ids)
                    generated.append(next_ids)
                    tokens += int(attention_mask.sum())
                    if eos_ids:
                        for eos_id in eos_ids:
                            done |= next_ids == eos_id
                    input_ids = torch.cat((input_ids, next_ids[:, None]), dim=1)
                    # Keep finished rows harmless while other ranks decode.
                    attention_mask = torch.cat((attention_mask, (~done).long()[:, None]), dim=1)
                    if bool(self._reduce(int(done.all()), dist.ReduceOp.MIN)):
                        break
                decoded = self.tokenizer.batch_decode(torch.stack(generated, dim=1).cpu(), skip_special_tokens=True)
                predictions.extend(zip(indices, decoded))
        finally:
            self.tokenizer.padding_side = old_padding
            self.model.train(previous_training)
        self.last_generation_tokens = int(self._reduce(tokens))
        ordered = sorted(self._gather(predictions))
        if [index for index, _ in ordered] != list(range(len(questions))):
            raise RuntimeError('distributed generation result coverage mismatch')
        return [prediction for _, prediction in ordered]

    def evaluate_checkpoint(
        self, rows: Sequence[dict[str, Any]], locality: Sequence[dict[str, Any]],
        multihop: Sequence[dict[str, Any]], checkpoint: str, current_stage: int | None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        metrics: dict[str, Any] = {'checkpoint': checkpoint, 'num_facts': len(rows), 'current_stage': current_stage}
        records = []
        for axis in AXES:
            question_key, answer_key = axis_fields(axis)
            axis_rows = locality if axis == 'locality' else multihop if axis == 'multihop' else rows
            valid = [row for row in axis_rows if isinstance(row.get(question_key), str) and row[question_key].strip()
                     and isinstance(row.get(answer_key), str) and row[answer_key].strip()]
            answers = self.generate_answers([row[question_key] for row in valid])
            axis_records = []
            for row, prediction in zip(valid, answers):
                axis_records.append({'checkpoint': checkpoint, 'sample_id': row['sample_id'], 'fact_stage': int(row['stage']),
                                     'axis': axis, 'question': row[question_key], 'gold': row[answer_key], 'prediction': prediction,
                                     **answer_scores(prediction, row[answer_key])})
            count = len(axis_records)
            exact_count, match_count = sum(int(r['exact']) for r in axis_records), sum(int(r['match']) for r in axis_records)
            metrics[axis] = {'count': count, 'exact_count': exact_count, 'match_count': match_count,
                             'exact_accuracy': exact_count / count if count else None,
                             'match_accuracy': match_count / count if count else None}
            records.extend(axis_records)
        metrics.update(summarize_checkpoint_records(records, current_stage))
        return metrics, records

    def save_checkpoint(self, path: Path, state: Any, manifest_sha256: str) -> None:
        path = Path(path)
        token = self.broadcast_object(str(uuid.uuid4()) if self.is_primary else None)
        temporary = path.with_name(path.name + f'.tmp-{token}')

        def prepare():
            if path.exists():
                raise FileExistsError(f'checkpoint already exists: {path}')
            temporary.mkdir(parents=True, exist_ok=False)

        self._primary_action(prepare)
        rng = _rng_state()
        if self.is_fsdp:
            with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT,
                                      ShardedStateDictConfig(offload_to_cpu=True),
                                      ShardedOptimStateDictConfig(offload_to_cpu=True)):
                payload = {'model': self.model.state_dict(), 'optimizer': FSDP.optim_state_dict(self.model, self.optimizer)}
                save_state_dict(payload, storage_writer=FileSystemWriter(str(temporary / 'shards')),
                                process_group=dist.group.WORLD)
        else:
            self._primary_action(lambda: torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()},
                                                    temporary / 'model_optimizer.pt'))
        torch.save(rng, temporary / f'rng-rank{self.rank}.pt')
        self.barrier()

        def commit():
            torch.save(state, temporary / 'writer_state.pt')
            atomic_json(temporary / 'checkpoint_meta.json', {
                'schema_version': CHECKPOINT_SCHEMA, 'model_dir': str(MODEL_DIR.resolve()),
                'manifest_sha256': manifest_sha256, 'world_size': self.world_size,
                'format': 'torch.distributed.checkpoint' if self.is_fsdp else 'cpu-test',
                'has_optimizer': True, 'optimizer_steps': self.optimizer_steps,
                'model_numel': self.model_numel, 'trainable_numel': self.trainable_numel,
                'parameter_dtype': 'float32', 'compute_dtype': 'bfloat16' if self.is_fsdp else 'float32',
                'created_unix_time': time.time(),
            })
            (temporary / 'COMMITTED').write_text('complete\n', encoding='utf-8')
            os.replace(temporary, path)

        self._primary_action(commit)
        self.barrier()

    def load_checkpoint(self, path: Path, manifest_sha256: str) -> Any:
        path = Path(path)

        def read_metadata():
            if not (path / 'COMMITTED').is_file():
                raise ValueError(f'checkpoint is not committed: {path}')
            return json.loads((path / 'checkpoint_meta.json').read_text(encoding='utf-8'))

        metadata = self._primary_action(read_metadata)
        if metadata.get('schema_version') != CHECKPOINT_SCHEMA:
            raise ValueError('unsupported full-FT writer checkpoint')
        if metadata.get('manifest_sha256') != manifest_sha256:
            raise ValueError('checkpoint was created from a different manifest')
        if metadata.get('model_dir') != str(MODEL_DIR.resolve()):
            raise ValueError('checkpoint model directory differs from required Qwen3-8B-Base')
        if metadata.get('world_size') != self.world_size:
            raise ValueError('exact RNG resume requires the original world size')
        if metadata.get('model_numel') != self.model_numel or not metadata.get('has_optimizer'):
            raise ValueError('checkpoint model/optimizer metadata mismatch')
        if (metadata.get('format') == 'torch.distributed.checkpoint') != self.is_fsdp:
            raise ValueError('checkpoint backend format mismatch')
        if self.is_fsdp:
            reader = FileSystemReader(str(path / 'shards'))
            with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT,
                                      ShardedStateDictConfig(offload_to_cpu=True),
                                      ShardedOptimStateDictConfig(offload_to_cpu=True)):
                payload = {'model': self.model.state_dict()}
                load_state_dict(payload, storage_reader=reader, process_group=dist.group.WORLD)
                self.model.load_state_dict(payload['model'], strict=True)
                optimizer_state = load_sharded_optimizer_state_dict(model_state_dict=payload['model'], optimizer_key='optimizer',
                                                                    storage_reader=reader)['optimizer']
                self.optimizer.load_state_dict(FSDP.optim_state_dict_to_load(self.model, self.optimizer, optimizer_state))
        else:
            payload = _torch_load(path / 'model_optimizer.pt')
            self.model.load_state_dict(payload['model'], strict=True)
            self.optimizer.load_state_dict(payload['optimizer'])
        self.optimizer_steps = int(metadata['optimizer_steps'])
        writer_state = self._primary_action(lambda: _torch_load(path / 'writer_state.pt'))
        rng = _torch_load(path / f'rng-rank{self.rank}.pt')
        self.barrier()
        _restore_rng(rng)
        return writer_state


def create_backend(args: Any) -> FullFTBackend:
    args.model_dir = validate_model_dir(args.model_dir)
    rank, world_size, _local_rank, device = setup_distributed(args)
    seed = _get(args, 'seed', 42) + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tokenizer = load_tokenizer(args.model_dir)
    base = load_model(args.model_dir, device, use_flash_attention=_get(args, 'flash_attention', False))
    # FP32 master shards ensure 1e-6-scale AdamW updates survive rounding.
    base.float()
    base.requires_grad_(True)
    base.config.use_cache = False
    if _get(args, 'activation_checkpointing', True):
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    numel = sum(parameter.numel() for parameter in base.parameters())
    model = FSDP(base, auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen3DecoderLayer}),
                 sharding_strategy=ShardingStrategy.FULL_SHARD,
                 mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16,
                                                 buffer_dtype=torch.bfloat16, cast_forward_inputs=True),
                 backward_prefetch=BackwardPrefetch.BACKWARD_PRE, device_id=device,
                 sync_module_states=False, limit_all_gathers=True, use_orig_params=False)
    return FullFTBackend(model, tokenizer, args, device, rank=rank, world_size=world_size, model_numel=numel)
