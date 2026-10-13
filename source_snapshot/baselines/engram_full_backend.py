"""Full-parameter or LoRA Qwen3 backend for an online external-memory writer.

Every public operation is collective: all ranks must call it with the same
rows/questions, in the same order. Tiny update batches are replicated on every
rank. FSDP averages their identical objective gradients; replication
does not increase a sample's optimization weight. Logical nonpadding tokens
count the batch once; physical tokens count all replicas. Padded-token counts
are also recorded and are a closer (though incomplete) compute proxy.

The original Qwen weights and Adam states are FP32 masters. FSDP forward and
gradient communication use BF16. This preserves small updates such as 2e-6,
which would often round away if Adam wrote directly into BF16 weights.
Full mode updates every original parameter; LoRA mode freezes original weights.
Neither mode uses retrieval inputs at inference.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
import uuid
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import __version__ as transformers_version
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
    AnswerOnlyDataset, atomic_json, load_model, load_tokenizer, make_collator,
    qa_prompt, sha256_file,
)
from baselines.engram_grad_norm import clip_full_shard_grad_norm_
from baselines.engram_training_views import (
    TRAINING_VIEW_VERSION, TRAINING_VIEW_POLICIES, REFORMULATION_PREFIX,
    expand_training_views, load_training_view_cache,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXED_MODEL_DIR = Path(os.environ.get('CAMR_MODEL_DIR', REPOSITORY_ROOT / 'models/Qwen3-8B-Base'))
CHECKPOINT_SCHEMA = 'engram-full-writer-backend/v1'
LORA_TARGET_MODULES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
HISTORY_ONLY_LOSS_POLICIES = ('follow_training_loss', 'legacy_fact_mean')


def _runtime(args):
    """Initialize only the torchrun process group owned by this backend."""
    if not torch.cuda.is_available():
        raise RuntimeError('The full Qwen backend requires CUDA and torchrun.')
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if not dist.is_initialized():
        if not all(key in os.environ for key in ('RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT')):
            raise RuntimeError('Launch the full Qwen backend with torchrun.')
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(minutes=int(getattr(args, 'distributed_timeout_minutes', 30))),
        )
    return dist.get_rank(), dist.get_world_size(), local_rank, device


def _configure_parameter_mode(base, args):
    """Configure a fresh base before sharding, with identical LoRA init on ranks."""
    if any('lora_' in name.lower() or 'engram' in name.lower()
           for name, _ in base.named_modules()):
        raise ValueError('The backend requires a fresh model without adapter or Engram modules.')
    mode = getattr(args, 'parameter_mode', 'full')
    original_count = sum(parameter.numel() for parameter in base.parameters())
    base.float()
    if mode == 'full':
        for parameter in base.parameters():
            parameter.requires_grad_(True)
    elif mode == 'lora':
        from peft import LoraConfig, get_peft_model
        device = next(base.parameters()).device
        devices = [device.index] if device.type == 'cuda' else []
        # The controller seeds rank-local RNG differently. LoRA is newly
        # initialized, so it must use an isolated common seed before sharding.
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(int(getattr(args, 'seed', 42)))
            if device.type == 'cuda':
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(int(getattr(args, 'seed', 42)))
            base = get_peft_model(base, LoraConfig(
                task_type='CAUSAL_LM', inference_mode=False,
                r=int(getattr(args, 'lora_rank', 16)),
                lora_alpha=int(getattr(args, 'lora_alpha', 32)),
                lora_dropout=float(getattr(args, 'lora_dropout', 0.0)),
                target_modules=list(LORA_TARGET_MODULES), bias='none',
            ))
        base.float()
        if any(parameter.requires_grad and 'lora_' not in name
               for name, parameter in base.named_parameters()):
            raise ValueError('LoRA mode must freeze every original model parameter.')
    else:
        raise ValueError('parameter_mode must be full or lora.')
    parameter_count = sum(parameter.numel() for parameter in base.parameters())
    trainable_count = sum(parameter.numel() for parameter in base.parameters() if parameter.requires_grad)
    if not trainable_count:
        raise ValueError('No trainable parameters.')
    if not all(parameter.dtype == torch.float32 for parameter in base.parameters()):
        raise ValueError('Every stored model parameter must be FP32 before FSDP wrapping.')
    return base, {'parameter_count': parameter_count, 'original_parameter_count': original_count,
                  'trainable_parameters': trainable_count}


def _build_model(args, device):
    tokenizer = load_tokenizer(FIXED_MODEL_DIR)
    base = load_model(FIXED_MODEL_DIR, device,
                      use_flash_attention=bool(getattr(args, 'flash_attention', True)))
    if base.config.model_type != 'qwen3':
        raise ValueError('The fixed model directory must contain a Qwen3 model.')
    base, parameter_info = _configure_parameter_mode(base, args)
    base.config.use_cache = False
    if bool(getattr(args, 'activation_checkpointing', True)):
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model = FSDP(
        base,
        auto_wrap_policy=partial(transformer_auto_wrap_policy,
                                 transformer_layer_cls={Qwen3DecoderLayer}),
        sharding_strategy=(ShardingStrategy.SHARD_GRAD_OP
                           if getattr(args, 'fsdp_sharding_strategy', 'full_shard') == 'shard_grad_op'
                           else ShardingStrategy.FULL_SHARD),
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                       reduce_dtype=torch.bfloat16,
                                       buffer_dtype=torch.bfloat16,
                                       cast_forward_inputs=True),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=device, sync_module_states=False, limit_all_gathers=True,
        use_orig_params=getattr(args, 'parameter_mode', 'full') == 'lora',
    )
    model._engram_parameter_info = parameter_info
    torch.cuda.empty_cache()
    return model, tokenizer, parameter_info['parameter_count']


def _answer_losses(logits, labels):
    """Per-example causal answer NLL sums and valid shifted-token counts."""
    targets = labels[:, 1:]
    counts = (targets != -100).sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError('An example has no answer tokens after causal shifting.')
    losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), targets,
                             ignore_index=-100, reduction='none')
    return losses.sum(dim=1), counts


def _training_objective(loss_sums, counts, training_loss='token_mean', current_loss_weight=0.5,
                        fact_indices=None, has_current=True,
                        history_only_loss_policy='follow_training_loss'):
    """Reduce answer losses with explicit current/history roles.

    Role averaging gives a fixed mass to the current fact and to historical
    replay collectively, independent of replay count and answer lengths. Each
    history row has equal mass within its role. A current-only batch receives
    full weight rather than being scaled by current_loss_weight. Without a
    current fact, role_mean gives every historical fact equal weight. Views
    are averaged within each fact first. token_mean always averages all valid
    answer tokens; legacy_fact_mean explicitly reproduces the old exception
    that made history-only token_mean batches use equal fact weights.
    """
    if type(has_current) is not bool:
        raise ValueError('has_current must be a boolean.')
    if training_loss not in ('token_mean', 'role_mean'):
        raise ValueError('training_loss must be token_mean or role_mean.')
    if history_only_loss_policy not in HISTORY_ONLY_LOSS_POLICIES:
        raise ValueError('history_only_loss_policy must be follow_training_loss or legacy_fact_mean.')
    if training_loss == 'token_mean' and (has_current or history_only_loss_policy == 'follow_training_loss'):
        return loss_sums.sum() / counts.sum()
    row_means = loss_sums / counts
    if fact_indices is not None:
        if len(fact_indices) != row_means.numel() or not fact_indices:
            raise ValueError('Each training view must map to one fact.')
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in fact_indices):
            raise ValueError('Fact indices must be nonnegative integers.')
        fact_count = max(fact_indices) + 1
        if set(fact_indices) != set(range(fact_count)):
            raise ValueError('Fact indices must cover a contiguous sequence.')
        index_tensor = torch.tensor(fact_indices, dtype=torch.long, device=row_means.device)
        row_means = torch.stack([row_means[index_tensor == index].mean() for index in range(fact_count)])
    if not has_current:
        return row_means.mean()
    if row_means.numel() == 1:
        return row_means[0]
    return (current_loss_weight * row_means[0]
            + (1.0 - current_loss_weight) * row_means[1:].mean())


class FullParameterBackend:
    """Persistent full-model optimizer, collective evaluation and checkpointing.

    Args are an argparse namespace; supported optional attributes are seed,
    learning_rate, weight_decay, max_grad_norm, max_length, max_new_tokens,
    eval_batch_size, activation_checkpointing, flash_attention, training_loss,
    current_loss_weight, training_views, training_views_file, training_view_policy,
    history_only_loss_policy, parameter_mode,
    lora_rank, lora_alpha, lora_dropout, and
    distributed_timeout_minutes. model_dir must resolve to FIXED_MODEL_DIR.
    save_checkpoint accepts an arbitrary trusted Python extra_state on rank 0;
    other ranks may pass None. Exact resume requires the same world size.
    """

    def __init__(self, args):
        requested = Path(getattr(args, 'model_dir', FIXED_MODEL_DIR)).resolve()
        if requested != FIXED_MODEL_DIR.resolve():
            raise ValueError(f'This experiment fixes the base model to {FIXED_MODEL_DIR}.')
        self.args = args
        self.max_length = int(getattr(args, 'max_length', 128))
        self.max_new_tokens = int(getattr(args, 'max_new_tokens', 16))
        self.eval_batch_size = int(getattr(args, 'eval_batch_size', 16))
        self.max_grad_norm = float(getattr(args, 'max_grad_norm', 1.0))
        self.learning_rate = float(getattr(args, 'learning_rate', 2e-6))
        self.weight_decay = float(getattr(args, 'weight_decay', 0.0))
        self.training_loss = getattr(args, 'training_loss', 'token_mean')
        self.current_loss_weight = float(getattr(args, 'current_loss_weight', 0.5))
        self.training_views = getattr(args, 'training_views', 1)
        self.training_view_policy = getattr(args, 'training_view_policy', 'accepted_only')
        self.history_only_loss_policy = getattr(args, 'history_only_loss_policy', 'follow_training_loss')
        self.parameter_mode = getattr(args, 'parameter_mode', 'full')
        self.lora_rank = getattr(args, 'lora_rank', 16)
        self.lora_alpha = getattr(args, 'lora_alpha', 32)
        self.lora_dropout = float(getattr(args, 'lora_dropout', 0.0))
        if isinstance(self.training_views, bool) or self.training_views not in (1, 2):
            raise ValueError('training_views must be 1 or 2.')
        if self.training_view_policy not in TRAINING_VIEW_POLICIES:
            raise ValueError('training_view_policy must be accepted_only or legacy_wrapper.')
        if self.history_only_loss_policy not in HISTORY_ONLY_LOSS_POLICIES:
            raise ValueError('history_only_loss_policy must be follow_training_loss or legacy_fact_mean.')
        if self.parameter_mode not in ('full', 'lora'):
            raise ValueError('parameter_mode must be full or lora.')
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in (self.lora_rank, self.lora_alpha)):
            raise ValueError('lora_rank and lora_alpha must be positive integers.')
        if not math.isfinite(self.lora_dropout) or not 0 <= self.lora_dropout < 1:
            raise ValueError('lora_dropout must be finite and in [0, 1).')
        if self.training_loss not in ('token_mean', 'role_mean'):
            raise ValueError('training_loss must be token_mean or role_mean.')
        if self.training_views > 1 and self.training_loss != 'role_mean':
            raise ValueError('Multiple training views require role_mean to preserve per-fact weights.')
        if not math.isfinite(self.current_loss_weight) or not 0 < self.current_loss_weight < 1:
            raise ValueError('current_loss_weight must be finite and strictly between 0 and 1.')
        if (self.max_length < 2 or self.max_new_tokens < 1 or self.eval_batch_size < 1
                or not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0
                or not math.isfinite(self.weight_decay) or self.weight_decay < 0):
            raise ValueError('Invalid full-parameter training or tokenization configuration.')
        self.training_view_cache, self.training_views_file_sha256 = load_training_view_cache(
            getattr(args, 'training_views_file', None))
        self.rank, self.world_size, self.local_rank, self.device = _runtime(args)
        self.numerical_runtime = {
            'torch_version': str(torch.__version__), 'transformers_version': transformers_version,
            'flash_attention_deterministic': os.environ.get('FLASH_ATTENTION_DETERMINISTIC', '0') == '1',
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'nccl_algo': os.environ.get('NCCL_ALGO'), 'nccl_proto': os.environ.get('NCCL_PROTO'),
            'torch_deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        }
        if self.parameter_mode == 'lora':
            from peft import __version__ as peft_version
            self.numerical_runtime['peft_version'] = peft_version
        seed = int(getattr(args, 'seed', 42)) + self.rank
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(seed)
        self.model, self.tokenizer, self.parameter_count = _build_model(args, self.device)
        self._is_fsdp = isinstance(self.model, FSDP)
        # Distributed checkpoint plans are Python objects.  Gathering a large
        # FSDP plan through the NCCL default group stages it on CUDA and can
        # fail even when model training has ample device memory.  The sharded
        # checkpoint tensors are already offloaded to CPU, so use Gloo for
        # checkpoint coordination and I/O collectives.
        self._checkpoint_pg = (dist.new_group(backend='gloo', timeout=timedelta(
            minutes=int(getattr(args, 'distributed_timeout_minutes', 30))))
            if self._is_fsdp and self.world_size > 1 else
            dist.group.WORLD if self._is_fsdp else None)
        parameter_info = getattr(self.model, '_engram_parameter_info', None)
        if parameter_info is None:
            if self._is_fsdp:
                raise ValueError('Production builder must report parameter counts before sharding.')
            parameter_info = {
                'original_parameter_count': self.parameter_count,
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            }
        self.trainable_parameter_count = parameter_info['trainable_parameters']
        self.optimizer = torch.optim.AdamW(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            lr=self.learning_rate, weight_decay=self.weight_decay,
            **({'fused': True} if self.device.type == 'cuda' else {}),
        )
        self.total_optimizer_steps = 0
        self.forward_accounting = {
            'nll_calls': 0, 'nll_forward_calls': 0,
            'nll_logical_sequence_tokens': 0, 'nll_physical_sequence_tokens': 0,
            'nll_physical_padded_sequence_tokens': 0,
            'generation_calls': 0, 'generation_forward_calls': 0,
            'generation_logical_sequence_tokens': 0,
            'generation_physical_sequence_tokens': 0,
            'generation_physical_padded_sequence_tokens': 0,
            'generation_output_tokens': 0,
        }
        self.last_nll_stats = {}
        self.last_generation_stats = {}
        self.initial_parameter_fingerprint = self.parameter_fingerprint()
        self._source_identity = {
            name: sha256_file(FIXED_MODEL_DIR / name)
            for name in ('config.json', 'model.safetensors.index.json')
            if (FIXED_MODEL_DIR / name).is_file()
        }
        self.metadata = {
            'model_dir': str(FIXED_MODEL_DIR.resolve()),
            'backend': 'full_parameter_fsdp' if self.parameter_mode == 'full' else 'lora_fsdp',
            'parameter_mode': self.parameter_mode,
            'parameter_count': self.parameter_count, 'original_parameter_count': parameter_info['original_parameter_count'],
            'trainable_parameters': self.trainable_parameter_count,
            'trainable_fraction': self.trainable_parameter_count / self.parameter_count,
            'all_original_parameters_trainable': self.parameter_mode == 'full',
            'base_weights_frozen': self.parameter_mode == 'lora',
            'lora': self.parameter_mode == 'lora', 'internal_engram_hooks': False,
            'lora_config': ({'rank': self.lora_rank, 'alpha': self.lora_alpha,
                             'dropout': self.lora_dropout, 'target_modules': list(LORA_TARGET_MODULES)}
                            if self.parameter_mode == 'lora' else None),
            'fsdp_use_orig_params': self.parameter_mode == 'lora',
            'master_dtype': 'float32', 'compute_dtype': 'bfloat16', 'reduce_dtype': 'bfloat16',
            'optimizer': 'AdamW', 'optimizer_state_dtypes': {},
            'gradient_clipping': ('stable_name_order_full_shard_global_norm' if self.parameter_mode == 'full'
                                  else 'torch_fsdp_global_norm_use_orig_params'),
            'numerical_runtime': self.numerical_runtime,
            'world_size': self.world_size, 'batch_strategy': 'replicated_identical_global_batch',
            'fsdp_sharding_strategy': getattr(self.args, 'fsdp_sharding_strategy', 'full_shard'),
            'objective': ('mean over all shifted answer tokens in the logical batch'
                          if self.training_loss == 'token_mean' else
                          'answer-token mean per view, mean of views per fact; current_loss_weight * current fact + (1-current_loss_weight) * mean of replay facts; current-only batches have full weight'),
            'training_loss': self.training_loss,
            'current_loss_weight': self.current_loss_weight,
            'history_only_loss_policy': self.history_only_loss_policy,
            'history_only_objective': (
                'mean over all shifted answer tokens in the logical historical batch'
                if self.training_loss == 'token_mean' and self.history_only_loss_policy == 'follow_training_loss'
                else 'answer-token mean per view, mean of views per fact, equal mean of all historical facts'),
            'training_views': self.training_views,
            'training_view_policy': self.training_view_policy,
            'training_view_version': TRAINING_VIEW_VERSION,
            'training_views_file_sha256': self.training_views_file_sha256,
            'training_view_source': (
                'original question only' if self.training_views == 1 else
                'source-bound accepted training-only cache; rejected or absent views use original only'
                if self.training_view_policy == 'accepted_only' else
                'source-bound accepted training-only cache; legacy instruction reformulation fallback'),
            'training_view_limitations': (
                'acceptance does not guarantee semantic equivalence; oversized accepted views are omitted; probes and evaluation are never augmented'
                if self.training_view_policy == 'accepted_only' else
                'legacy fallback changes prompting but does not semantically rewrite the question; probes and evaluation are never augmented'),
            'cost_accounting': 'logical nonpadding tokens counted once; physical tokens include every rank replica; padded token counts reported separately',
            'cost_limitations': 'token counts do not include activation-checkpoint recomputation, backward operations, or optimizer FLOPs',
            'generation': 'closed-book common.qa_prompt; greedy; no KV cache; per-step rank-0 token broadcast',
            'initial_parameter_fingerprint': self.initial_parameter_fingerprint,
            'fingerprint_scope': 'SHA256 of 64 evenly spaced FP32 master values per local parameter shard on every rank; not a full-checkpoint hash',
            'source_config_sha256': self._source_identity,
        }

    def _reduce(self, value, op=dist.ReduceOp.SUM):
        tensor = torch.tensor(float(value), dtype=torch.float64, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(tensor, op=op)
        return float(tensor.item())

    def _barrier(self):
        if dist.is_initialized():
            dist.barrier()

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _sample_parameters(self):
        samples = []
        for parameter in self.model.parameters():
            flat = parameter.detach().reshape(-1)
            if flat.numel():
                # FP32 cannot represent every large shard index; rounding its
                # final endpoint upward could index one past the shard.
                indices = torch.linspace(0, flat.numel() - 1, min(64, flat.numel()),
                                         device=flat.device, dtype=torch.float64).long()
                samples.append(flat[indices].float().clone())
        return torch.cat(samples) if samples else torch.empty(0, device=self.device)

    def _fingerprint_sample(self, sample):
        digest = hashlib.sha256(sample.detach().float().cpu().numpy().tobytes()).hexdigest()
        digests = [None] * self.world_size
        if self.world_size > 1:
            dist.all_gather_object(digests, digest)
        else:
            digests[0] = digest
        return hashlib.sha256('|'.join(digests).encode('ascii')).hexdigest()

    def parameter_fingerprint(self):
        """Collective fingerprint of deterministic samples of the FP32 shards."""
        return self._fingerprint_sample(self._sample_parameters())

    def _optimizer_dtypes(self):
        counts = {}
        for state in self.optimizer.state.values():
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    key = f'{name}:{str(value.dtype).removeprefix("torch.")}'
                    counts[key] = counts.get(key, 0) + value.numel()
        # These are per-rank counts: padding and shard sizes may differ.
        return counts

    def _batch(self, rows):
        # Pass only fields used to construct training QA, even when the caller
        # retained evaluation fields in its original dataset records.
        train_rows = [{'update': row['update'], 'ans': row['ans']} for row in rows]
        dataset = AnswerOnlyDataset(train_rows, self.tokenizer, self.max_length)
        if len(dataset) != len(rows):
            raise ValueError('A requested training/probe example was truncated out.')
        return {key: value.to(self.device) for key, value in
                make_collator(self.tokenizer)(list(dataset)).items()}

    def _training_batch(self, rows):
        expanded, fact_indices = expand_training_views(
            rows, self.training_views, self.training_view_cache, self.training_view_policy)
        fallbacks = 0
        if self.training_views > 1:
            # Fact IDs, not a fixed stride, identify original versus extra
            # views: accepted_only permits a different count for every fact.
            # Oversized accepted views are omitted, keeping the original's
            # full fact weight. Legacy mode retains its wrapper/original
            # replacement chain and records each length fallback.
            kept_rows, kept_indices, original_facts = [], [], set()
            for view, fact_index in zip(expanded, fact_indices):
                if fact_index not in original_facts:
                    original_facts.add(fact_index)
                    kept_rows.append(view)
                    kept_indices.append(fact_index)
                    continue
                answer = view['ans']
                answer_ids = self.tokenizer(' ' + answer.strip() + (self.tokenizer.eos_token or ''),
                                            add_special_tokens=False).input_ids
                def fits(question):
                    return len(self.tokenizer(qa_prompt(question), add_special_tokens=True).input_ids) + len(answer_ids) <= self.max_length
                if not fits(view['update']):
                    if self.training_view_policy == 'accepted_only':
                        if not fits(rows[fact_index]['update']):
                            raise ValueError('Original training QA exceeds max_length; full augmented answers cannot be preserved.')
                        fallbacks += 1
                        continue
                    candidates = [REFORMULATION_PREFIX + rows[fact_index]['update'], rows[fact_index]['update']]
                    replacement = next((question for question in candidates if fits(question)), None)
                    if replacement is None:
                        raise ValueError('Original training QA exceeds max_length; full augmented answers cannot be preserved.')
                    view['update'] = replacement
                    fallbacks += 1
                kept_rows.append(view)
                kept_indices.append(fact_index)
            expanded, fact_indices = kept_rows, kept_indices
        return self._batch(expanded), fact_indices, fallbacks

    def train(self, rows: Sequence[dict[str, Any]], steps: int, *, has_current: bool = True):
        if type(has_current) is not bool:
            raise ValueError('has_current must be a boolean.')
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError('steps must be a nonnegative integer.')
        if not rows and steps:
            raise ValueError('Cannot update the model with an empty batch.')
        if steps == 0:
            return {'loss': 0.0, 'optimizer_steps': 0, 'sequence_tokens': 0,
                    'has_current': has_current,
                    'training_role': 'current_and_history' if has_current else 'history_only',
                    'answer_tokens': 0, 'physical_sequence_tokens': 0,
                    'physical_answer_tokens': 0, 'physical_padded_sequence_tokens': 0,
                    'wall_seconds': 0.0, 'peak_vram_gb': 0.0,
                    'sampled_parameter_delta_l2': 0.0, 'parameter_changed': False}
        batch, fact_indices, view_length_fallbacks = self._training_batch(rows)
        sequence_count = int(batch['attention_mask'].sum().item())
        answer_count = int((batch['labels'][:, 1:] != -100).sum().item())
        # Count roles before any rank replication. Block baselines explicitly
        # tag every row; CAMR uses the first fact as current by default.
        current_facts = {i for i, row in enumerate(rows)
                         if row.get('_stream_role', 'current' if has_current and i == 0 else 'replay') == 'current'}
        current_views = [i for i, fact in enumerate(fact_indices) if fact in current_facts]
        current_sequence = int(batch['attention_mask'][current_views].sum().item()) if current_views else 0
        current_answers = int((batch['labels'][current_views, 1:] != -100).sum().item()) if current_views else 0
        before_sample = self._sample_parameters()
        before_fingerprint = self._fingerprint_sample(before_sample)
        self.model.train()
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        self._sync()
        started = time.perf_counter()
        loss_sum = 0.0
        last_grad_norm = 0.0
        for _ in range(steps):
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'], use_cache=False).logits
            loss_sums, counts = _answer_losses(logits, batch['labels'])
            loss = _training_objective(loss_sums, counts, self.training_loss, self.current_loss_weight,
                                       fact_indices, has_current=has_current,
                                       history_only_loss_policy=self.history_only_loss_policy)
            if self._reduce(int(bool(torch.isfinite(loss))), dist.ReduceOp.MIN) != 1:
                raise FloatingPointError('Nonfinite full-parameter training loss.')
            loss.backward()
            if self._is_fsdp:
                grad_norm = (clip_full_shard_grad_norm_(self.model, self.max_grad_norm)
                             if self.parameter_mode == 'full' else self.model.clip_grad_norm_(self.max_grad_norm))
            else:  # Reachable only when CPU tests replace the production builder.
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            last_grad_norm = float(grad_norm)
            if not math.isfinite(last_grad_norm):
                raise FloatingPointError('Nonfinite global gradient norm.')
            self.optimizer.step()
            self.total_optimizer_steps += 1
            loss_sum += float(loss.detach())
        self.optimizer.zero_grad(set_to_none=True)
        self._sync()
        wall_seconds = self._reduce(time.perf_counter() - started, dist.ReduceOp.MAX)
        peak = torch.cuda.max_memory_allocated(self.device) / 2**30 if self.device.type == 'cuda' else 0.0
        after_sample = self._sample_parameters()
        delta_squared = self._reduce(float((after_sample - before_sample).double().square().sum()))
        after_fingerprint = self._fingerprint_sample(after_sample)
        self.metadata['optimizer_state_dtypes'] = self._optimizer_dtypes()
        return {
            'loss': self._reduce(loss_sum / steps) / self.world_size,
            'has_current': has_current,
            'training_role': 'current_and_history' if has_current else 'history_only',
            'training_facts': len(rows), 'training_view_rows': len(fact_indices),
            'training_view_length_fallbacks': view_length_fallbacks,
            'optimizer_steps': steps, 'total_optimizer_steps': self.total_optimizer_steps,
            'sequence_tokens': sequence_count * steps, 'answer_tokens': answer_count * steps,
            'current_sequence_tokens': current_sequence * steps,
            'replay_sequence_tokens': (sequence_count - current_sequence) * steps,
            'current_answer_tokens': current_answers * steps,
            'replay_answer_tokens': (answer_count - current_answers) * steps,
            'current_fact_presentations': len(current_facts) * steps,
            'replay_fact_presentations': (len(rows) - len(current_facts)) * steps,
            'current_view_presentations': len(current_views) * steps,
            'replay_view_presentations': (len(fact_indices) - len(current_views)) * steps,
            'physical_sequence_tokens': sequence_count * steps * self.world_size,
            'physical_answer_tokens': answer_count * steps * self.world_size,
            'physical_padded_sequence_tokens': batch['input_ids'].numel() * steps * self.world_size,
            'wall_seconds': wall_seconds, 'peak_vram_gb': self._reduce(peak, dist.ReduceOp.MAX),
            'last_global_grad_norm': last_grad_norm,
            'sampled_parameter_delta_l2': math.sqrt(delta_squared),
            'sampled_parameter_count': int(self._reduce(after_sample.numel())),
            'parameter_fingerprint_before': before_fingerprint,
            'parameter_fingerprint_after': after_fingerprint,
            'parameter_changed': before_fingerprint != after_fingerprint,
        }

    @torch.no_grad()
    def answer_nll(self, rows: Sequence[dict[str, Any]]):
        self.model.eval()
        started = time.perf_counter()
        values, tokens, padded, calls = [], 0, 0, 0
        for start in range(0, len(rows), self.eval_batch_size):
            batch = self._batch(rows[start:start + self.eval_batch_size])
            logits = self.model(input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'], use_cache=False).logits
            sums, counts = _answer_losses(logits, batch['labels'])
            values.extend((sums / counts).float().cpu().tolist())
            tokens += int(batch['attention_mask'].sum().item())
            padded += batch['input_ids'].numel()
            calls += 1
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError('Nonfinite answer NLL.')
        self._sync()
        self.last_nll_stats = {
            'forward_calls': calls, 'logical_sequence_tokens': tokens,
            'physical_sequence_tokens': tokens * self.world_size,
            'physical_padded_sequence_tokens': padded * self.world_size,
            'wall_seconds': self._reduce(time.perf_counter() - started, dist.ReduceOp.MAX),
        }
        self.forward_accounting['nll_calls'] += 1
        for key in ('forward_calls', 'logical_sequence_tokens', 'physical_sequence_tokens',
                    'physical_padded_sequence_tokens'):
            self.forward_accounting[f'nll_{key}'] += self.last_nll_stats[key]
        return values, tokens

    @torch.no_grad()
    def generate(self, questions: Sequence[str]):
        self.model.eval()
        started = time.perf_counter()
        answers, tokens, padded, forward_calls, output_tokens = [], 0, 0, 0, 0
        previous_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos
        if pad is None:
            raise ValueError('A pad or EOS token is required for synchronized generation.')
        try:
            for start in range(0, len(questions), self.eval_batch_size):
                prompts = [qa_prompt(str(question)) for question in questions[start:start + self.eval_batch_size]]
                encoded = self.tokenizer(prompts, return_tensors='pt', padding=True,
                                         truncation=True, max_length=self.max_length,
                                         add_special_tokens=True)
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=self.device)
                generated = []
                for _ in range(self.max_new_tokens):
                    positions = attention_mask.long().cumsum(dim=-1) - 1
                    positions.masked_fill_(attention_mask == 0, 0)
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                        position_ids=positions, use_cache=False).logits
                    next_tokens = logits[:, -1].argmax(dim=-1)
                    if self.world_size > 1:
                        dist.broadcast(next_tokens, src=0)
                    active = ~finished
                    next_tokens = torch.where(active, next_tokens, torch.full_like(next_tokens, pad))
                    generated.append(next_tokens)
                    tokens += int(attention_mask.sum().item())
                    padded += input_ids.numel()
                    forward_calls += 1
                    output_tokens += int(active.sum().item())
                    if eos is not None:
                        finished |= next_tokens == eos
                    if bool(finished.all()):
                        break
                    input_ids = torch.cat((input_ids, next_tokens[:, None]), dim=1)
                    attention_mask = torch.cat((attention_mask, active.long()[:, None]), dim=1)
                answers.extend(self.tokenizer.batch_decode(torch.stack(generated, dim=1),
                                                            skip_special_tokens=True))
        finally:
            self.tokenizer.padding_side = previous_padding
        self._sync()
        self.last_generation_stats = {
            'forward_calls': forward_calls, 'logical_sequence_tokens': tokens,
            'physical_sequence_tokens': tokens * self.world_size,
            'physical_padded_sequence_tokens': padded * self.world_size,
            'output_tokens': output_tokens,
            'wall_seconds': self._reduce(time.perf_counter() - started, dist.ReduceOp.MAX),
        }
        self.forward_accounting['generation_calls'] += 1
        for key in ('forward_calls', 'logical_sequence_tokens', 'physical_sequence_tokens',
                    'physical_padded_sequence_tokens', 'output_tokens'):
            self.forward_accounting[f'generation_{key}'] += self.last_generation_stats[key]
        return answers

    def _rng_state(self):
        return {
            'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda_local_device': torch.cuda.get_rng_state(self.device) if self.device.type == 'cuda' else None,
        }

    def _restore_rng(self, state):
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch_cpu'])
        if state['torch_cuda_local_device'] is not None:
            torch.cuda.set_rng_state(state['torch_cuda_local_device'], self.device)

    def _rank_zero_io(self, operation):
        """Broadcast rank-0 filesystem failures before any later collective."""
        result = [None, None]
        if self.rank == 0:
            try:
                result[1] = operation()
            except Exception as error:
                result[0] = f'{type(error).__name__}: {error}'
        if self.world_size > 1:
            dist.broadcast_object_list(result, src=0)
        if result[0] is not None:
            raise RuntimeError(result[0])
        return result[1]

    def save_checkpoint(self, path, extra_state=None):
        destination = Path(path).resolve()
        if destination == FIXED_MODEL_DIR.resolve() or FIXED_MODEL_DIR.resolve() in destination.parents:
            raise ValueError('Checkpoints must not modify the source model directory.')
        rng_state = self._rng_state()

        def prepare():
            if destination.exists():
                raise FileExistsError(f'Checkpoint already exists: {destination}')
            temporary = destination.with_name(destination.name + '.tmp-' + uuid.uuid4().hex)
            temporary.mkdir(parents=True, exist_ok=False)
            return str(temporary)

        temporary = Path(self._rank_zero_io(prepare))
        if self._is_fsdp:
            with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT,
                                     ShardedStateDictConfig(offload_to_cpu=True),
                                     ShardedOptimStateDictConfig(offload_to_cpu=True)):
                state = {'model': self.model.state_dict(),
                         'optimizer': FSDP.optim_state_dict(self.model, self.optimizer)}
                save_state_dict(state, storage_writer=FileSystemWriter(str(temporary / 'distributed')),
                                process_group=self._checkpoint_pg)
        else:  # Small CPU tests exercise the same complete-resume protocol.
            torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()},
                       temporary / f'cpu-state-rank-{self.rank}.pt')
        torch.save(rng_state, temporary / f'rng-rank-{self.rank}.pt')
        self._barrier()

        def commit():
            torch.save(extra_state, temporary / 'extra_state.pt')
            atomic_json(temporary / 'checkpoint_meta.json', {
                'schema_version': CHECKPOINT_SCHEMA, 'complete': True,
                'format': 'torch.distributed.checkpoint' if self._is_fsdp else 'cpu-test-state',
                'world_size': self.world_size, 'model_dir': str(FIXED_MODEL_DIR.resolve()),
                'source_identity': self._source_identity,
                'total_optimizer_steps': self.total_optimizer_steps,
                'forward_accounting': self.forward_accounting,
                'initial_parameter_fingerprint': self.initial_parameter_fingerprint,
                'master_dtype': 'float32', 'compute_dtype': 'bfloat16',
                'optimizer_state_dtypes': self._optimizer_dtypes(),
                'max_length': self.max_length, 'max_new_tokens': self.max_new_tokens,
                'max_grad_norm': self.max_grad_norm, 'eval_batch_size': self.eval_batch_size,
                'learning_rate': self.learning_rate, 'weight_decay': self.weight_decay,
                'training_loss': self.training_loss, 'current_loss_weight': self.current_loss_weight,
                'history_only_loss_policy': self.history_only_loss_policy,
                'training_views': self.training_views, 'training_view_version': TRAINING_VIEW_VERSION,
                'training_view_policy': self.training_view_policy,
                'training_views_file_sha256': self.training_views_file_sha256,
                'parameter_mode': self.parameter_mode, 'lora_rank': self.lora_rank,
                'lora_alpha': self.lora_alpha, 'lora_dropout': self.lora_dropout,
                'numerical_runtime': self.numerical_runtime,
                'created_unix_time': time.time(),
                'rng_scope': 'Python, NumPy, Torch CPU and the rank-local CUDA device',
                'backend_metadata': self.metadata,
            })
            # The preexisting-target check is repeated immediately before the
            # rename. A failed run leaves a named temporary directory to inspect.
            if destination.exists():
                raise FileExistsError(f'Checkpoint appeared during save: {destination}')
            os.rename(temporary, destination)
            return str(destination)

        saved = self._rank_zero_io(commit)
        self._barrier()
        self._restore_rng(rng_state)
        return saved

    def load_checkpoint(self, path):
        source = Path(path).resolve()

        def inspect():
            metadata = json.loads((source / 'checkpoint_meta.json').read_text(encoding='utf-8'))
            if metadata.get('schema_version') != CHECKPOINT_SCHEMA or not metadata.get('complete'):
                raise ValueError('Checkpoint is incomplete or has an unsupported schema.')
            if metadata['world_size'] != self.world_size:
                raise ValueError('Exact RNG/optimizer resume requires the same world size.')
            if metadata.get('model_dir') != str(FIXED_MODEL_DIR.resolve()) or metadata.get('source_identity') != self._source_identity:
                raise ValueError('Checkpoint belongs to a different source model.')
            if metadata.get('numerical_runtime') != self.numerical_runtime:
                raise ValueError('Checkpoint numerical runtime differs; exact resume requires matching settings and libraries.')
            expected_format = 'torch.distributed.checkpoint' if self._is_fsdp else 'cpu-test-state'
            if metadata.get('format') != expected_format:
                raise ValueError('Checkpoint backend format does not match the current model.')
            for name in ('max_length', 'max_new_tokens', 'max_grad_norm',
                         'eval_batch_size', 'learning_rate', 'weight_decay'):
                if metadata.get(name) != getattr(self, name):
                    raise ValueError(f'Resume configuration differs: {name}')
            for name, legacy_default in (
                ('training_loss', 'token_mean'), ('current_loss_weight', 0.5),
                ('training_views', 1), ('training_views_file_sha256', None),
                ('parameter_mode', 'full'), ('lora_rank', 16), ('lora_alpha', 32), ('lora_dropout', 0.0),
            ):
                if metadata.get(name, legacy_default) != getattr(self, name):
                    raise ValueError(f'Resume configuration differs: {name}')
            if self.training_views > 1 and metadata.get('training_view_version') != TRAINING_VIEW_VERSION:
                raise ValueError('Resume configuration differs: training_view_version')
            # Missing fields identify the historical semantics. Require an
            # explicit legacy choice when it can change future gradients.
            # With one view / role_mean these respective options are inert.
            if self.training_views > 1 and metadata.get('training_view_policy', 'legacy_wrapper') != self.training_view_policy:
                raise ValueError('Resume configuration differs: training_view_policy (old checkpoints require legacy_wrapper)')
            if self.training_loss == 'token_mean' and metadata.get('history_only_loss_policy', 'legacy_fact_mean') != self.history_only_loss_policy:
                raise ValueError('Resume configuration differs: history_only_loss_policy (old checkpoints require legacy_fact_mean)')
            for rank in range(self.world_size):
                if not (source / f'rng-rank-{rank}.pt').is_file():
                    raise ValueError(f'Checkpoint is missing RNG state for rank {rank}.')
            if not (source / 'extra_state.pt').is_file():
                raise ValueError('Checkpoint is missing the external writer state.')
            return metadata

        metadata = self._rank_zero_io(inspect)
        if self._is_fsdp:
            reader = FileSystemReader(str(source / 'distributed'))
            with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT,
                                     ShardedStateDictConfig(offload_to_cpu=True),
                                     ShardedOptimStateDictConfig(offload_to_cpu=True)):
                state = {'model': self.model.state_dict()}
                load_state_dict(state, storage_reader=reader, process_group=self._checkpoint_pg)
                self.model.load_state_dict(state['model'])
                optimizer_state = load_sharded_optimizer_state_dict(
                    model_state_dict=state['model'], optimizer_key='optimizer',
                    storage_reader=reader)['optimizer']
                self.optimizer.load_state_dict(FSDP.optim_state_dict_to_load(
                    self.model, self.optimizer, optimizer_state))
        else:
            state = torch.load(source / f'cpu-state-rank-{self.rank}.pt', map_location='cpu', weights_only=False)
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
        extra_state = self._rank_zero_io(
            lambda: torch.load(source / 'extra_state.pt', map_location='cpu', weights_only=False))
        self.total_optimizer_steps = int(metadata['total_optimizer_steps'])
        self.forward_accounting = dict(metadata['forward_accounting'])
        self.initial_parameter_fingerprint = metadata['initial_parameter_fingerprint']
        # Preserve current, validated descriptions when an older checkpoint
        # lacks policy metadata (or an inactive option differs).
        objective_metadata = {key: self.metadata[key] for key in (
            'history_only_loss_policy', 'history_only_objective', 'training_view_policy',
            'training_view_source', 'training_view_limitations')}
        self.metadata.update(metadata.get('backend_metadata', {}))
        self.metadata.update(objective_metadata)
        self.metadata['optimizer_state_dtypes'] = self._optimizer_dtypes()
        self._barrier()
        self._restore_rng(torch.load(source / f'rng-rank-{self.rank}.pt', map_location='cpu', weights_only=False))
        return extra_state
