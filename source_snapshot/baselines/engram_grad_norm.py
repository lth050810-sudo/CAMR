"""Stable-order global clipping for the full writer's uniform FSDP shards.

PyTorch 2.3 FSDP.clip_grad_norm_ gathers parameters in Python sets before its
FP32 norm reduction. Set iteration follows object identity, so independent
process launches can sum identical per-parameter norms in different orders.
A one-ULP clipping difference can later cross a BF16 rounding boundary.

This narrow replacement retains FSDP's global reduction and clipping formula,
but traverses named parameters in lexical order. It supports only a root FSDP
model whose parameters all use FULL_SHARD, one process group, no CPU parameter
offload, and use_orig_params=False, matching the full writer's configuration.
It does not promise determinism of attention kernels or other collectives.
"""

from __future__ import annotations

import math
import warnings
from functools import reduce
from typing import Iterable

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp._common_utils import TrainingState
from torch.distributed.fsdp._runtime_utils import _lazy_init


def ordered_local_grad_norm(
    named_parameters: Iterable[tuple[str, nn.Parameter]], norm_type: float = 2.0,
    *, device: torch.device | None = None,
) -> torch.Tensor:
    """FP32 norm with a fixed parameter order; does not perform communication."""
    norm_type = float(norm_type)
    if norm_type not in (2.0, math.inf):
        raise ValueError('Only L2 and infinity gradient norms are supported.')
    gradients = [parameter.grad for _, parameter in sorted(named_parameters, key=lambda item: item[0])
                 if parameter.grad is not None]
    if not gradients:
        return torch.tensor(0.0, dtype=torch.float32, device=device)
    if len({gradient.dtype for gradient in gradients}) != 1:
        raise ValueError('Uniform gradient dtype is required, matching PyTorch 2.3 FSDP.')
    if any(gradient.is_sparse for gradient in gradients):
        raise ValueError('Sparse gradients are not supported by the full writer.')
    return torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(gradient.detach(), norm_type, dtype=torch.float32)
                     for gradient in gradients]),
        norm_type, dtype=torch.float32,
    )


def _validated_parameters(model):
    if not isinstance(model, FSDP):
        raise TypeError('Global shard clipping requires the root FSDP model.')
    _lazy_init(model, model)
    if not model._is_root:
        raise ValueError('Call global gradient clipping on the root FSDP model.')
    model._assert_state(TrainingState.IDLE)
    wrappers = FSDP.fsdp_modules(model)
    for wrapper in wrappers:
        if wrapper.sharding_strategy not in (ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP):
            raise ValueError('Stable shard clipping requires uniform FULL_SHARD or SHARD_GRAD_OP.')
        if wrapper.sharding_strategy != model.sharding_strategy:
            raise ValueError('Stable shard clipping requires one uniform sharding strategy.')
        if wrapper.process_group is not model.process_group:
            raise ValueError('All FSDP wrappers must use the same process group.')
        if wrapper.cpu_offload.offload_params:
            raise ValueError('CPU parameter offload is outside this helper\'s supported configuration.')
    managed_ids = set()
    for handle in model._all_handles:
        if handle._use_orig_params:
            raise ValueError('Stable shard clipping requires use_orig_params=False.')
        if not handle.uses_sharded_strategy:
            raise ValueError('Every FSDP handle must have sharded parameters.')
        managed_ids.add(id(handle.flat_param))
    parameters = sorted(model.named_parameters(), key=lambda item: item[0])
    if {id(parameter) for _, parameter in parameters} != managed_ids:
        raise ValueError('Unmanaged, shared, or missing FSDP flat parameters are unsupported.')
    return parameters


@torch.no_grad()
def clip_full_shard_grad_norm_(
    model: FSDP, max_norm: float, norm_type: float = 2.0,
) -> torch.Tensor:
    """Clip the globally reconstructed norm, collectively on every rank.

    Each local parameter contains a disjoint shard after backward. Summing
    squared local norms (without dividing by world size) reconstructs the L2
    norm of the whole model. All ranks participate even when locally empty.
    The result uses the promoted local gradient dtype, or FP32 for no gradients,
    as in FSDP.clip_grad_norm_. The coefficient includes FSDP's 1e-6 epsilon.
    """
    max_norm, norm_type = float(max_norm), float(norm_type)
    if not math.isfinite(max_norm) or max_norm < 0:
        raise ValueError('max_norm must be finite and nonnegative.')
    if norm_type not in (2.0, math.inf):
        raise ValueError('Only L2 and infinity gradient norms are supported.')
    parameters = _validated_parameters(model)
    gradients = [parameter.grad for _, parameter in parameters if parameter.grad is not None]
    local_norm = ordered_local_grad_norm(parameters, norm_type, device=model.compute_device).to(model.compute_device)
    if norm_type == math.inf:
        total_norm = local_norm
        dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=model.process_group)
    else:
        total_norm = local_norm.square()
        dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=model.process_group)
        total_norm = total_norm.sqrt()
    clip_coefficient = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    for gradient in gradients:
        gradient.detach().mul_(clip_coefficient.to(device=gradient.device, dtype=gradient.dtype))
    if not gradients:
        warnings.warn('This FSDP rank has no gradients; returning the collective norm in FP32.',
                      RuntimeWarning, stacklevel=2)
        return total_norm
    return total_norm.to(reduce(torch.promote_types, (gradient.dtype for gradient in gradients)))
