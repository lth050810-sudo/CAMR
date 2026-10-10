"""Bounded-device-memory UltraEdit ridge updates for complete source stages.

The feature cache and cumulative RunningMoments are produced by the original
runner without modification. For each token block, this adapter constructs
the original pseudo-value V and accumulates raw K.T @ K and K.T @ V. Thus it
solves the same ridge objective, without transferring an entire stage to CUDA.

Every requested solver is evaluated in primal form. Primal and dual are
algebraically equivalent for positive ridge, but block accumulation and a
change from dual to primal alter floating-point operation order. Module logs
record both changes; bitwise identity is not claimed.
"""
from __future__ import annotations

import math
from typing import Any

import torch

from baselines.run_ultraedit import RunningMoments, get_module


DEFAULT_TOKEN_BLOCK_SIZE = 4096


def concatenate_stage_features(
    captured_keys: dict[str, list[torch.Tensor]],
    captured_grads: dict[str, list[torch.Tensor]],
    module_names: list[str],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Join cached CPU chunks, releasing each original list immediately.

    This replaces only the final dict comprehension in cache_stage_features.
    Key/gradient values and token order are identical to the original torch.cat
    calls. The input dictionaries/lists are consumed after feature extraction;
    the resulting stage feature tensors remain immutable solver inputs.

    Logical tensor storage peaks at one complete feature cache plus the largest
    individual K or G matrix, rather than two complete caches. Process RSS may
    also include allocator retention, model/tokenizer state, and other jobs.
    """
    if set(module_names) != set(captured_keys) or set(module_names) != set(captured_grads):
        raise ValueError('captured module keys/gradients do not match module_names')
    if len(module_names) != len(set(module_names)):
        raise ValueError('duplicate module names')
    features = {}
    for name in module_names:
        key_chunks = captured_keys.pop(name)
        keys = torch.cat(key_chunks, dim=0)
        key_chunks.clear()
        del key_chunks
        grad_chunks = captured_grads.pop(name)
        grads = torch.cat(grad_chunks, dim=0)
        grad_chunks.clear()
        del grad_chunks
        features[name] = (keys, grads)
    return features


def _validate_features(keys: torch.Tensor, grads: torch.Tensor) -> None:
    if keys.ndim != 2 or grads.ndim != 2 or keys.shape[0] != grads.shape[0]:
        raise ValueError('keys and gradients must be matrices with equal token counts')
    if keys.shape[1] < 1 or grads.shape[1] < 1:
        raise ValueError('feature dimensions must be positive')
    if keys.device.type != 'cpu' or grads.device.type != 'cpu':
        raise ValueError('full-stage features must remain in the CPU cache')
    if keys.dtype != grads.dtype or keys.dtype not in (torch.float32, torch.float64):
        raise ValueError('feature cache must use one FP32 or FP64 dtype')


@torch.no_grad()
def solve_ultraedit_primal_blocks(
    keys_cpu: torch.Tensor,
    grads_cpu: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    *,
    learning_rate: float,
    ridge: float,
    token_block_size: int = DEFAULT_TOKEN_BLOCK_SIZE,
) -> torch.Tensor:
    """Return the original UltraEdit shift from token-block normal equations.

    V_i = -lr * sum(((K_i - mean_K) / std_K)**2) *
          ((G_i - mean_G) / std_G)
    shift = solve(sum(K_i.T @ K_i) + ridge*I, sum(K_i.T @ V_i))

    The equations use unnormalized K. The CPU features and supplied moments
    remain unchanged. Device memory scales with the feature dimensions and
    block size, not the number of cached stage tokens.
    """
    _validate_features(keys_cpu, grads_cpu)
    if not isinstance(token_block_size, int) or token_block_size < 1:
        raise ValueError('token_block_size must be a positive integer')
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError('ridge must be positive and finite')
    if not math.isfinite(learning_rate):
        raise ValueError('learning_rate must be finite')
    key_size, grad_size = keys_cpu.shape[1], grads_cpu.shape[1]
    if mean.ndim != 1 or std.ndim != 1 or mean.numel() != key_size + grad_size or std.shape != mean.shape:
        raise ValueError('normalizer dimensions do not match the cached keys and gradients')
    device = torch.device(device)
    mean = mean.detach().to(device=device, dtype=keys_cpu.dtype)
    std = std.detach().to(device=device, dtype=keys_cpu.dtype)
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()) or not bool((std > 0).all()):
        raise ValueError('normalizer mean/std must be finite with positive std')
    gram = torch.zeros((key_size, key_size), device=device, dtype=keys_cpu.dtype)
    rhs = torch.zeros((key_size, grad_size), device=device, dtype=keys_cpu.dtype)
    for start in range(0, keys_cpu.shape[0], token_block_size):
        # Transfer only this block. Arithmetic below is out-of-place so CPU
        # execution cannot accidentally modify a slice of the feature cache.
        keys = keys_cpu[start:start + token_block_size].to(device)
        grads = grads_cpu[start:start + token_block_size].to(device)
        normalized_keys = (keys - mean[:key_size]) / std[:key_size]
        normalized_grads = (grads - mean[key_size:]) / std[key_size:]
        coefficients = -learning_rate * normalized_keys.square().sum(dim=-1, keepdim=True)
        pseudo_values = coefficients * normalized_grads
        gram.addmm_(keys.T, keys)
        rhs.addmm_(keys.T, pseudo_values)
        del keys, grads, normalized_keys, normalized_grads, coefficients, pseudo_values
    gram.diagonal().add_(ridge)
    if not bool(torch.isfinite(gram).all()) or not bool(torch.isfinite(rhs).all()):
        raise FloatingPointError('nonfinite UltraEdit normal equations; weights were not updated')
    shift = torch.linalg.solve(gram, rhs)
    if not bool(torch.isfinite(shift).all()):
        raise FloatingPointError('nonfinite UltraEdit shift; weights were not updated')
    return shift


@torch.no_grad()
def apply_ultraedit_stage(
    model: torch.nn.Module,
    features: dict[str, tuple[torch.Tensor, torch.Tensor]],
    normalizer: RunningMoments,
    device: torch.device,
    *,
    learning_rate: float,
    ridge: float,
    solver: str,
    epsilon: float,
    token_block_size: int = DEFAULT_TOKEN_BLOCK_SIZE,
) -> list[dict[str, Any]]:
    """Drop-in stage update; only the original solver's storage order changes."""
    if solver not in ('auto', 'primal', 'dual'):
        raise ValueError(f'unknown solver: {solver}')
    if normalizer.mean is None:
        raise RuntimeError('normalizer was not updated')
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('normalizer epsilon must be positive and finite')
    std = normalizer.std(epsilon)
    module_log = []
    for name, (keys_cpu, grads_cpu) in features.items():
        module = get_module(model, name)
        _validate_features(keys_cpu, grads_cpu)
        if tuple(module.weight.shape) != (grads_cpu.shape[1], keys_cpu.shape[1]):
            raise ValueError(f'cached feature dimensions differ from weight shape: {name}')
        shift = solve_ultraedit_primal_blocks(
            keys_cpu, grads_cpu, normalizer.mean, std, device,
            learning_rate=learning_rate, ridge=ridge, token_block_size=token_block_size,
        )
        applied = shift.T.to(module.weight.dtype)
        module.weight.add_(applied)
        original_solver = ('dual' if keys_cpu.shape[0] < keys_cpu.shape[1] else 'primal') if solver == 'auto' else solver
        module_log.append({
            'module': name, 'tokens': int(keys_cpu.shape[0]), 'solver': 'primal',
            'requested_solver': solver, 'original_selected_solver': original_solver,
            'shift_frobenius_norm': float(shift.norm()),
            'applied_nonzero': int(torch.count_nonzero(applied)),
            'token_block_size': token_block_size,
            'token_blocks': (int(keys_cpu.shape[0]) + token_block_size - 1) // token_block_size,
            'scale_adapter': 'ultraedit-full-scale-primal-blocks/v1',
            'normal_equations': 'sum raw K.T@K and raw K.T@V; original normalized pseudo-value V',
            'floating_point_note': 'token-block accumulation changes reduction order; requested dual is evaluated via the equivalent primal identity; no bitwise identity claim',
            'cpu_feature_cache_modified': False,
        })
        del shift, applied
        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()
    return module_log
