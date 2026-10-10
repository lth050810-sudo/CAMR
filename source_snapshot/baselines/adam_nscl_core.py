"""Source-aligned Adam-NSCL math for Linear weights stored as [out, in].

Reference: ShipengWang/Adam-NSCL, a2f39b4273aa300739c460b33a5e8d2c674632b2,
optim/adam_svd.py. This module does not supply a Qwen3/WikiBigEdit preset.
The official optimizer projects the Adam UPDATE, on the RIGHT, using a
Frobenius-normalized input-covariance nullspace transform. Projecting the
gradient before Adam is a different algorithm and need not preserve features.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch


UPSTREAM_REPOSITORY = "ShipengWang/Adam-NSCL"
UPSTREAM_COMMIT = "a2f39b4273aa300739c460b33a5e8d2c674632b2"


def source_transform(covariance: torch.Tensor, threshold: float = 10.0):
    """Reproduce upstream get_eigens/get_transforms on an input covariance.

    Singular values <= threshold * smallest singular value are retained.
    The projection is divided by its Frobenius norm, exactly as in the source.
    This is NOT CrispEdit's cumulative-energy threshold selection.
    """
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Input covariance must be square")
    if not math.isfinite(threshold) or threshold < 1:
        raise ValueError("threshold must be finite and at least one")
    if not torch.isfinite(covariance).all():
        raise ValueError("Input covariance must be finite")
    _, singular, vh = torch.linalg.svd(covariance, full_matrices=True)
    selected = singular <= singular[-1] * threshold
    basis = vh.mT[:, selected]
    transform = basis @ basis.mT
    transform = transform / torch.linalg.vector_norm(transform)
    return transform, {
        "threshold": threshold,
        "selected_rank": int(selected.sum()),
        "input_dimension": covariance.shape[0],
        "min_singular_value": float(singular[-1]),
        "max_singular_value": float(singular[0]),
        "frobenius_normalized": True,
    }


class AdamNSCL(torch.optim.Optimizer):
    """Pinned-source Adam update followed by the input-space transform.

    Optimizer states use the parameter dtype, as in upstream. For mixed
    precision, the caller must explicitly decide any state-precision change.
    """

    def __init__(self, params, transforms: Mapping, *, lr=1e-3,
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if not math.isfinite(lr) or lr < 0 or not math.isfinite(eps) or eps < 0:
            raise ValueError("Invalid learning rate or epsilon")
        if any(not 0 <= beta < 1 for beta in betas):
            raise ValueError("Invalid Adam betas")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("Invalid weight decay")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                     weight_decay=weight_decay))
        self.set_transforms(transforms)

    def set_transforms(self, transforms: Mapping):
        resolved = {}
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim != 2:
                    raise ValueError("AdamNSCL requires a [out, in] Linear weight")
                if parameter not in transforms:
                    raise ValueError("Missing input-space transform")
                transform = transforms[parameter]
                width = parameter.shape[1]
                if transform.shape != (width, width):
                    raise ValueError("Transform must match the input dimension")
                if not torch.isfinite(transform).all():
                    raise ValueError("Transform must be finite")
                resolved[parameter] = transform.to(parameter).detach()
        self.transforms = resolved

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("AdamNSCL does not support sparse gradients")
                grad = parameter.grad
                if group["weight_decay"]:
                    grad = grad.add(parameter, alpha=group["weight_decay"])
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                mean, square = state["exp_avg"], state["exp_avg_sq"]
                mean.mul_(beta1).add_(grad, alpha=1 - beta1)
                square.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denominator = square.sqrt().add_(group["eps"])
                scale = group["lr"] * math.sqrt(1 - beta2 ** state["step"])
                scale /= 1 - beta1 ** state["step"]
                update = (-scale * mean / denominator) @ self.transforms[parameter]
                if not torch.isfinite(update).all():
                    raise FloatingPointError("Non-finite AdamNSCL update")
                parameter.add_(update)
        return loss
