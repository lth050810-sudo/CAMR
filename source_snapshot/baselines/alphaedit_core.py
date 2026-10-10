#!/usr/bin/env python3
"""Auditable linear-algebra and cache helpers for the Qwen3 AlphaEdit port.

AlphaEdit constrains every ``down_proj`` update to the low-eigenvalue
subspace of a checkpoint-specific input second moment.  The upstream code
materializes the dense projector ``P = U_null U_null.T``.  Qwen3-8B has a
12,288-dimensional ``down_proj`` input, so this port stores the orthonormal
null-space basis instead.  Solving in that basis is algebraically equivalent
to the upstream dense equation and avoids silently reusing a Qwen2.5
``null_space_project.pt`` file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


PROJECTION_SCHEMA = "engram-alphaedit-nullspace/v1"
UPSTREAM_REPOSITORY = "jianghoucheng/AlphaEdit"
UPSTREAM_COMMIT = "b84624f44dfe8fc6cd9e41df916c44124a0c46dc"
EASYEDIT_REPOSITORY = "zjunlp/EasyEdit"
EASYEDIT_COMMIT = "14cea8245f06715684592ab55184939b99d70784"
REWRITE_MODULE_TEMPLATE = "model.layers.{}.mlp.down_proj"

UPSTREAM_QWEN25_PRESET: dict[str, Any] = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "layers": [4, 5, 6, 7, 8],
    "clamp_norm_factor": 4.0,
    "fact_token": "subject_last",
    "v_num_grad_steps": 25,
    "v_lr": 5e-1,
    "v_loss_layer": 27,
    "v_weight_decay": 1e-3,
    "kl_factor": 0.0625,
    "mom2_dataset": "wikipedia",
    "mom2_hf_dataset_at_easyedit_commit": "wikimedia/wikipedia",
    "mom2_hf_config_at_easyedit_commit": "20231101.en",
    "mom2_n_samples": 100000,
    "mom2_dtype": "float32",
    "nullspace_threshold": 2e-2,
    "L2": 1.0,
    "rewrite_module_tmp": REWRITE_MODULE_TEMPLATE,
}

LICENSE_PROVENANCE: dict[str, Any] = {
    "AlphaEdit": {
        "license": "MIT",
        "copyright": "Copyright (c) 2025 jianghc",
        "repository": UPSTREAM_REPOSITORY,
        "commit": UPSTREAM_COMMIT,
    },
    "EasyEdit": {
        "license": "MIT template at audited commit",
        "note": "upstream LICENSE copyright fields are placeholders",
        "repository": EASYEDIT_REPOSITORY,
        "commit": EASYEDIT_COMMIT,
    },
    "MEMIT": {
        "license": "MIT",
        "copyright": "Copyright (c) 2022 Kevin Meng",
        "note": "AlphaEdit states that its implementation is based on MEMIT and EMMET",
    },
    "statistics_corpus": {
        "license_note": (
            "Wikipedia data licensing is separate from code licensing; retain the "
            "Wikimedia/Hugging Face dataset version and attribution when distributing "
            "a frozen raw corpus sample"
        )
    },
}


def module_name_for_layer(layer: int) -> str:
    return REWRITE_MODULE_TEMPLATE.format(int(layer))


def projection_filename(layer: int) -> str:
    return f"layer_{int(layer)}_alphaedit_projection.pt"


def compute_nullspace_basis(
    moment: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all eigenvalues and the upstream absolute-threshold null basis.

    The empirical second moment is symmetric up to accumulation roundoff, so
    ``eigh`` is both cheaper and more stable than the upstream generic SVD.
    For a symmetric positive-semidefinite matrix its left singular vectors
    and eigenvectors span the same selected subspace.
    """

    if moment.ndim != 2 or moment.shape[0] != moment.shape[1]:
        raise ValueError(f"AlphaEdit moment must be square, got {tuple(moment.shape)}")
    if threshold < 0 or not math.isfinite(threshold):
        raise ValueError("null-space threshold must be finite and non-negative")
    if not torch.isfinite(moment).all():
        raise ValueError("AlphaEdit moment contains non-finite values")
    symmetric = (moment.float() + moment.float().T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    # Upstream uses SVD and thresholds singular values.  For an exact PSD
    # moment this equals thresholding eigenvalues, but abs() preserves the SVD
    # behavior when finite-precision accumulation yields a small negative
    # eigenvalue and avoids accepting a materially negative direction.
    selected = eigenvalues.abs() < threshold
    return eigenvalues, eigenvectors[:, selected].contiguous()


def projector_from_basis(basis: torch.Tensor) -> torch.Tensor:
    if basis.ndim != 2:
        raise ValueError("null-space basis must be a matrix")
    return basis @ basis.T


def orthogonality_probe_error(basis: torch.Tensor, max_columns: int = 64) -> float:
    """Cheap deterministic integrity check for a potentially very wide basis."""

    if basis.ndim != 2:
        raise ValueError("null-space basis must be a matrix")
    width = int(basis.shape[1])
    if width == 0:
        return 0.0
    count = min(width, max_columns)
    indices = torch.linspace(0, width - 1, count, dtype=torch.float64)
    indices = indices.round().to(torch.long).unique()
    probe = basis[:, indices].float()
    gram = probe.T @ probe
    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    return float((gram - identity).abs().max())


def projected_alphaedit_solve(
    basis: torch.Tensor,
    keys: torch.Tensor,
    residual: torch.Tensor,
    history_gram: torch.Tensor,
    l2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the AlphaEdit update in null-space coordinates.

    Args:
        basis: Orthonormal ``[input_dim, nullity]`` matrix ``U``.
        keys: Current edit keys ``[input_dim, edits]``.
        residual: Desired output residuals ``[output_dim, edits]``.
        history_gram: ``U.T @ cache_c @ U`` from earlier edit batches.
        l2: The upstream diagonal stabilization constant.

    Returns:
        ``(update, projected_keys)`` where ``update`` has the same orientation
        as a PyTorch Linear weight: ``[output_dim, input_dim]``.

    The upstream equation is

    ``(P (K K.T + C) + l2 I) X = P K R.T`` with ``P = U U.T``.

    Its component orthogonal to ``U`` is zero, leaving the reduced system
    ``(U.T (K K.T + C) U + l2 I) Y = U.T K R.T`` and ``X = U Y``.
    """

    if basis.ndim != 2 or keys.ndim != 2 or residual.ndim != 2:
        raise ValueError("basis, keys, and residual must be matrices")
    input_dim, nullity = basis.shape
    if keys.shape[0] != input_dim:
        raise ValueError(f"key dimension {keys.shape[0]} != basis dimension {input_dim}")
    if residual.shape[1] != keys.shape[1]:
        raise ValueError("residual and keys must describe the same number of edits")
    if tuple(history_gram.shape) != (nullity, nullity):
        raise ValueError(
            f"history Gram shape {tuple(history_gram.shape)} != {(nullity, nullity)}"
        )
    if l2 <= 0 or not math.isfinite(l2):
        raise ValueError("AlphaEdit L2 must be finite and positive")

    solve_device = keys.device
    solve_dtype = torch.float32
    basis_f = basis.to(device=solve_device, dtype=solve_dtype)
    keys_f = keys.to(dtype=solve_dtype)
    residual_f = residual.to(device=solve_device, dtype=solve_dtype)
    projected_keys = basis_f.T @ keys_f
    if nullity == 0:
        return torch.zeros(
            residual.shape[0], input_dim, device=solve_device, dtype=solve_dtype
        ), projected_keys
    system = projected_keys @ projected_keys.T
    system.add_(history_gram.to(device=solve_device, dtype=solve_dtype))
    system.diagonal().add_(float(l2))
    rhs = projected_keys @ residual_f.T
    coefficients = torch.linalg.solve(system, rhs)
    update = (basis_f @ coefficients).T.contiguous()
    return update, projected_keys


def full_projector_reference_solve(
    basis: torch.Tensor,
    keys: torch.Tensor,
    residual: torch.Tensor,
    history_keys: torch.Tensor,
    l2: float,
) -> torch.Tensor:
    """Literal upstream dense solve, retained as a CPU test oracle."""

    dtype = torch.float64
    basis = basis.to(dtype)
    keys = keys.to(dtype)
    residual = residual.to(dtype)
    history_keys = history_keys.to(dtype)
    projector = projector_from_basis(basis)
    cache = history_keys @ history_keys.T
    system = projector @ (keys @ keys.T + cache)
    system.diagonal().add_(float(l2))
    rhs = projector @ keys @ residual.T
    return torch.linalg.solve(system, rhs).T


@dataclass
class ProjectedHistory:
    """Exact low-rank representation of AlphaEdit's historical ``cache_c``.

    The upstream cache is ``C = H H.T`` in the original feature space.  Only
    ``U.T C U`` can affect a null-space-constrained update, so this class
    stores ``U.T H`` plus an incrementally maintained Cholesky factor of
    ``I + H.T H / l2``.  Applying the historical inverse to a new edit batch
    then uses Woodbury with a right-hand side of only ``batch_size`` columns;
    no ``nullity x nullity`` factorization is needed per edit.
    """

    nullity: int
    l2: float
    device: torch.device
    keys: torch.Tensor
    cholesky: torch.Tensor
    cholesky_retry_count: int
    max_cholesky_jitter: float

    @classmethod
    def empty(
        cls, nullity: int, l2: float, device: torch.device
    ) -> "ProjectedHistory":
        if nullity < 0:
            raise ValueError("nullity must be non-negative")
        if l2 <= 0 or not math.isfinite(l2):
            raise ValueError("AlphaEdit L2 must be finite and positive")
        return cls(
            nullity=int(nullity),
            l2=float(l2),
            device=device,
            # Long histories make the Woodbury subtraction poorly conditioned
            # in float32.  Keep the compact history state in the same double
            # precision used by the upstream reference solve; model features
            # and weight updates remain float32/bfloat16.
            keys=torch.empty((nullity, 0), dtype=torch.float64, device=device),
            cholesky=torch.empty((0, 0), dtype=torch.float64, device=device),
            cholesky_retry_count=0,
            max_cholesky_jitter=0.0,
        )

    @property
    def count(self) -> int:
        return int(self.keys.shape[1])

    def _validate_keys(self, keys: torch.Tensor) -> torch.Tensor:
        if keys.ndim != 2 or keys.shape[0] != self.nullity:
            raise ValueError(
                f"projected keys must have shape [{self.nullity}, n], got "
                f"{tuple(keys.shape)}"
            )
        return keys.to(device=self.device, dtype=torch.float64)

    def _stable_cholesky(self, matrix: torch.Tensor, *, label: str) -> torch.Tensor:
        """Factor a theoretically SPD matrix with bounded roundoff recovery."""

        symmetric = (matrix + matrix.T) * 0.5
        if not torch.isfinite(symmetric).all():
            raise RuntimeError(f"AlphaEdit {label} contains non-finite values")
        factor, info = torch.linalg.cholesky_ex(symmetric, check_errors=False)
        if int(info.item()) == 0:
            return factor

        diagonal_scale = max(
            1.0,
            float(torch.max(torch.abs(torch.diagonal(symmetric))).item()),
        )
        epsilon = torch.finfo(symmetric.dtype).eps
        last_info = int(info.item())
        for multiplier in (16.0, 256.0, 4096.0, 65536.0):
            jitter = diagonal_scale * epsilon * multiplier
            candidate = symmetric.clone()
            candidate.diagonal().add_(jitter)
            factor, info = torch.linalg.cholesky_ex(candidate, check_errors=False)
            last_info = int(info.item())
            if last_info == 0:
                self.cholesky_retry_count += 1
                self.max_cholesky_jitter = max(self.max_cholesky_jitter, jitter)
                return factor
        raise RuntimeError(
            f"AlphaEdit {label} remained non-positive-definite after bounded "
            f"float64 jitter retries (last leading minor {last_info})"
        )

    def base_inverse(self, keys: torch.Tensor) -> torch.Tensor:
        """Return ``(l2 I + H H.T)^-1 keys`` exactly via Woodbury."""

        keys = self._validate_keys(keys)
        inverse = keys / self.l2
        if self.count == 0 or self.nullity == 0:
            return inverse
        cross = self.keys.T @ keys
        correction = torch.cholesky_solve(cross, self.cholesky)
        return inverse - (self.keys @ correction) / (self.l2 * self.l2)

    def adjusted_keys(self, keys: torch.Tensor) -> torch.Tensor:
        """Return ``(l2 I + H H.T + K K.T)^-1 K``."""

        keys = self._validate_keys(keys)
        if keys.shape[1] == 0:
            return keys
        base_inverse_keys = self.base_inverse(keys)
        small = torch.eye(
            keys.shape[1], dtype=torch.float64, device=self.device
        ) + keys.T @ base_inverse_keys
        factor = self._stable_cholesky(small, label="current-batch Woodbury system")
        return torch.cholesky_solve(base_inverse_keys.T, factor).T

    def append(self, keys: torch.Tensor) -> None:
        """Append projected post-edit keys and update the dual Cholesky."""

        keys = self._validate_keys(keys)
        width = int(keys.shape[1])
        if width == 0 or self.nullity == 0:
            return
        diagonal = torch.eye(width, dtype=torch.float64, device=self.device)
        diagonal.add_(keys.T @ keys / self.l2)
        if self.count == 0:
            self.cholesky = self._stable_cholesky(
                diagonal, label="initial history block"
            )
            self.keys = keys.contiguous()
            return
        cross = self.keys.T @ keys / self.l2
        lower_left_transpose = torch.linalg.solve_triangular(
            self.cholesky, cross, upper=False
        )
        schur = diagonal - lower_left_transpose.T @ lower_left_transpose
        new_diagonal = self._stable_cholesky(
            schur, label="incremental history Schur complement"
        )
        old_count = self.count
        expanded = torch.zeros(
            old_count + width,
            old_count + width,
            dtype=torch.float64,
            device=self.device,
        )
        expanded[:old_count, :old_count] = self.cholesky
        expanded[old_count:, :old_count] = lower_left_transpose.T
        expanded[old_count:, old_count:] = new_diagonal
        self.keys = torch.cat((self.keys, keys), dim=1)
        self.cholesky = expanded


def validate_projection_payload(
    payload: Any,
    *,
    layer: int,
    module_name: str,
    input_dimension: int,
    nullspace_threshold: float,
    model_fingerprint: str,
    model_config_sha256: str,
    sample_size: int,
    allow_diagnostic: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Validate an AlphaEdit cache before any tensor reaches the model.

    Bare tensors and the upstream ``null_space_project.pt`` layout are
    intentionally rejected: neither carries enough provenance to establish
    that it was computed from this Qwen3 checkpoint.
    """

    if not isinstance(payload, dict):
        raise TypeError("AlphaEdit projection must be a metadata-bearing dictionary")
    metadata = payload.get("metadata")
    basis = payload.get("basis")
    if not isinstance(metadata, dict) or not isinstance(basis, torch.Tensor):
        raise ValueError("AlphaEdit projection requires tensor `basis` and dict `metadata`")
    expected = {
        "schema_version": PROJECTION_SCHEMA,
        "model_type": "qwen3",
        "model_fingerprint": model_fingerprint,
        "model_config_sha256": model_config_sha256,
        "layer": int(layer),
        "module_name": module_name,
        "feature_dimension": int(input_dimension),
        "sample_size_requested": int(sample_size),
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"AlphaEdit projection metadata mismatch: {mismatches}")
    recorded_threshold = metadata.get("nullspace_threshold")
    if recorded_threshold is None or not math.isclose(
        float(recorded_threshold), float(nullspace_threshold), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "AlphaEdit projection threshold mismatch: "
            f"{recorded_threshold!r} != {nullspace_threshold!r}"
        )
    if basis.ndim != 2 or basis.shape[0] != input_dimension:
        raise ValueError(
            f"AlphaEdit basis shape {tuple(basis.shape)} does not start with {input_dimension}"
        )
    if int(metadata.get("nullity", -1)) != basis.shape[1]:
        raise ValueError("AlphaEdit basis nullity does not match metadata")
    if not torch.isfinite(basis).all():
        raise ValueError("AlphaEdit basis contains non-finite values")
    orthogonality_error = orthogonality_probe_error(basis)
    if orthogonality_error > 5e-3:
        raise ValueError(
            "AlphaEdit basis failed its orthogonality integrity probe: "
            f"max error {orthogonality_error:.6g}"
        )
    corpus = metadata.get("corpus")
    if not isinstance(corpus, dict) or not corpus.get("cache_sha256"):
        raise ValueError("AlphaEdit projection has no auditable corpus fingerprint")
    formal = metadata.get("formal_baseline_eligible") is True
    if not formal and not allow_diagnostic:
        raise ValueError(
            "AlphaEdit projection is marked diagnostic/non-publication; pass "
            "--allow-diagnostic-projection only for a compatibility smoke"
        )
    return basis.float(), metadata


def load_projection_payload(path: Path, **expected: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return validate_projection_payload(payload, **expected)
