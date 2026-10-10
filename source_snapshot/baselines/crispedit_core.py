#!/usr/bin/env python3
"""Auditable building blocks for a Qwen3 CrispEdit port.

The upstream CrispEdit implementation computes a Kronecker-factored curvature
approximation for each edited ``mlp.down_proj`` matrix.  This module keeps the
math independent of the upstream EasyEdit fork so that cached statistics can
be validated before a long WikiBigEdit run.

This is a porting aid, not a claim that the Qwen2.5 layer preset transfers to
Qwen3 unchanged.  Qwen3 statistics must be computed from Qwen3 activations and
gradients.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


STATS_FORMAT = "crispedit-kfac/v1"
PROJECTION_FORMAT = "crispedit-projection/v1"
UPSTREAM_REPOSITORY = "zarifikram/CrispEdit"
UPSTREAM_COMMIT = "09035f16695998f3a71ec6006245d99e8cc648c8"
REWRITE_MODULE_TEMPLATE = "model.layers.{}.mlp.down_proj.weight"


def parse_layers(value: str | Sequence[int]) -> list[int]:
    """Parse and validate a layer list while preserving its order."""

    if isinstance(value, str):
        raw = [item.strip() for item in value.split(",") if item.strip()]
        layers = [int(item) for item in raw]
    else:
        layers = [int(item) for item in value]
    if not layers:
        raise ValueError("at least one edited layer is required")
    if any(layer < 0 for layer in layers):
        raise ValueError(f"layer indices must be non-negative: {layers}")
    if len(layers) != len(set(layers)):
        raise ValueError(f"layer indices must be unique: {layers}")
    return layers


def module_name_for_layer(layer: int) -> str:
    return REWRITE_MODULE_TEMPLATE.format(layer).removesuffix(".weight")


def parameter_name_for_layer(layer: int) -> str:
    return REWRITE_MODULE_TEMPLATE.format(layer)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_identity(model_dir: Path) -> dict[str, Any]:
    """Return a cheap identity without hashing the 16 GB of weight shards."""

    model_dir = model_dir.resolve()
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    shard_inventory = [
        {"name": path.name, "bytes": path.stat().st_size}
        for path in sorted(model_dir.glob("*.safetensors"))
    ]
    return {
        "resolved_path": str(model_dir),
        "config_sha256": _sha256(config_path),
        "weight_index_sha256": _sha256(index_path),
        "weight_shards": shard_inventory,
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "hidden_size": config.get("hidden_size"),
        "intermediate_size": config.get("intermediate_size"),
        "vocab_size": config.get("vocab_size"),
    }


def identities_compatible(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    """Compare stable model fields and deliberately ignore the absolute path."""

    keys = (
        "config_sha256",
        "weight_index_sha256",
        "weight_shards",
        "model_type",
        "architectures",
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "vocab_size",
    )
    return all(expected.get(key) == actual.get(key) for key in keys)


def resolve_modules(
    model: torch.nn.Module,
    layers: Sequence[int],
) -> dict[int, torch.nn.Module]:
    modules = dict(model.named_modules())
    resolved: dict[int, torch.nn.Module] = {}
    for layer in layers:
        name = module_name_for_layer(layer)
        if name not in modules:
            nearby = [candidate for candidate in modules if f"layers.{layer}.mlp" in candidate]
            raise KeyError(f"missing CrispEdit module {name!r}; nearby={nearby[:12]}")
        module = modules[name]
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"{name} is {type(module).__name__}, expected torch.nn.Linear")
        resolved[layer] = module
    return resolved


def resolve_parameters(
    model: torch.nn.Module,
    layers: Sequence[int],
) -> dict[int, torch.nn.Parameter]:
    parameters = dict(model.named_parameters())
    result: dict[int, torch.nn.Parameter] = {}
    for layer in layers:
        name = parameter_name_for_layer(layer)
        if name not in parameters:
            nearby = [candidate for candidate in parameters if f"layers.{layer}.mlp" in candidate]
            raise KeyError(f"missing CrispEdit parameter {name!r}; nearby={nearby[:12]}")
        parameter = parameters[name]
        if parameter.ndim != 2:
            raise ValueError(f"{name} has shape {tuple(parameter.shape)}, expected a matrix")
        result[layer] = parameter
    return result


@dataclass
class KfacResult:
    A: torch.Tensor
    B: torch.Tensor
    num_documents: int
    num_tokens: int


def collect_kfac_stats(
    model: torch.nn.Module,
    layer_modules: Mapping[int, torch.nn.Module],
    batches: Iterable[Mapping[str, torch.Tensor]],
    device: torch.device,
    *,
    accumulation_dtype: torch.dtype = torch.float32,
) -> dict[int, KfacResult]:
    """Collect the same empirical-Fisher KFAC factors used by CrispEdit.

    ``A`` is the second moment of inputs to ``down_proj`` and ``B`` is the
    second moment of the loss gradient with respect to its output.  The loss
    is the summed next-token cross entropy over non-padding tokens, matching
    the upstream implementation's scale.
    """

    if not layer_modules:
        raise ValueError("no layers were supplied for KFAC collection")
    accumulators: dict[int, dict[str, torch.Tensor]] = {}
    captured_inputs: dict[int, torch.Tensor] = {}
    captured_grad_outputs: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    for layer, module in layer_modules.items():
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"layer {layer} is not linear: {type(module).__name__}")
        accumulators[layer] = {
            "A": torch.zeros(
                (module.in_features, module.in_features),
                dtype=accumulation_dtype,
                device=device,
            ),
            "B": torch.zeros(
                (module.out_features, module.out_features),
                dtype=accumulation_dtype,
                device=device,
            ),
        }

        def capture(
            _module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            layer_index: int = layer,
        ) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError("CrispEdit currently expects a tensor-valued linear module")
            captured_inputs[layer_index] = inputs[0].detach()
            if not output.requires_grad:
                output.requires_grad_(True)

            def capture_grad(gradient: torch.Tensor) -> None:
                captured_grad_outputs[layer_index] = gradient.detach()

            output.register_hook(capture_grad)

        handles.append(module.register_forward_hook(capture))

    requires_grad = {parameter: parameter.requires_grad for parameter in model.parameters()}
    was_training = model.training
    old_use_cache = getattr(model.config, "use_cache", None)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    if old_use_cache is not None:
        model.config.use_cache = False

    num_documents = 0
    num_tokens = 0
    try:
        with torch.enable_grad():
            for raw_batch in batches:
                batch = {
                    key: value.to(device, non_blocking=True)
                    for key, value in raw_batch.items()
                    if key in {"input_ids", "attention_mask"}
                }
                if "input_ids" not in batch:
                    raise KeyError("KFAC batch has no input_ids")
                if "attention_mask" not in batch:
                    batch["attention_mask"] = torch.ones_like(batch["input_ids"])
                if batch["input_ids"].shape[1] < 2:
                    continue
                captured_inputs.clear()
                captured_grad_outputs.clear()
                outputs = model(**batch, use_cache=False)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch["input_ids"][:, 1:].contiguous().clone()
                valid_mask = batch["attention_mask"][:, 1:].bool()
                shift_labels[~valid_mask] = -100
                valid_count = int(valid_mask.sum())
                if valid_count == 0:
                    continue
                loss = F.cross_entropy(
                    shift_logits.float().view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                loss.backward()

                missing = [
                    layer
                    for layer in layer_modules
                    if layer not in captured_inputs or layer not in captured_grad_outputs
                ]
                if missing:
                    raise RuntimeError(f"KFAC hooks did not capture layers: {missing}")
                with torch.no_grad():
                    for layer in layer_modules:
                        features = captured_inputs[layer][:, :-1, :]
                        gradients = captured_grad_outputs[layer][:, :-1, :]
                        feature_rows = features[valid_mask].to(accumulation_dtype)
                        gradient_rows = gradients[valid_mask].to(accumulation_dtype)
                        accumulators[layer]["A"].addmm_(feature_rows.T, feature_rows)
                        accumulators[layer]["B"].addmm_(gradient_rows.T, gradient_rows)
                num_documents += int(batch["input_ids"].shape[0])
                num_tokens += valid_count
                del loss, outputs, logits, shift_logits, shift_labels
    finally:
        for handle in handles:
            handle.remove()
        for parameter, flag in requires_grad.items():
            parameter.requires_grad = flag
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache
        model.train(was_training)

    if num_tokens == 0:
        raise ValueError("KFAC collection saw no valid next-token positions")
    result: dict[int, KfacResult] = {}
    for layer, matrices in accumulators.items():
        A = (matrices["A"] / num_tokens).cpu()
        B = (matrices["B"] / num_tokens).cpu()
        result[layer] = KfacResult(
            A=A,
            B=B,
            num_documents=num_documents,
            num_tokens=num_tokens,
        )
    return result


def _validate_square_psd_factor(name: str, matrix: torch.Tensor) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square, got {tuple(matrix.shape)}")
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinity")


def low_curvature_projection(
    A: torch.Tensor,
    B: torch.Tensor,
    energy_threshold: float,
) -> dict[str, Any]:
    """Build CrispEdit's low-curvature mask from KFAC eigenpairs."""

    if not 0.0 < energy_threshold <= 1.0:
        raise ValueError("energy_threshold must be in (0, 1]")
    _validate_square_psd_factor("A", A)
    _validate_square_psd_factor("B", B)
    A = ((A.float() + A.float().T) * 0.5).contiguous()
    B = ((B.float() + B.float().T) * 0.5).contiguous()
    eigenvalues_a, eigenvectors_a = torch.linalg.eigh(A)
    eigenvalues_b, eigenvectors_b = torch.linalg.eigh(B)
    # Empirical second moments are PSD; tiny negative values are numerical
    # noise and would otherwise create nonsensical negative curvature products.
    eigenvalues_a = eigenvalues_a.clamp_min_(0)
    eigenvalues_b = eigenvalues_b.clamp_min_(0)
    products = torch.outer(eigenvalues_a, eigenvalues_b)
    total_energy = products.sum()
    if not torch.isfinite(total_energy) or float(total_energy) <= 0.0:
        mask = torch.ones_like(products, dtype=torch.bool)
        rank = 0
        threshold = math.inf
        retained_energy = 0.0
    else:
        sorted_products = torch.sort(products.flatten(), descending=True).values
        cumulative = torch.cumsum(sorted_products, dim=0) / total_energy
        index = int(torch.searchsorted(cumulative, energy_threshold).item())
        index = min(index, sorted_products.numel() - 1)
        threshold_tensor = sorted_products[index]
        mask = products < threshold_tensor
        rank = index + 1
        threshold = float(threshold_tensor)
        retained_energy = float(cumulative[index])
        del sorted_products, cumulative, threshold_tensor
    return {
        "Ua": eigenvectors_a,
        "Ub": eigenvectors_b,
        "mask": mask,
        "rank": rank,
        "total_directions": products.numel(),
        "curvature_threshold": threshold,
        "retained_high_curvature_energy": retained_energy,
        "allowed_low_curvature_directions": int(mask.sum()),
        # With eigenvalue ties, the strict upstream ``< threshold`` mask can
        # exclude more directions than the cumulative-energy search rank.
        "excluded_high_curvature_directions": int((~mask).sum()),
        "eigenvalue_a_min": float(eigenvalues_a.min()),
        "eigenvalue_a_max": float(eigenvalues_a.max()),
        "eigenvalue_b_min": float(eigenvalues_b.min()),
        "eigenvalue_b_max": float(eigenvalues_b.max()),
    }


def project_gradient(
    gradient: torch.Tensor,
    Ua: torch.Tensor,
    Ub: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Project a matrix gradient into the cached low-curvature subspace."""

    if gradient.ndim != 2:
        raise ValueError(f"gradient must be a matrix, got {tuple(gradient.shape)}")
    expected = (Ub.shape[0], Ua.shape[0])
    if tuple(gradient.shape) != expected:
        raise ValueError(f"gradient shape {tuple(gradient.shape)} != expected {expected}")
    if tuple(mask.shape) != (Ua.shape[0], Ub.shape[0]):
        raise ValueError(
            f"mask shape {tuple(mask.shape)} != expected {(Ua.shape[0], Ub.shape[0])}"
        )
    rotated = Ub.T @ gradient @ Ua
    rotated.mul_(mask.T.to(device=rotated.device, dtype=rotated.dtype))
    return Ub @ rotated @ Ua.T


class ProjectedAdam(torch.optim.Adam):
    """Adam with the upstream CrispEdit projection applied before each step."""

    def __init__(
        self,
        parameters: Sequence[torch.nn.Parameter],
        projections: Mapping[torch.nn.Parameter, Mapping[str, torch.Tensor]],
        *,
        lr: float,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            foreach=False,
            fused=False,
        )
        self.projections = dict(projections)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None or parameter.ndim != 2:
                    continue
                cache = self.projections.get(parameter)
                if cache is None:
                    raise KeyError("trainable matrix has no CrispEdit projection cache")
                parameter.grad.copy_(
                    project_gradient(
                        parameter.grad,
                        cache["Ua"],
                        cache["Ub"],
                        cache["mask"],
                    )
                )
        return super().step(closure)


def stats_filename(layer: int) -> str:
    return f"layer_{layer:03d}_down_proj_kfac.pt"


def projection_filename(layer: int, energy_threshold: float) -> str:
    threshold = f"{energy_threshold:.6f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"layer_{layer:03d}_down_proj_projection_e{threshold}.pt"


def validate_stats_payload(
    payload: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    layer: int,
) -> None:
    if payload.get("format") != STATS_FORMAT:
        raise ValueError(f"unsupported stats format: {payload.get('format')!r}")
    if int(payload.get("layer", -1)) != layer:
        raise ValueError(f"stats layer {payload.get('layer')} != requested layer {layer}")
    if payload.get("module_name") != module_name_for_layer(layer):
        raise ValueError("stats module name does not match the Qwen3 down projection")
    if not identities_compatible(payload.get("model_identity", {}), identity):
        raise ValueError("stats were produced from a different model identity")
    A = payload.get("A")
    B = payload.get("B")
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError("stats payload does not contain tensor A/B factors")
    _validate_square_psd_factor("A", A)
    _validate_square_psd_factor("B", B)


def validate_projection_payload(
    payload: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    layer: int,
    energy_threshold: float,
) -> None:
    if payload.get("format") != PROJECTION_FORMAT:
        raise ValueError(f"unsupported projection format: {payload.get('format')!r}")
    if int(payload.get("layer", -1)) != layer:
        raise ValueError(f"projection layer {payload.get('layer')} != requested {layer}")
    if payload.get("module_name") != module_name_for_layer(layer):
        raise ValueError("projection module name does not match the Qwen3 down projection")
    if not identities_compatible(payload.get("model_identity", {}), identity):
        raise ValueError("projection was produced from a different model identity")
    actual_threshold = float(payload.get("energy_threshold", -1))
    if not math.isclose(actual_threshold, energy_threshold, rel_tol=0, abs_tol=1e-12):
        raise ValueError(
            f"projection energy threshold {actual_threshold} != requested {energy_threshold}"
        )
    for key in ("Ua", "Ub", "mask"):
        if not isinstance(payload.get(key), torch.Tensor):
            raise TypeError(f"projection payload has no tensor {key}")
