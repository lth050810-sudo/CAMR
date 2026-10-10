#!/usr/bin/env python3
"""Run a minimal, auditable StableEdit port on Qwen3-8B-Base.

The upstream implementation in ``MINE-USTC/StableEdit`` cannot currently be
used as an entry point without patching unrelated syntax/import/configuration
errors.  This runner ports the StableEdit update itself and connects it to the
shared WikiBigEdit manifest and evaluator used by the other local baselines.

The port preserves the important upstream mechanics:

* answer-token input activations (keys) and output gradients (values),
* running feature moments shared by modules with the same linear shape,
* value-gradient whitening after enough lifelong observations,
* the StableEdit norm-scaled pseudo-value update, and
* sequential additive updates to MLP ``down_proj`` weights.

For efficiency, the ridge system is solved in the mathematically equivalent
dual form whenever the number of cached answer tokens is smaller than the
input dimension.  The default Qwen3 layers are a relative-depth transfer of
the official Qwen2.5 WikiBigEdit preset; they are not tuned Qwen3 hyperparams.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from baselines.common import (
    append_jsonl,
    atomic_json,
    environment_metadata,
    fixed_panel,
    group_by_stage,
    load_manifest,
    load_model,
    load_tokenizer,
    manifest_panel,
    sha256_file,
)
from baselines.run_lora_family import evaluate_and_write
from baselines.run_ultraedit import (
    ActivationGradientCapture,
    collate_examples,
    get_module,
    parse_int_range,
    restrict_stages,
    solve_ridge_shift,
    ultraedit_tokenize,
)


UPSTREAM_REPOSITORY = "MINE-USTC/StableEdit"
UPSTREAM_COMMIT = "be5769232cf5ba160d8f011dfb9a6d0b5a2a6470"
QWEN25_WIKIBIGEDIT_LAYERS = [*range(8, 15), *range(18, 25)]


def transfer_layers_by_relative_depth(
    layers: Sequence[int],
    *,
    source_depth: int,
    target_depth: int,
) -> list[int]:
    """Map zero-based block centers between model depths, preserving count."""

    if source_depth <= 0 or target_depth <= 0:
        raise ValueError("model depths must be positive")
    mapped = [
        round((int(layer) + 0.5) * target_depth / source_depth - 0.5)
        for layer in layers
    ]
    if any(layer < 0 or layer >= target_depth for layer in mapped):
        raise ValueError(f"mapped layer outside target depth: {mapped}")
    if len(mapped) != len(set(mapped)):
        raise ValueError(f"relative-depth mapping produced duplicates: {mapped}")
    return mapped


QWEN3_TRANSFER_LAYERS = transfer_layers_by_relative_depth(
    QWEN25_WIKIBIGEDIT_LAYERS,
    source_depth=28,
    target_depth=36,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("Qwen3-8B-Base"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--layers",
        type=parse_int_range,
        default=QWEN3_TRANSFER_LAYERS,
        help="Qwen block indices; defaults to the relative-depth Qwen2.5 transfer",
    )
    parser.add_argument("--projection", choices=("down_proj",), default="down_proj")
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument(
        "--edits-per-step",
        type=int,
        default=100,
        help="number of facts jointly converted into one additive update",
    )
    parser.add_argument(
        "--cache-batch-size",
        type=int,
        default=10,
        help="micro-batch used to collect answer-token activations and gradients",
    )
    parser.add_argument("--eval-batch-size", type=int, default=48)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--locality-panel-size", type=int, default=0)
    parser.add_argument("--multihop-panel-size", type=int, default=0)
    parser.add_argument("--solver", choices=("auto", "primal", "dual"), default="auto")
    parser.add_argument("--normalizer-epsilon", type=float, default=1e-7)
    parser.add_argument("--whitening-eigenvalue-floor", type=float, default=1e-6)
    parser.add_argument(
        "--disable-value-whitening",
        action="store_true",
        help="diagnostic only: always use diagonal standardization",
    )
    parser.add_argument("--max-stages", type=int, default=0)
    parser.add_argument("--examples-per-stage", type=int, default=0)
    parser.add_argument("--no-stage-eval", action="store_true")
    parser.add_argument("--skip-base-eval", action="store_true")
    parser.add_argument(
        "--save-delta",
        type=Path,
        default=None,
        help="optional path for final edited-module deltas",
    )
    parser.add_argument("--allow-existing-output", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def target_module_names(layers: Sequence[int], projection: str) -> list[str]:
    return [f"model.layers.{layer}.mlp.{projection}" for layer in layers]


@dataclass
class StableNormalization:
    mean: torch.Tensor
    std: torch.Tensor
    value_inverse_sqrt: torch.Tensor | None
    used_value_whitening: bool


class StableRunningStats:
    """Online moments matching StableEdit's ``RunningMeanStd('stable')``.

    The value covariance is retained because StableEdit switches from diagonal
    scaling to full whitening once ``count > value_dim + 1``.  Tensors stay on
    the editor device, matching upstream behavior and avoiding CPU covariance
    updates for Qwen's 4096-dimensional down-projection outputs.
    """

    def __init__(self, key_dim: int, value_dim: int, device: torch.device) -> None:
        if key_dim <= 0 or value_dim <= 0:
            raise ValueError("key/value dimensions must be positive")
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.device = device
        self.count = 0
        self.mean = torch.zeros(key_dim + value_dim, device=device, dtype=torch.float32)
        self.m2 = torch.zeros_like(self.mean)
        self.value_scatter = torch.zeros(
            (value_dim, value_dim), device=device, dtype=torch.float32
        )

    def update(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        keys = keys.detach().to(device=self.device, dtype=torch.float32)
        values = values.detach().to(device=self.device, dtype=torch.float32)
        if keys.ndim != 2 or values.ndim != 2 or keys.shape[0] != values.shape[0]:
            raise ValueError("keys and values must be rank-2 with equal row count")
        if keys.shape[1] != self.key_dim or values.shape[1] != self.value_dim:
            raise ValueError(
                f"feature shape mismatch: got {tuple(keys.shape)}, {tuple(values.shape)}"
            )
        batch_count = int(keys.shape[0])
        if batch_count == 0:
            return
        joined = torch.cat((keys, values), dim=-1)
        batch_mean = joined.mean(dim=0)
        centered = joined - batch_mean
        batch_m2 = centered.square().sum(dim=0)
        value_centered = values - values.mean(dim=0)
        batch_value_scatter = value_centered.T @ value_centered

        if self.count == 0:
            self.count = batch_count
            self.mean.copy_(batch_mean)
            self.m2.copy_(batch_m2)
            self.value_scatter.copy_(batch_value_scatter)
            return

        old_count = self.count
        total = old_count + batch_count
        delta = batch_mean - self.mean
        correction = old_count * batch_count / total
        self.mean.add_(delta, alpha=batch_count / total)
        self.m2.add_(batch_m2 + delta.square() * correction)
        value_delta = delta[self.key_dim :]
        self.value_scatter.add_(
            batch_value_scatter + torch.outer(value_delta, value_delta) * correction
        )
        self.count = total

    def normalization(
        self,
        *,
        epsilon: float,
        eigenvalue_floor: float,
        enable_value_whitening: bool,
        inference: bool = True,
    ) -> StableNormalization:
        if self.count == 0:
            raise RuntimeError("normalizer has no observations")
        if epsilon <= 0 or eigenvalue_floor <= 0:
            raise ValueError("normalizer floors must be positive")
        denominator = self.count - 1 + epsilon
        std = (self.m2 / denominator).clamp_min(0).sqrt()
        use_whitening = (
            inference
            and enable_value_whitening
            and self.count > self.value_dim + 1
        )
        inverse_sqrt = None
        if use_whitening:
            # This unusual denominator is preserved from the official code.
            covariance = self.value_scatter / (self.count - self.value_dim - 1 + epsilon)
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            inverse_sqrt = (
                eigenvectors
                @ torch.diag(eigenvalues.clamp_min(eigenvalue_floor).rsqrt())
                @ eigenvectors.T
            )
        return StableNormalization(
            mean=self.mean,
            std=std,
            value_inverse_sqrt=inverse_sqrt,
            used_value_whitening=use_whitening,
        )

    def transform(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        normalization: StableNormalization,
        *,
        epsilon: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys = keys.to(device=self.device, dtype=torch.float32)
        values = values.to(device=self.device, dtype=torch.float32)
        key_mean = normalization.mean[: self.key_dim]
        value_mean = normalization.mean[self.key_dim :]
        normalized_keys = (keys - key_mean) / (
            normalization.std[: self.key_dim] + epsilon
        )
        centered_values = values - value_mean
        if normalization.value_inverse_sqrt is not None:
            normalized_values = centered_values @ normalization.value_inverse_sqrt
        else:
            normalized_values = centered_values / (
                normalization.std[self.key_dim :] + epsilon
            )
        return normalized_keys, normalized_values


def cache_edit_features(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    module_names: Sequence[str],
    normalizers: dict[tuple[int, int], StableRunningStats],
    device: torch.device,
    *,
    batch_size: int,
    max_length: int,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    """Collect one edit step and update shared-by-shape lifelong moments."""

    examples = [
        ultraedit_tokenize(tokenizer, str(row["update"]), str(row["ans"]), max_length)
        for row in rows
    ]
    examples.sort(key=lambda item: int(item["input_ids"].numel()), reverse=True)
    captured_keys: dict[str, list[torch.Tensor]] = {name: [] for name in module_names}
    captured_grads: dict[str, list[torch.Tensor]] = {name: [] for name in module_names}
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    capture = ActivationGradientCapture(model, module_names)
    loss_total = 0.0
    token_count = 0
    cache_batches = 0
    started = time.perf_counter()
    model.eval()
    model.config.use_cache = False
    try:
        for start in range(0, len(examples), batch_size):
            batch = collate_examples(examples[start : start + batch_size], int(pad_id))
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            answer_mask = batch["labels"].ne(-100)
            capture.begin(answer_mask)
            model.zero_grad(set_to_none=True)
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits
            loss = F.cross_entropy(logits[answer_mask].float(), batch["labels"][answer_mask])
            loss.backward()
            loss_total += float(loss.detach())
            token_count += int(answer_mask.sum())
            cache_batches += 1
            missing = [name for name in module_names if name not in capture.value_grads]
            if missing:
                raise RuntimeError(f"missing captured gradients: {missing[:3]}")
            for name in module_names:
                keys = capture.keys[name].float()
                grads = capture.value_grads[name].float()
                shape = (int(keys.shape[1]), int(grads.shape[1]))
                if shape not in normalizers:
                    normalizers[shape] = StableRunningStats(*shape, device=device)
                normalizers[shape].update(keys, grads)
                captured_keys[name].append(keys.cpu())
                captured_grads[name].append(grads.cpu())
            del logits, loss, batch
    finally:
        capture.close()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    features = {
        name: (torch.cat(captured_keys[name], dim=0), torch.cat(captured_grads[name], dim=0))
        for name in module_names
    }
    return features, {
        "cache_batches": cache_batches,
        "answer_tokens": token_count,
        "mean_cache_loss": loss_total / max(cache_batches, 1),
        "cache_wall_seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def apply_stableedit_step(
    model: torch.nn.Module,
    features: dict[str, tuple[torch.Tensor, torch.Tensor]],
    normalizers: dict[tuple[int, int], StableRunningStats],
    device: torch.device,
    *,
    learning_rate: float,
    ridge: float,
    solver: str,
    epsilon: float,
    eigenvalue_floor: float,
    enable_value_whitening: bool,
) -> list[dict[str, Any]]:
    """Compute and apply one StableEdit update for every target module."""

    normalization_cache: dict[tuple[int, int], StableNormalization] = {}
    module_log: list[dict[str, Any]] = []
    for name, (keys_cpu, grads_cpu) in features.items():
        module = get_module(model, name)
        shape = (int(keys_cpu.shape[1]), int(grads_cpu.shape[1]))
        normalizer = normalizers[shape]
        if shape not in normalization_cache:
            normalization_cache[shape] = normalizer.normalization(
                epsilon=epsilon,
                eigenvalue_floor=eigenvalue_floor,
                enable_value_whitening=enable_value_whitening,
                inference=True,
            )
        normalization = normalization_cache[shape]
        keys = keys_cpu.to(device=device, dtype=torch.float32)
        grads = grads_cpu.to(device=device, dtype=torch.float32)
        normalized_keys, pseudo_gradients = normalizer.transform(
            keys,
            grads,
            normalization,
            epsilon=epsilon,
        )
        coefficients = -learning_rate * normalized_keys.square().sum(dim=-1, keepdim=True)
        pseudo_values = coefficients * pseudo_gradients
        shift, selected_solver = solve_ridge_shift(
            keys,
            pseudo_values,
            ridge=ridge,
            solver=solver,
        )
        applied = shift.T.to(module.weight.dtype)
        module.weight.add_(applied)
        module_log.append(
            {
                "module": name,
                "tokens": int(keys.shape[0]),
                "normalizer_count": normalizer.count,
                "value_whitening": normalization.used_value_whitening,
                "solver": selected_solver,
                "shift_frobenius_norm": float(shift.norm()),
                "max_abs_shift": float(shift.abs().max()),
            }
        )
        del keys, grads, normalized_keys, pseudo_gradients, coefficients, pseudo_values
        del shift, applied
        torch.cuda.empty_cache()
    return module_log


def chunks(rows: Sequence[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [list(rows[start : start + size]) for start in range(0, len(rows), size)]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.allow_existing_output:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    rows, metadata = load_manifest(args.manifest)
    stages = restrict_stages(
        group_by_stage(rows),
        max_stages=args.max_stages,
        examples_per_stage=args.examples_per_stage,
    )
    selected_rows = [row for stage_rows in stages.values() for row in stage_rows]
    if not selected_rows:
        raise ValueError("no selected WikiBigEdit rows")
    module_names = target_module_names(args.layers, args.projection)

    locality_panel = manifest_panel(metadata, "locality", limit=args.locality_panel_size) or fixed_panel(
        rows, "loc", "loc_ans", args.locality_panel_size, args.seed + 1001
    )
    multihop_panel = manifest_panel(metadata, "mhop", limit=args.multihop_panel_size) or fixed_panel(
        rows, "mhop", "mhop_ans", args.multihop_panel_size, args.seed + 2001
    )
    tokenizer = load_tokenizer(args.model_dir)
    model = load_model(args.model_dir, device)
    model_depth = int(model.config.num_hidden_layers)
    invalid_layers = [layer for layer in args.layers if layer < 0 or layer >= model_depth]
    if invalid_layers:
        raise ValueError(f"layers outside model depth {model_depth}: {invalid_layers}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        embedding = model.get_input_embeddings()
        embedding.register_forward_hook(lambda _m, _i, output: output.requires_grad_(True))
    for name in module_names:
        module = get_module(model, name)
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"StableEdit target is not Linear: {name} ({type(module)})")

    original_weights: dict[str, torch.Tensor] | None = None
    if args.save_delta is not None:
        original_weights = {
            name: get_module(model, name).weight.detach().cpu().clone()
            for name in module_names
        }

    config = {
        **vars(args),
        "layers": list(args.layers),
        "model_dir": str(args.model_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "save_delta": str(args.save_delta.resolve()) if args.save_delta else None,
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_protocol_id": metadata.get("protocol_id"),
        "method": "StableEdit",
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_qwen25_wikibigedit_layers": QWEN25_WIKIBIGEDIT_LAYERS,
        "qwen3_transfer_layers": QWEN3_TRANSFER_LAYERS,
        "qwen3_preset_status": (
            "Qwen2.5 WikiBigEdit relative-depth transfer; not officially tuned for Qwen3"
        ),
        "edit_modules": module_names,
        "num_facts": len(selected_rows),
        "stage_counts": {str(stage): len(stage_rows) for stage, stage_rows in stages.items()},
        "closed_book_evaluation": True,
        "prompt_template": "Question: <question>\\nAnswer:",
        "upstream_label_alignment": True,
        "shared_normalizer_by_linear_shape": True,
        "ridge_solver_note": "dual is algebraically equivalent to upstream primal solve",
        "preheat": (
            "disabled in this compatibility smoke; official runs preheat on a separate train split"
        ),
        "known_upstream_blockers_bypassed": [
            "editor/base.py syntax error in WikiBigEdit train_max_steps branch",
            "missing hessian_dominant_subspace import in model.py",
            "missing config.for_lora default",
            "RunningMeanStd_mode=stale typo in Qwen WikiBigEdit run.sh",
        ],
        **environment_metadata(),
    }
    atomic_json(args.output_dir / "run_config.json", config)

    metrics: list[dict[str, Any]] = []
    if not args.skip_base_eval:
        metrics.append(
            evaluate_and_write(
                model,
                tokenizer,
                selected_rows,
                locality_panel,
                multihop_panel,
                device,
                args,
                checkpoint="base",
                current_stage=None,
            )
        )
        atomic_json(args.output_dir / "metrics.json", metrics)

    normalizers: dict[tuple[int, int], StableRunningStats] = {}
    training_log: list[dict[str, Any]] = []
    seen: list[dict[str, Any]] = []
    started = time.time()
    global_step = 0
    for stage, current_rows in stages.items():
        stage_started = time.perf_counter()
        stage_steps: list[dict[str, Any]] = []
        torch.cuda.reset_peak_memory_stats(device)
        for edit_rows in chunks(current_rows, args.edits_per_step):
            global_step += 1
            features, cache_log = cache_edit_features(
                model,
                tokenizer,
                edit_rows,
                module_names,
                normalizers,
                device,
                batch_size=args.cache_batch_size,
                max_length=args.max_length,
            )
            solve_started = time.perf_counter()
            module_log = apply_stableedit_step(
                model,
                features,
                normalizers,
                device,
                learning_rate=args.learning_rate,
                ridge=args.ridge,
                solver=args.solver,
                epsilon=args.normalizer_epsilon,
                eigenvalue_floor=args.whitening_eigenvalue_floor,
                enable_value_whitening=not args.disable_value_whitening,
            )
            del features
            torch.cuda.empty_cache()
            step_log = {
                "global_step": global_step,
                "stage": stage,
                "edit_count": len(edit_rows),
                "sample_ids": [row["sample_id"] for row in edit_rows],
                **cache_log,
                "solve_wall_seconds": time.perf_counter() - solve_started,
                "modules": module_log,
            }
            stage_steps.append(step_log)
            append_jsonl(
                args.output_dir / "module_trace.jsonl",
                [{"global_step": global_step, "stage": stage, **row} for row in module_log],
            )
        seen.extend(current_rows)
        stage_log = {
            "stage": stage,
            "current_count": len(current_rows),
            "edit_steps": stage_steps,
            "stage_wall_seconds": time.perf_counter() - stage_started,
            "peak_vram_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
            "normalizers": {
                f"{shape[0]}x{shape[1]}": stats.count
                for shape, stats in normalizers.items()
            },
        }
        training_log.append(stage_log)
        atomic_json(args.output_dir / "training_log.json", training_log)

        if not args.no_stage_eval or stage == max(stages):
            metrics.append(
                evaluate_and_write(
                    model,
                    tokenizer,
                    seen,
                    locality_panel,
                    multihop_panel,
                    device,
                    args,
                    checkpoint=f"tau{stage}",
                    current_stage=stage,
                )
            )
            atomic_json(args.output_dir / "metrics.json", metrics)

    if args.save_delta is not None:
        assert original_weights is not None
        args.save_delta.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "stableedit_delta/v1",
                "base_model": str(args.model_dir.resolve()),
                "edit_modules": module_names,
                "delta": {
                    name: get_module(model, name).weight.detach().cpu() - original_weights[name]
                    for name in module_names
                },
            },
            args.save_delta,
        )

    atomic_json(args.output_dir / "training_log.json", training_log)
    atomic_json(args.output_dir / "metrics.json", metrics)
    atomic_json(
        args.output_dir / "completed.json",
        {
            "status": "complete",
            "method": "StableEdit",
            "elapsed_seconds": time.time() - started,
            "edit_steps": global_step,
            "normalizer_counts": {
                f"{shape[0]}x{shape[1]}": stats.count
                for shape, stats in normalizers.items()
            },
        },
    )


if __name__ == "__main__":
    main()
