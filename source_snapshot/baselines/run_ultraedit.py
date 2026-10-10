#!/usr/bin/env python3
"""Run UltraEdit on the shared Qwen3-8B-Base WikiBigEdit protocol.

This is a compatibility port of ``XiaojieGu/UltraEdit`` rather than a new
editing method.  It preserves the upstream key/value-gradient construction,
cumulative lifelong normalization, ridge solve, and additive parameter
updates while fixing the upstream WikiBigEdit tail-batch bug and routing all
evaluation through the common closed-book generation harness.

The bundled Qwen2.5 preset edits MLP gate/up projections in late layers.  The
default Qwen3 layer range (24--33) is therefore recorded as a transferred
preset, not as an officially tuned Qwen3 configuration.
"""

from __future__ import annotations

import argparse
import contextlib
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

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
    qa_prompt,
    sha256_file,
)
from baselines.run_lora_family import evaluate_and_write


UPSTREAM_REPOSITORY = "XiaojieGu/UltraEdit"
UPSTREAM_COMMIT = "4cbd796dc605dafb97777e83821a8670d3f68f17"


def parse_int_range(value: str) -> list[int]:
    """Parse comma-separated integers and inclusive ranges."""

    result: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start, stop = int(left), int(right)
            if stop < start:
                raise argparse.ArgumentTypeError(f"descending range: {part}")
            result.extend(range(start, stop + 1))
        else:
            result.append(int(part))
    if not result:
        raise argparse.ArgumentTypeError("empty layer selection")
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("duplicate layer selection")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("Qwen3-8B-Base"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--layers",
        type=parse_int_range,
        default=parse_int_range("24-33"),
        help="Qwen block indices, e.g. 24-33 or 24,28,32",
    )
    parser.add_argument(
        "--projections",
        default="gate_proj,up_proj",
        help="comma-separated MLP projection names",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--edit-batch-size", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=48)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--locality-panel-size", type=int, default=0)
    parser.add_argument("--multihop-panel-size", type=int, default=0)
    parser.add_argument("--solver", choices=("auto", "primal", "dual"), default="auto")
    parser.add_argument("--normalizer-epsilon", type=float, default=1e-7)
    parser.add_argument("--max-stages", type=int, default=0)
    parser.add_argument("--examples-per-stage", type=int, default=0)
    parser.add_argument("--no-stage-eval", action="store_true")
    parser.add_argument("--skip-base-eval", action="store_true")
    parser.add_argument(
        "--save-delta",
        type=Path,
        default=None,
        help="optional path for the final edited-module delta checkpoint",
    )
    parser.add_argument("--allow-existing-output", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_module(root: torch.nn.Module, name: str) -> torch.nn.Module:
    module = root
    for part in name.split("."):
        module = getattr(module, part)
    return module


def target_module_names(layers: Sequence[int], projections: Sequence[str]) -> list[str]:
    return [
        f"model.layers.{layer}.mlp.{projection}"
        for projection in projections
        for layer in layers
    ]


@dataclass
class RunningMoments:
    """Mergeable sample moments matching UltraEdit's lifelong normalizer."""

    count: int = 0
    mean: torch.Tensor | None = None
    m2: torch.Tensor | None = None

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().to(device="cpu", dtype=torch.float32)
        if values.ndim != 2 or values.shape[0] == 0:
            return
        batch_count = int(values.shape[0])
        batch_mean = values.mean(dim=0)
        centered = values - batch_mean
        batch_m2 = (centered * centered).sum(dim=0)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        assert self.mean is not None and self.m2 is not None
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / total)
        self.m2 = self.m2 + batch_m2 + delta.square() * (self.count * batch_count / total)
        self.count = total

    def std(self, epsilon: float) -> torch.Tensor:
        if self.mean is None or self.m2 is None or self.count == 0:
            raise RuntimeError("normalizer has no observations")
        denominator = max(self.count - 1, 1)
        return (self.m2 / denominator).clamp_min(0).sqrt().clamp_min(epsilon)


def ultraedit_tokenize(
    tokenizer: Any,
    question: str,
    answer: str,
    max_length: int,
) -> dict[str, torch.Tensor]:
    """Create the upstream unshifted answer-label layout.

    Inputs end at the penultimate answer token and labels begin on the last
    prompt token, so ``logits[labels != -100]`` predict every answer token.
    """

    prompt_ids = tokenizer(qa_prompt(question), add_special_tokens=True).input_ids
    eos = tokenizer.eos_token or ""
    answer_ids = tokenizer(
        " " + answer.strip() + eos,
        add_special_tokens=False,
    ).input_ids
    if not answer_ids:
        raise ValueError("answer tokenization is empty")
    available_prompt = max_length - len(answer_ids) + 1
    if available_prompt < 1:
        answer_ids = answer_ids[:max_length]
        available_prompt = 1
    prompt_ids = prompt_ids[:available_prompt]
    input_ids = prompt_ids + answer_ids[:-1]
    labels = [-100] * max(len(prompt_ids) - 1, 0) + answer_ids
    if len(input_ids) != len(labels):
        raise AssertionError("UltraEdit input/label alignment failed")
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def collate_examples(
    examples: Sequence[dict[str, torch.Tensor]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    width = max(int(example["input_ids"].numel()) for example in examples)
    input_ids = torch.full((len(examples), width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), width), dtype=torch.long)
    labels = torch.full((len(examples), width), -100, dtype=torch.long)
    for index, example in enumerate(examples):
        length = int(example["input_ids"].numel())
        input_ids[index, :length] = example["input_ids"]
        attention_mask[index, :length] = 1
        labels[index, :length] = example["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class ActivationGradientCapture:
    def __init__(self, model: torch.nn.Module, module_names: Sequence[str]) -> None:
        self.module_names = list(module_names)
        self.answer_mask: torch.Tensor | None = None
        self.keys: dict[str, torch.Tensor] = {}
        self.value_grads: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []
        for name in self.module_names:
            module = get_module(model, name)
            if not isinstance(module, torch.nn.Linear):
                raise TypeError(f"UltraEdit target is not Linear: {name} ({type(module)})")
            self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...], output: torch.Tensor):
            if self.answer_mask is None:
                raise RuntimeError("answer mask was not set before forward")
            mask = self.answer_mask
            self.keys[name] = inputs[0][mask].detach()

            def save_grad(gradient: torch.Tensor) -> None:
                self.value_grads[name] = gradient[mask].detach()

            output.register_hook(save_grad)

        return hook

    def begin(self, answer_mask: torch.Tensor) -> None:
        self.answer_mask = answer_mask
        self.keys.clear()
        self.value_grads.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def solve_ridge_shift(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    ridge: float,
    solver: str,
) -> tuple[torch.Tensor, str]:
    """Compute ``(K^T K + ridge I)^-1 K^T V`` in primal or dual form."""

    if keys.ndim != 2 or values.ndim != 2 or keys.shape[0] != values.shape[0]:
        raise ValueError("keys and values must be rank-2 tensors with equal row count")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    selected = solver
    if selected == "auto":
        selected = "dual" if keys.shape[0] < keys.shape[1] else "primal"
    if selected == "dual":
        gram = keys @ keys.T
        gram.diagonal().add_(ridge)
        solved = torch.linalg.solve(gram, values)
        return keys.T @ solved, selected
    if selected == "primal":
        gram = keys.T @ keys
        gram.diagonal().add_(ridge)
        rhs = keys.T @ values
        return torch.linalg.solve(gram, rhs), selected
    raise ValueError(f"unknown solver: {solver}")


def cache_stage_features(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    module_names: Sequence[str],
    normalizer: RunningMoments,
    device: torch.device,
    *,
    batch_size: int,
    max_length: int,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
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
            missing = [name for name in module_names if name not in capture.value_grads]
            if missing:
                raise RuntimeError(f"missing captured gradients: {missing[:3]}")
            for name in module_names:
                keys = capture.keys[name].float().cpu()
                grads = capture.value_grads[name].float().cpu()
                captured_keys[name].append(keys)
                captured_grads[name].append(grads)
                normalizer.update(torch.cat((keys, grads), dim=-1))
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
        "cache_batches": (len(examples) + batch_size - 1) // batch_size,
        "answer_tokens": token_count,
        "mean_cache_loss": loss_total / max((len(examples) + batch_size - 1) // batch_size, 1),
        "cache_wall_seconds": time.perf_counter() - started,
    }


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
) -> list[dict[str, Any]]:
    if normalizer.mean is None:
        raise RuntimeError("normalizer was not updated")
    mean = normalizer.mean.to(device)
    std = normalizer.std(epsilon).to(device)
    module_log: list[dict[str, Any]] = []
    for name, (keys_cpu, grads_cpu) in features.items():
        module = get_module(model, name)
        key_size = keys_cpu.shape[1]
        keys = keys_cpu.to(device)
        grads = grads_cpu.to(device)
        normalized_keys = (keys - mean[:key_size]) / std[:key_size]
        normalized_grads = (grads - mean[key_size:]) / std[key_size:]
        coefficients = -learning_rate * normalized_keys.square().sum(dim=-1, keepdim=True)
        pseudo_values = coefficients * normalized_grads
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
                "solver": selected_solver,
                "shift_frobenius_norm": float(shift.norm()),
                "applied_nonzero": int(torch.count_nonzero(applied)),
            }
        )
        del keys, grads, normalized_keys, normalized_grads, coefficients, pseudo_values, shift, applied
        torch.cuda.empty_cache()
    return module_log


def restrict_stages(
    stages: dict[int, list[dict[str, Any]]],
    *,
    max_stages: int,
    examples_per_stage: int,
) -> dict[int, list[dict[str, Any]]]:
    selected: dict[int, list[dict[str, Any]]] = {}
    for stage, rows in stages.items():
        if max_stages > 0 and len(selected) >= max_stages:
            break
        selected[stage] = rows[:examples_per_stage] if examples_per_stage > 0 else rows
    return selected


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
    projections = [item.strip() for item in args.projections.split(",") if item.strip()]
    module_names = target_module_names(args.layers, projections)

    locality_panel = manifest_panel(metadata, "locality", limit=args.locality_panel_size) or fixed_panel(
        rows, "loc", "loc_ans", args.locality_panel_size, args.seed + 1001
    )
    multihop_panel = manifest_panel(metadata, "mhop", limit=args.multihop_panel_size) or fixed_panel(
        rows, "mhop", "mhop_ans", args.multihop_panel_size, args.seed + 2001
    )
    tokenizer = load_tokenizer(args.model_dir)
    model = load_model(args.model_dir, device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        embedding = model.get_input_embeddings()
        embedding.register_forward_hook(lambda _m, _i, output: output.requires_grad_(True))
    for name in module_names:
        get_module(model, name)

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
        "method": "UltraEdit",
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "qwen3_preset_status": "Qwen2.5 relative-depth transfer; not officially tuned",
        "edit_modules": module_names,
        "num_facts": len(selected_rows),
        "stage_counts": {str(stage): len(stage_rows) for stage, stage_rows in stages.items()},
        "closed_book_evaluation": True,
        "prompt_template": "Question: <question>\\nAnswer:",
        "upstream_label_alignment": True,
        "in_memory_cache": True,
        "known_upstream_config_corrections": [
            "WikiBigEdit dataset config defines num_seq=170 although the public "
            "Qwen2.5 example command uses num_seq=17; this runner derives all "
            "stage counts from the frozen manifest"
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

    normalizer = RunningMoments()
    training_log: list[dict[str, Any]] = []
    seen: list[dict[str, Any]] = []
    started = time.time()
    for stage, current_rows in stages.items():
        torch.cuda.reset_peak_memory_stats(device)
        features, cache_log = cache_stage_features(
            model,
            tokenizer,
            current_rows,
            module_names,
            normalizer,
            device,
            batch_size=args.edit_batch_size,
            max_length=args.max_length,
        )
        solve_started = time.perf_counter()
        module_log = apply_ultraedit_stage(
            model,
            features,
            normalizer,
            device,
            learning_rate=args.learning_rate,
            ridge=args.ridge,
            solver=args.solver,
            epsilon=args.normalizer_epsilon,
        )
        del features
        torch.cuda.empty_cache()
        seen.extend(current_rows)
        stage_log = {
            "stage": stage,
            "current_count": len(current_rows),
            **cache_log,
            "solve_wall_seconds": time.perf_counter() - solve_started,
            "normalizer_count": normalizer.count,
            "peak_vram_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
            "modules": module_log,
        }
        training_log.append(stage_log)
        atomic_json(args.output_dir / "training_log.json", training_log)
        append_jsonl(args.output_dir / "module_trace.jsonl", module_log)

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
                "format": "ultraedit_delta/v1",
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
            "method": "UltraEdit",
            "elapsed_seconds": time.time() - started,
            "normalizer_count": normalizer.count,
        },
    )


if __name__ == "__main__":
    main()
