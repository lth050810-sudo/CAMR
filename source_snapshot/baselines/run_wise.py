#!/usr/bin/env python3
"""Run a Qwen3-compatible WISE-Retrieve baseline on WikiBigEdit.

WISE is a side-parametric-memory method, so its results should be reported
separately from methods that permanently internalize every edit in the base
backbone.  This port follows the WISE implementation bundled with the
WikiBigEdit repository: one ``down_proj`` side weight, random parameter
sharding, periodic TIES consolidation, and activation-distance routing.

The upstream WikiBigEdit Llama-3 recipe uses layer 29, 10 iterations per edit,
and 500/1000 edit save/merge intervals.  Qwen3 layer 30 is a relative-depth
transfer from the current EasyEdit Qwen2.5 layer-23 preset and is explicitly
recorded as untuned.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from baselines.common import (
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
from baselines.run_ultraedit import get_module, restrict_stages


UPSTREAM_REPOSITORY = "ExplainableML/WikiBigEdit (adapted EasyEdit WISE)"
UPSTREAM_COMMIT = "66b428560254c38755ab70068cb0b0179aa442e1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("Qwen3-8B-Base"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--edit-lr", type=float, default=0.9)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--mask-ratio", type=float, default=0.2)
    parser.add_argument("--norm-constraint", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=20.0)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--activation-ratio", type=float, default=0.5)
    parser.add_argument("--save-frequency", type=int, default=500)
    parser.add_argument("--merge-frequency", type=int, default=1000)
    parser.add_argument("--ties-density", type=float, default=0.53)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--locality-panel-size", type=int, default=0)
    parser.add_argument("--multihop-panel-size", type=int, default=0)
    parser.add_argument("--max-stages", type=int, default=0)
    parser.add_argument("--examples-per-stage", type=int, default=0)
    parser.add_argument("--no-stage-eval", action="store_true")
    parser.add_argument("--skip-base-eval", action="store_true")
    parser.add_argument("--save-adapter", type=Path, default=None)
    parser.add_argument("--allow-existing-output", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_subsequence(sequence: Sequence[int], pattern: Sequence[int]) -> int | None:
    if not pattern:
        return None
    for start in range(len(sequence) - len(pattern) + 1):
        if list(sequence[start : start + len(pattern)]) == list(pattern):
            return start
    return None


@dataclass
class ActivationMeter:
    count: int = 0
    total: float = 0.0
    minimum: float = math.inf

    def update(self, value: float) -> None:
        self.count += 1
        self.total += float(value)
        self.minimum = min(self.minimum, float(value))

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else math.inf


def activation_distance(
    original: torch.Tensor,
    candidate: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(F.silu(candidate) - F.silu(original), dim=-1)
    if mask is None:
        if distance.shape[-1] > 100:
            return distance.topk(k=1, dim=-1).values.squeeze(-1)
        return distance.mean(dim=-1)
    numeric_mask = mask.to(device=distance.device, dtype=distance.dtype)
    denominator = numeric_mask.sum(dim=-1).clamp_min(1)
    return (distance * numeric_mask).sum(dim=-1) / denominator


def magnitude_sparsify(delta: torch.Tensor, density: float) -> torch.Tensor:
    if not 0 < density <= 1:
        raise ValueError("density must lie in (0, 1]")
    if density == 1:
        return delta
    flattened = delta.abs().reshape(-1)
    keep = max(1, int(density * flattened.numel()))
    threshold = torch.topk(flattened, k=keep, largest=True, sorted=False).values.min()
    return delta * delta.abs().ge(threshold)


def ties_merge(
    base: torch.Tensor,
    tensors: Sequence[torch.Tensor],
    *,
    density: float,
    total_weight: float = 1.0,
) -> torch.Tensor:
    """Match WISE's magnitude + sign-consensus + normalized GTA merge."""

    if not tensors:
        return base.clone()
    deltas = torch.stack(
        [magnitude_sparsify(tensor - base, density) for tensor in tensors],
        dim=0,
    )
    weights = torch.full(
        (len(tensors),) + (1,) * base.ndim,
        total_weight / len(tensors),
        dtype=deltas.dtype,
        device=deltas.device,
    )
    weighted = deltas * weights
    majority = torch.where(
        weighted.sum(dim=0) >= 0,
        torch.ones((), device=base.device, dtype=base.dtype),
        -torch.ones((), device=base.device, dtype=base.dtype),
    )
    consensus = deltas.sign().eq(majority)
    numerator = (weighted * consensus).sum(dim=0)
    denominator = (weights * consensus).sum(dim=0)
    denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
    return (base + numerator / denominator).to(base.dtype)


class WISEAdapter(torch.nn.Module):
    def __init__(
        self,
        layer: torch.nn.Linear,
        *,
        activation_ratio: float,
    ) -> None:
        super().__init__()
        if not isinstance(layer, torch.nn.Linear):
            raise TypeError(f"WISE expects nn.Linear, got {type(layer)}")
        self.base_layer = layer
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False
        self.original_layer = copy.deepcopy(layer)
        for parameter in self.original_layer.parameters():
            parameter.requires_grad = False
        self.new_weight = torch.nn.Parameter(layer.weight.detach().clone(), requires_grad=True)
        self.activation_ratio = float(activation_ratio)
        self.editing = False
        self.original_output: torch.Tensor | None = None
        self.new_output: torch.Tensor | None = None
        self.current_meter = ActivationMeter()
        self.memory_weights: list[torch.Tensor] = []
        self.memory_meters: list[ActivationMeter] = []
        self.edit_count = 0
        self.weight_mask: torch.Tensor | None = None

    def _linear(self, inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, weight, self.base_layer.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original = self.original_layer(inputs)
        candidate = self._linear(inputs, self.new_weight)
        if self.editing:
            self.original_output = original
            self.new_output = candidate
            return candidate

        current_distance = activation_distance(original, candidate)
        current_threshold = self.current_meter.minimum * self.activation_ratio
        choose_current = current_distance >= current_threshold
        selected = torch.where(choose_current[..., None, None], candidate, original)
        best_distance = torch.where(
            choose_current,
            current_distance,
            torch.full_like(current_distance, -torch.inf),
        )
        for memory_weight, meter in zip(self.memory_weights, self.memory_meters):
            memory_output = self._linear(inputs, memory_weight)
            distance = activation_distance(original, memory_output)
            choose = (distance > best_distance) & (
                distance >= meter.minimum * self.activation_ratio
            )
            selected = torch.where(choose[..., None, None], memory_output, selected)
            best_distance = torch.where(choose, distance, best_distance)
        return selected

    @torch.no_grad()
    def reset_new_weight(self) -> None:
        self.new_weight = torch.nn.Parameter(
            self.original_layer.weight.detach().clone(),
            requires_grad=True,
        )
        self.current_meter = ActivationMeter()


def replace_module(model: torch.nn.Module, name: str, replacement: torch.nn.Module) -> None:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


def make_edit_batch(
    tokenizer: Any,
    row: dict[str, Any],
    device: torch.device,
    max_length: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    edit_prompt = qa_prompt(str(row["update"]))
    answer = " " + str(row["ans"]).strip() + (tokenizer.eos_token or "")
    subject = str(row.get("subject") or row["update"])
    texts = [edit_prompt + answer, subject]
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        tokens = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
    finally:
        tokenizer.padding_side = old_side
    labels = tokens["input_ids"].clone()
    prompt_length = len(
        tokenizer(
            edit_prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        ).input_ids
    )
    labels[0, :prompt_length] = -100
    labels[0, tokens["attention_mask"][0].eq(0)] = -100
    labels[1].fill_(-100)
    tokens["labels"] = labels

    edit_ids = tokens["input_ids"][0].tolist()
    subject_variants = [
        tokenizer(" " + subject, add_special_tokens=False).input_ids,
        tokenizer(subject, add_special_tokens=False).input_ids,
    ]
    start = next(
        (found for pattern in subject_variants if (found := find_subsequence(edit_ids, pattern)) is not None),
        None,
    )
    valid = tokens["attention_mask"][0].bool()
    active = torch.zeros_like(valid)
    if start is None:
        active[:prompt_length] = valid[:prompt_length]
    else:
        pattern = next(pattern for pattern in subject_variants if find_subsequence(edit_ids, pattern) == start)
        active[start : start + len(pattern)] = True
    inactive = valid & ~active
    return (
        {key: value.to(device) for key, value in tokens.items()},
        active.unsqueeze(0).to(device),
        inactive.unsqueeze(0).to(device),
    )


def masked_answer_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]).float(),
        shifted_labels.view(-1),
        ignore_index=-100,
    )


def activation_margin_loss(
    original: torch.Tensor,
    candidate: torch.Tensor,
    active_mask: torch.Tensor,
    inactive_mask: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edit_original = original[:1]
    edit_candidate = candidate[:1]
    in_scope = activation_distance(edit_original, edit_candidate, mask=active_mask)
    out_scope = activation_distance(edit_original, edit_candidate, mask=inactive_mask)
    ranking = F.relu(out_scope[:, None] - in_scope + gamma).mean()
    upper = F.relu(out_scope - alpha).mean()
    lower = F.relu(beta - in_scope).mean()
    return ranking + upper + lower, in_scope.mean(), out_scope.mean()


def train_one_edit(
    model: torch.nn.Module,
    adapter: WISEAdapter,
    tokenizer: Any,
    row: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    tokens, active_mask, inactive_mask = make_edit_batch(
        tokenizer, row, device, args.max_length
    )
    if adapter.edit_count % args.save_frequency == 0:
        adapter.weight_mask = torch.rand_like(adapter.new_weight, dtype=torch.float32).lt(
            args.mask_ratio
        )
    adapter.editing = True
    adapter.new_weight.requires_grad_(True)
    optimizer = torch.optim.SGD([adapter.new_weight], lr=args.edit_lr, weight_decay=1e-5)
    previous = 1e9
    current = 1e9
    iteration_log: list[dict[str, float]] = []
    started = time.perf_counter()
    for iteration in range(1, args.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            use_cache=False,
        )
        if adapter.original_output is None or adapter.new_output is None:
            raise RuntimeError("WISE adapter did not capture edit activations")
        fact_loss = masked_answer_loss(outputs.logits[:1], tokens["labels"][:1])
        act_loss, in_scope, out_scope = activation_margin_loss(
            adapter.original_output,
            adapter.new_output,
            active_mask,
            inactive_mask,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
        )
        loss = fact_loss + act_loss.to(fact_loss.device)
        if abs(current - previous) <= 1e-4 and current <= 0.02:
            break
        loss.backward()
        if adapter.new_weight.grad is None or adapter.weight_mask is None:
            raise RuntimeError("WISE gradient mask state is missing")
        adapter.new_weight.grad.mul_(adapter.weight_mask)
        optimizer.step()
        with torch.no_grad():
            adapter.new_weight.clamp_(
                min=adapter.original_layer.weight - args.norm_constraint,
                max=adapter.original_layer.weight + args.norm_constraint,
            )
        previous, current = current, float(loss.detach())
        iteration_log.append(
            {
                "iteration": float(iteration),
                "fact_loss": float(fact_loss.detach()),
                "activation_loss": float(act_loss.detach()),
                "total_loss": current,
                "in_scope_distance": float(in_scope.detach()),
                "out_scope_distance": float(out_scope.detach()),
            }
        )

    if adapter.original_output is None or adapter.new_output is None:
        raise RuntimeError("WISE adapter activation state vanished")
    with torch.no_grad():
        final_distance = activation_distance(
            adapter.original_output[:1], adapter.new_output[:1]
        ).mean()
        adapter.current_meter.update(float(final_distance))
    adapter.editing = False
    adapter.edit_count += 1

    saved = False
    merged = False
    if adapter.edit_count % args.save_frequency == 0:
        adapter.memory_weights.append(adapter.new_weight.detach().clone())
        adapter.memory_meters.append(adapter.current_meter)
        adapter.reset_new_weight()
        saved = True
    if adapter.edit_count % args.merge_frequency == 0:
        merge_count = args.merge_frequency // args.save_frequency
        if len(adapter.memory_weights) < merge_count:
            raise RuntimeError("WISE merge interval has insufficient memories")
        to_merge = adapter.memory_weights[-merge_count:]
        meters = adapter.memory_meters[-merge_count:]
        merged_weight = ties_merge(
            adapter.original_layer.weight,
            to_merge,
            density=args.ties_density,
        )
        del adapter.memory_weights[-merge_count:]
        del adapter.memory_meters[-merge_count:]
        merged_meter = ActivationMeter(minimum=min(meter.minimum for meter in meters))
        adapter.memory_weights.append(merged_weight)
        adapter.memory_meters.append(merged_meter)
        merged = True
    model.zero_grad(set_to_none=True)
    return {
        "sample_id": row["sample_id"],
        "iterations": len(iteration_log),
        "final": iteration_log[-1] if iteration_log else None,
        "wall_seconds": time.perf_counter() - started,
        "saved_memory": saved,
        "merged_memory": merged,
        "memory_count": len(adapter.memory_weights),
        "current_activation_min": (
            adapter.current_meter.minimum
            if math.isfinite(adapter.current_meter.minimum)
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0 < args.mask_ratio <= 1:
        raise ValueError("--mask-ratio must lie in (0, 1]")
    if args.save_frequency <= 0 or args.merge_frequency <= 0:
        raise ValueError("save/merge frequencies must be positive")
    if args.merge_frequency % args.save_frequency:
        raise ValueError("merge frequency must be divisible by save frequency")
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
    if not 0 <= args.layer < int(model.config.num_hidden_layers):
        raise ValueError(f"layer {args.layer} outside model depth")
    module_name = f"model.layers.{args.layer}.mlp.down_proj"
    original_module = get_module(model, module_name)
    adapter = WISEAdapter(original_module, activation_ratio=args.activation_ratio).to(device)
    replace_module(model, module_name, adapter)

    config = {
        **vars(args),
        "model_dir": str(args.model_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "save_adapter": str(args.save_adapter.resolve()) if args.save_adapter else None,
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_protocol_id": metadata.get("protocol_id"),
        "method": "WISE-Retrieve",
        "method_family": "side parametric memory with activation router",
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "qwen3_preset_status": "relative-depth transfer; not officially tuned for Qwen3",
        "edit_module": module_name,
        "context_templates": ["identity fixed QA template"],
        "num_facts": len(selected_rows),
        "stage_counts": {str(stage): len(stage_rows) for stage, stage_rows in stages.items()},
        "closed_book_evaluation": True,
        "prompt_template": "Question: <question>\\nAnswer:",
        "router_batch_fix": "per-example routing instead of upstream Tensor.item()",
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

    training_log: list[dict[str, Any]] = []
    seen: list[dict[str, Any]] = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    for stage, current_rows in stages.items():
        stage_started = time.perf_counter()
        edits = [
            train_one_edit(model, adapter, tokenizer, row, device, args)
            for row in current_rows
        ]
        seen.extend(current_rows)
        stage_log = {
            "stage": stage,
            "current_count": len(current_rows),
            "stage_wall_seconds": time.perf_counter() - stage_started,
            "peak_vram_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
            "memory_count": len(adapter.memory_weights),
            "edit_count": adapter.edit_count,
            "mean_iterations": sum(item["iterations"] for item in edits) / len(edits),
            "mean_edit_seconds": sum(item["wall_seconds"] for item in edits) / len(edits),
            "last_edit": edits[-1],
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

    if args.save_adapter is not None:
        args.save_adapter.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "wise_retrieve_adapter/v1",
                "base_model": str(args.model_dir.resolve()),
                "module_name": module_name,
                "new_weight": adapter.new_weight.detach().cpu(),
                "current_meter": vars(adapter.current_meter),
                "memory_weights": [weight.cpu() for weight in adapter.memory_weights],
                "memory_meters": [vars(meter) for meter in adapter.memory_meters],
                "edit_count": adapter.edit_count,
            },
            args.save_adapter,
        )
    atomic_json(args.output_dir / "metrics.json", metrics)
    atomic_json(
        args.output_dir / "completed.json",
        {
            "status": "complete",
            "method": "WISE-Retrieve",
            "elapsed_seconds": time.time() - started,
            "edit_count": adapter.edit_count,
            "memory_count": len(adapter.memory_weights),
        },
    )


if __name__ == "__main__":
    main()
