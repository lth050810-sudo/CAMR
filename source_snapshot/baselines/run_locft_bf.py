#!/usr/bin/env python3
"""Port the official LocFT-BF recipe to local Qwen3-8B-Base.

The upstream Qwen2.5 configuration updates only layer 6's MLP down projection,
uses batches of 100, Adam at 5e-4, at most 25 breadth-first epochs, and stops
when mean answer loss drops below 1e-2.  This runner preserves those choices
while using the shared closed-book prompt and WikiBigEdit manifest.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch

from baselines.common import (
    atomic_json,
    environment_metadata,
    fixed_panel,
    group_by_stage,
    load_manifest,
    load_model,
    load_tokenizer,
    manifest_panel,
    sha256_file,
    train_answer_only,
)
from baselines.run_lora_family import evaluate_and_write


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("Qwen3-8B-Base"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("batch", "sequential"), default="sequential")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--loss-threshold", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--locality-panel-size", type=int, default=0)
    parser.add_argument("--multihop-panel-size", type=int, default=0)
    parser.add_argument("--no-stage-eval", action="store_true")
    parser.add_argument("--skip-base-eval", action="store_true")
    parser.add_argument("--allow-existing-output", action="store_true")
    return parser.parse_args()


def locate_weight(model: torch.nn.Module, layer: int) -> tuple[str, torch.nn.Parameter]:
    expected = f"model.layers.{layer}.mlp.down_proj.weight"
    for name, parameter in model.named_parameters():
        if name == expected:
            return name, parameter
    matches = [name for name, _ in model.named_parameters() if f"layers.{layer}.mlp.down_proj" in name]
    raise ValueError(f"could not locate {expected}; nearby matches={matches}")


def train_until_threshold(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    device: torch.device,
    args: argparse.Namespace,
    seed_offset: int,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[list[dict[str, Any]], torch.optim.Optimizer]:
    epoch_log: list[dict[str, Any]] = []
    assert args.epochs > 0
    for epoch in range(1, args.epochs + 1):
        stats, optimizer = train_answer_only(
            model,
            tokenizer,
            rows,
            device,
            seed=args.seed + args.layer * 10_000 + seed_offset * 100 + epoch,
            epochs=1,
            batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            max_length=args.max_length,
            optimizer=optimizer,
        )
        epoch_log.append({"epoch": epoch, **asdict(stats)})
        if stats.loss < args.loss_threshold:
            break
    return epoch_log, optimizer


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.allow_existing_output:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    rows, metadata = load_manifest(args.manifest)
    stages = group_by_stage(rows)
    locality_panel = manifest_panel(metadata, "locality", limit=args.locality_panel_size) or fixed_panel(
        rows, "loc", "loc_ans", args.locality_panel_size, 1043
    )
    multihop_panel = manifest_panel(metadata, "mhop", limit=args.multihop_panel_size) or fixed_panel(
        rows, "mhop", "mhop_ans", args.multihop_panel_size, 2043
    )
    tokenizer = load_tokenizer(args.model_dir)
    model = load_model(args.model_dir, device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    parameter_name, edited_weight = locate_weight(model, args.layer)
    edited_weight.requires_grad = True
    original_weight = edited_weight.detach().cpu().clone()
    config = {
        **vars(args),
        "model_dir": str(args.model_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_protocol_id": metadata.get("protocol_id"),
        "method": "LocFT-BF",
        "upstream_repository": "WanliYoung/FT4Editing",
        "upstream_qwen25_recipe": {
            "layer": 6,
            "rewrite_module": "model.layers.{}.mlp.down_proj.weight",
            "lr": 5e-4,
            "batch_size": 100,
            "num_steps": 25,
            "loss_threshold": 1e-2,
        },
        "edited_parameter": parameter_name,
        "trainable_parameters": edited_weight.numel(),
        "closed_book_evaluation": True,
        **environment_metadata(),
    }
    atomic_json(args.output_dir / "run_config.json", config)
    metrics: list[dict[str, Any]] = []
    if not args.skip_base_eval:
        metrics.append(
            evaluate_and_write(
                model,
                tokenizer,
                rows,
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
    optimizer: torch.optim.Optimizer | None = None
    started = time.time()
    if args.mode == "batch":
        epochs, optimizer = train_until_threshold(
            model, tokenizer, rows, device, args, seed_offset=0, optimizer=optimizer
        )
        training_log.append({"scope": "all_3k", "num_facts": len(rows), "epochs": epochs})
        metrics.append(
            evaluate_and_write(
                model,
                tokenizer,
                rows,
                locality_panel,
                multihop_panel,
                device,
                args,
                checkpoint="batch_3k",
                current_stage=max(stages),
            )
        )
    else:
        seen: list[dict[str, Any]] = []
        for stage, current in stages.items():
            # The upstream script creates one optimizer for a full BF run.  We
            # keep it across chronological stages for the continual variant.
            epochs, optimizer = train_until_threshold(
                model, tokenizer, current, device, args, seed_offset=stage, optimizer=optimizer
            )
            seen.extend(current)
            training_log.append(
                {"scope": f"tau{stage}", "num_facts": len(current), "epochs": epochs}
            )
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

    delta = edited_weight.detach().cpu() - original_weight
    torch.save(
        {
            "format": "single_parameter_delta/v1",
            "base_model": str(args.model_dir.resolve()),
            "parameter_name": parameter_name,
            "delta": delta,
        },
        args.output_dir / "locft_delta.pt",
    )
    atomic_json(args.output_dir / "training_log.json", training_log)
    atomic_json(args.output_dir / "metrics.json", metrics)
    atomic_json(
        args.output_dir / "completed.json",
        {
            "status": "complete",
            "elapsed_seconds": time.time() - started,
            "delta_frobenius_norm": float(delta.float().norm()),
        },
    )


if __name__ == "__main__":
    main()
