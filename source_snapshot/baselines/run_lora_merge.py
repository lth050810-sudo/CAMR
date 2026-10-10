#!/usr/bin/env python3
"""WikiBigEdit's stage-wise LoRA-Merge baseline for Qwen3-8B-Base.

The official benchmark trains a fresh q/v LoRA adapter for each chronological
increment, scales its effective update by ``merge_alpha`` (0.25 in the Llama-3
configuration), merges it into the backbone, and repeats from that updated
backbone.  Scaling each LoRA-B matrix before PEFT's merge is algebraically
equivalent to interpolating the affected base and adapted q/v weights.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model

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
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=6.25e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--merge-alpha", type=float, default=0.25)
    parser.add_argument("--locality-panel-size", type=int, default=0)
    parser.add_argument("--multihop-panel-size", type=int, default=0)
    parser.add_argument("--no-stage-eval", action="store_true")
    parser.add_argument("--skip-base-eval", action="store_true")
    parser.add_argument("--save-edited-qv", action="store_true")
    parser.add_argument("--allow-existing-output", action="store_true")
    return parser.parse_args()


def fresh_adapter(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    return get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        ),
    )


def scale_lora_delta(model: torch.nn.Module, alpha: float) -> int:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("merge alpha must be in [0, 1]")
    count = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.mul_(alpha)
                count += 1
    if count == 0:
        raise RuntimeError("no LoRA-B parameters found before merge")
    return count


def qv_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if name.endswith("q_proj.weight") or name.endswith("v_proj.weight")
    }


def clear_merged_peft_metadata(model: torch.nn.Module) -> None:
    """Remove stale PEFT markers left on the unwrapped backbone.

    PEFT 0.17 leaves ``peft_config`` on the returned base model after
    ``merge_and_unload``.  A subsequent fresh adapter otherwise emits a
    multiple-adapter warning and can inherit stale bookkeeping.
    """

    if "peft_config" in getattr(model, "__dict__", {}):
        delattr(model, "peft_config")
    if hasattr(model, "_hf_peft_config_loaded"):
        model._hf_peft_config_loaded = False


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.allow_existing_output:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    rows, metadata = load_manifest(args.manifest)
    stages = group_by_stage(rows)
    locality_panel = manifest_panel(metadata, "locality", limit=args.locality_panel_size) or fixed_panel(
        rows, "loc", "loc_ans", args.locality_panel_size, args.seed + 1001
    )
    multihop_panel = manifest_panel(metadata, "mhop", limit=args.multihop_panel_size) or fixed_panel(
        rows, "mhop", "mhop_ans", args.multihop_panel_size, args.seed + 2001
    )
    tokenizer = load_tokenizer(args.model_dir)
    model = load_model(args.model_dir, device)
    config = {
        **vars(args),
        "model_dir": str(args.model_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_protocol_id": metadata.get("protocol_id"),
        "method": "LoRA-Merge",
        "upstream_repository": "ExplainableML/WikiBigEdit",
        "upstream_semantics": "fresh q/v adapter per increment; merge; interpolate affected weights",
        "closed_book_evaluation": True,
        **environment_metadata(),
    }
    atomic_json(args.output_dir / "run_config.json", config)
    metrics: list[dict[str, Any]] = []
    if not args.skip_base_eval:
        metrics.append(
            evaluate_and_write(
                model, tokenizer, rows, locality_panel, multihop_panel, device, args,
                checkpoint="base", current_stage=None,
            )
        )

    seen: list[dict[str, Any]] = []
    training_log: list[dict[str, Any]] = []
    started = time.time()
    for stage, current in stages.items():
        adapter = fresh_adapter(model, args)
        stats, _optimizer = train_answer_only(
            adapter,
            tokenizer,
            current,
            device,
            seed=args.seed + stage,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            max_length=args.max_length,
            optimizer=None,
        )
        scaled_modules = scale_lora_delta(adapter, args.merge_alpha)
        model = adapter.merge_and_unload(safe_merge=True)
        clear_merged_peft_metadata(model)
        seen.extend(current)
        stage_log = {
            "stage": stage,
            "current_count": len(current),
            "scaled_lora_b_modules": scaled_modules,
            **asdict(stats),
        }
        training_log.append(stage_log)
        atomic_json(args.output_dir / "training_log.json", training_log)
        if not args.no_stage_eval or stage == max(stages):
            metrics.append(
                evaluate_and_write(
                    model, tokenizer, seen, locality_panel, multihop_panel, device, args,
                    checkpoint=f"tau{stage}", current_stage=stage,
                )
            )
            atomic_json(args.output_dir / "metrics.json", metrics)

    if args.save_edited_qv:
        torch.save(
            {
                "format": "qwen3_qv_replacement/v1",
                "base_model": str(args.model_dir.resolve()),
                "merge_alpha": args.merge_alpha,
                "state_dict": qv_state_dict(model),
            },
            args.output_dir / "edited_qv_weights.pt",
        )
    atomic_json(args.output_dir / "metrics.json", metrics)
    atomic_json(
        args.output_dir / "completed.json",
        {"status": "complete", "elapsed_seconds": time.time() - started},
    )


if __name__ == "__main__":
    main()
