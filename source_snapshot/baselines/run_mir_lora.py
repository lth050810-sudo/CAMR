#!/usr/bin/env python3
"""Run Maximally Interfered Retrieval (MIR) with LoRA on WikiBigEdit.

This is the online MIR algorithm from Aljundi et al. (NeurIPS 2019), adapted
to answer-only causal-language-model loss.  For every incoming minibatch it:

1. samples a uniform candidate pool from a reservoir of previously seen facts;
2. measures every candidate's answer loss at the current parameters;
3. makes one *virtual* SGD step on the incoming minibatch (LoRA parameters only);
4. ranks candidates by post-step minus pre-step answer loss;
5. restores the parameters exactly; and
6. makes the real AdamW step on the incoming and top-ranked replay examples.

The candidate scores and selected IDs are written to ``mir_trace.jsonl`` so a
run can be audited rather than inferred from aggregate replay counts.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from baselines.common import (
    AnswerOnlyDataset,
    Reservoir,
    append_jsonl,
    atomic_json,
    environment_metadata,
    fixed_panel,
    group_by_stage,
    load_manifest,
    load_model,
    load_tokenizer,
    make_collator,
    manifest_panel,
    sha256_file,
)
from baselines.run_lora_family import evaluate_and_write


@dataclass
class MIRStageStats:
    loss: float
    optimizer_steps: int
    sequence_tokens: int
    answer_tokens: int
    wall_seconds: float
    peak_vram_gb: float
    candidate_examples_scored: int
    replay_examples_selected: int
    minibatches_with_candidates: int
    minibatches_without_candidates: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("Qwen3-8B-Base"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="online passes over each stage; reservoir insertion happens only on first sight",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument(
        "--target-modules",
        default="q_proj,v_proj",
        help="comma-separated PEFT target module suffixes",
    )
    parser.add_argument("--memory-capacity", type=int, default=1000)
    parser.add_argument("--candidate-pool-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=4)
    parser.add_argument(
        "--mir-score-batch-size",
        type=int,
        default=8,
        help="microbatch size for per-example candidate loss scoring",
    )
    parser.add_argument(
        "--mir-virtual-step-size",
        type=float,
        default=None,
        help="SGD step used only for MIR ranking (defaults to --learning-rate)",
    )
    parser.add_argument("--locality-panel-size", type=int, default=0)
    parser.add_argument("--multihop-panel-size", type=int, default=0)
    parser.add_argument("--save-every-adapter", action="store_true")
    parser.add_argument("--no-stage-eval", action="store_true")
    parser.add_argument("--skip-base-eval", action="store_true")
    parser.add_argument(
        "--max-stages",
        type=int,
        default=0,
        help="debug only: keep the first N chronological stages; 0 keeps all",
    )
    parser.add_argument(
        "--examples-per-stage",
        type=int,
        default=0,
        help="debug only: keep the frozen prefix of each stage; 0 keeps all",
    )
    parser.add_argument("--allow-existing-output", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limited_stages(
    rows: Sequence[dict[str, Any]],
    max_stages: int,
    examples_per_stage: int,
) -> dict[int, list[dict[str, Any]]]:
    stages = group_by_stage(rows)
    if max_stages > 0:
        stages = dict(list(stages.items())[:max_stages])
    if examples_per_stage > 0:
        stages = {
            stage: stage_rows[:examples_per_stage]
            for stage, stage_rows in stages.items()
        }
    return stages


def lora_model(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    if not target_modules:
        raise ValueError("--target-modules cannot be empty")
    wrapped = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        ),
    )
    wrapped.print_trainable_parameters()
    return wrapped


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return parameters


def batch_to_device(
    indices: Sequence[int],
    dataset: AnswerOnlyDataset,
    collator: Any,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not indices:
        raise ValueError("cannot collate an empty batch")
    batch = collator([dataset[index] for index in indices])
    return {key: value.to(device, non_blocking=device.type == "cuda") for key, value in batch.items()}


def per_example_answer_losses(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return one mean next-token loss per example, ignoring prompt tokens."""

    outputs = model(**batch, use_cache=False)
    logits = outputs.logits[:, :-1, :].float()
    labels = batch["labels"][:, 1:]
    flat_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(labels)
    valid = labels.ne(-100)
    token_counts = valid.sum(dim=1)
    if bool((token_counts == 0).any()):
        raise ValueError("candidate batch contains an example with no answer tokens")
    return (flat_losses * valid).sum(dim=1) / token_counts


def candidate_losses(
    model: torch.nn.Module,
    candidate_indices: Sequence[int],
    dataset: AnswerOnlyDataset,
    collator: Any,
    device: torch.device,
    score_batch_size: int,
) -> torch.Tensor:
    if score_batch_size <= 0:
        raise ValueError("MIR score batch size must be positive")
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(candidate_indices), score_batch_size):
            indices = candidate_indices[start : start + score_batch_size]
            batch = batch_to_device(indices, dataset, collator, device)
            chunks.append(per_example_answer_losses(model, batch).detach().cpu())
    return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float32)


def rank_interference(
    candidate_indices: Sequence[int],
    before_losses: torch.Tensor,
    after_losses: torch.Tensor,
    replay_size: int,
) -> tuple[list[int], list[float]]:
    if len(candidate_indices) != before_losses.numel() or len(candidate_indices) != after_losses.numel():
        raise ValueError("candidate IDs and MIR losses have inconsistent lengths")
    scores = (after_losses.float() - before_losses.float()).tolist()
    # Stable position tie-break makes selection reproducible even when many
    # bf16 losses quantize to the same value.
    ranking = sorted(range(len(scores)), key=lambda position: (-scores[position], position))
    selected_positions = ranking[: min(max(replay_size, 0), len(ranking))]
    return [int(candidate_indices[position]) for position in selected_positions], scores


def mir_select(
    model: torch.nn.Module,
    current_batch: dict[str, torch.Tensor],
    candidate_indices: Sequence[int],
    dataset: AnswerOnlyDataset,
    collator: Any,
    device: torch.device,
    *,
    replay_size: int,
    virtual_step_size: float,
    score_batch_size: int,
) -> tuple[list[int], torch.Tensor, torch.Tensor, list[float], float]:
    """Select replay examples and restore model parameters before returning."""

    if virtual_step_size < 0:
        raise ValueError("MIR virtual step size must be non-negative")
    if not candidate_indices or replay_size <= 0:
        empty = torch.empty(0, dtype=torch.float32)
        return [], empty, empty, [], 0.0

    parameters = trainable_parameters(model)
    was_training = model.training
    old_cache = model.config.use_cache
    model.eval()  # deterministic MIR scores: LoRA dropout is disabled here.
    model.config.use_cache = False
    originals: list[torch.Tensor] = []
    try:
        before = candidate_losses(
            model, candidate_indices, dataset, collator, device, score_batch_size
        )
        current_outputs = model(**current_batch, use_cache=False)
        current_loss = current_outputs.loss
        gradients = torch.autograd.grad(
            current_loss,
            parameters,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        originals = [parameter.detach().clone() for parameter in parameters]
        with torch.no_grad():
            for parameter, gradient in zip(parameters, gradients):
                if gradient is not None:
                    parameter.add_(gradient, alpha=-virtual_step_size)
        after = candidate_losses(
            model, candidate_indices, dataset, collator, device, score_batch_size
        )
    finally:
        if originals:
            with torch.no_grad():
                for parameter, original in zip(parameters, originals):
                    parameter.copy_(original)
        model.config.use_cache = old_cache
        model.train(was_training)

    selected, scores = rank_interference(candidate_indices, before, after, replay_size)
    return selected, before, after, scores, float(current_loss.detach())


def sample_candidates(
    reservoir: Reservoir,
    count: int,
    excluded: set[int],
) -> list[int]:
    """Uniformly sample without replacement from eligible reservoir entries."""

    if count <= 0:
        return []
    eligible = [index for index in reservoir.items if index not in excluded]
    return reservoir.rng.sample(eligible, min(count, len(eligible)))


def make_optimizer(
    parameters: Sequence[torch.nn.Parameter],
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    try:
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            fused=True,
        )
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )


def train_mir_stage(
    model: torch.nn.Module,
    dataset: AnswerOnlyDataset,
    collator: Any,
    rows: Sequence[dict[str, Any]],
    current_indices: Sequence[int],
    reservoir: Reservoir,
    already_inserted: set[int],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    stage: int,
    trace_path: Path | None,
) -> MIRStageStats:
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if args.candidate_pool_size < 0 or args.replay_size < 0:
        raise ValueError("candidate pool size and replay size cannot be negative")
    parameters = trainable_parameters(model)
    virtual_step_size = (
        args.learning_rate if args.mir_virtual_step_size is None else args.mir_virtual_step_size
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.config.use_cache = False
    loss_total = 0.0
    optimizer_steps = 0
    sequence_tokens = 0
    answer_tokens = 0
    candidate_total = 0
    replay_total = 0
    minibatches_with_candidates = 0
    minibatches_without_candidates = 0
    trace_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        generator = torch.Generator()
        generator.manual_seed(args.seed + stage * 10_000 + epoch)
        order = torch.randperm(len(current_indices), generator=generator).tolist()
        shuffled = [int(current_indices[position]) for position in order]
        for batch_number, start in enumerate(range(0, len(shuffled), args.batch_size), start=1):
            incoming = shuffled[start : start + args.batch_size]
            current_batch = batch_to_device(incoming, dataset, collator, device)
            candidates = []
            if args.replay_size > 0:
                candidates = sample_candidates(
                    reservoir,
                    args.candidate_pool_size,
                    excluded=set(incoming),
                )
            optimizer.zero_grad(set_to_none=True)
            selected, before, after, scores, virtual_current_loss = mir_select(
                model,
                current_batch,
                candidates,
                dataset,
                collator,
                device,
                replay_size=args.replay_size,
                virtual_step_size=virtual_step_size,
                score_batch_size=args.mir_score_batch_size,
            )

            actual_indices = incoming + selected
            actual_batch = batch_to_device(actual_indices, dataset, collator, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**actual_batch, use_cache=False)
            loss = outputs.loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            optimizer.step()

            loss_total += float(loss.detach())
            optimizer_steps += 1
            sequence_tokens += int(actual_batch["attention_mask"].sum())
            answer_tokens += int(actual_batch["labels"].ne(-100).sum())
            candidate_total += len(scores)
            replay_total += len(selected)
            if candidates:
                minibatches_with_candidates += 1
            else:
                minibatches_without_candidates += 1

            # Each stream item enters the reservoir exactly once.  This avoids
            # overweighting it when --epochs > 1 while allowing later batches
            # in the same chronological stage to retrieve earlier facts.
            new_items = [index for index in incoming if index not in already_inserted]
            reservoir.add_many(new_items)
            already_inserted.update(new_items)

            selected_set = set(selected)
            trace_rows.append(
                {
                    "stage": stage,
                    "epoch": epoch,
                    "minibatch": batch_number,
                    "current_sample_ids": [rows[index]["sample_id"] for index in incoming],
                    "candidate_count": len(candidates),
                    "replay_count": len(selected),
                    "candidate_sample_ids": [rows[index]["sample_id"] for index in candidates],
                    "candidate_loss_before": [float(value) for value in before.tolist()],
                    "candidate_loss_after_virtual_step": [float(value) for value in after.tolist()],
                    "interference_scores": [float(value) for value in scores],
                    "selected_sample_ids": [rows[index]["sample_id"] for index in selected],
                    "selected_candidate_positions": [
                        position for position, index in enumerate(candidates) if index in selected_set
                    ],
                    "virtual_current_loss": virtual_current_loss,
                    "actual_combined_loss": float(loss.detach()),
                    "actual_grad_norm_before_clip": float(grad_norm),
                    "reservoir_size_after_step": len(reservoir.items),
                    "reservoir_seen_after_step": reservoir.seen,
                    "virtual_optimizer": "one-step SGD on current answer loss",
                    "actual_optimizer": "AdamW on concatenated current+selected answer loss",
                }
            )
            if trace_path is not None and len(trace_rows) >= 32:
                append_jsonl(trace_path, trace_rows)
                trace_rows.clear()

    if trace_path is not None and trace_rows:
        append_jsonl(trace_path, trace_rows)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    else:
        peak_vram_gb = 0.0
    return MIRStageStats(
        loss=loss_total / max(optimizer_steps, 1),
        optimizer_steps=optimizer_steps,
        sequence_tokens=sequence_tokens,
        answer_tokens=answer_tokens,
        wall_seconds=time.perf_counter() - started,
        peak_vram_gb=peak_vram_gb,
        candidate_examples_scored=candidate_total,
        replay_examples_selected=replay_total,
        minibatches_with_candidates=minibatches_with_candidates,
        minibatches_without_candidates=minibatches_without_candidates,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen3 MIR runner")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.allow_existing_output:
        raise FileExistsError(
            f"non-empty output directory: {args.output_dir}; pass --allow-existing-output to reuse"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    all_rows, manifest_metadata = load_manifest(args.manifest)
    stages = limited_stages(all_rows, args.max_stages, args.examples_per_stage)
    if not stages or any(not stage_rows for stage_rows in stages.values()):
        raise ValueError("empty selection after applying debug limits")
    rows = [row for stage_rows in stages.values() for row in stage_rows]
    locality_panel = manifest_panel(
        manifest_metadata, "locality", limit=args.locality_panel_size
    ) or fixed_panel(rows, "loc", "loc_ans", args.locality_panel_size, args.seed + 1001)
    multihop_panel = manifest_panel(
        manifest_metadata, "mhop", limit=args.multihop_panel_size
    ) or fixed_panel(rows, "mhop", "mhop_ans", args.multihop_panel_size, args.seed + 2001)

    tokenizer = load_tokenizer(args.model_dir)
    model = load_model(args.model_dir, device)
    virtual_step_size = (
        args.learning_rate if args.mir_virtual_step_size is None else args.mir_virtual_step_size
    )
    config = {
        **vars(args),
        "model_dir": str(args.model_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_protocol_id": manifest_metadata.get("protocol_id"),
        "method": "MIR + LoRA",
        "num_facts": len(rows),
        "stage_counts": {str(stage): len(stage_rows) for stage, stage_rows in stages.items()},
        "locality_panel_size_actual": len(locality_panel),
        "multihop_panel_size_actual": len(multihop_panel),
        "closed_book_evaluation": True,
        "prompt_template": "Question: <question>\\nAnswer:",
        "answer_loss_only": True,
        "mir_algorithm": {
            "reference": "Aljundi et al., Online Continual Learning with Maximally Interfered Retrieval, NeurIPS 2019",
            "memory": "uniform reservoir over each fact's first stream occurrence",
            "candidate_sampling": "uniform without replacement from eligible reservoir entries",
            "candidate_pool_size_requested_per_minibatch": args.candidate_pool_size,
            "replay_size_requested_per_minibatch": args.replay_size,
            "score": "answer_loss(candidate, theta - virtual_lr * grad_current) - answer_loss(candidate, theta)",
            "score_reduction": "mean over non-masked answer tokens, independently per candidate",
            "virtual_optimizer": "one-step SGD without momentum or weight decay",
            "virtual_step_size": virtual_step_size,
            "virtual_parameters": "all and only trainable LoRA parameters",
            "selection_mode": "eval (dropout disabled), parameters restored exactly before real step",
            "actual_optimizer": "AdamW on one concatenated current+selected minibatch",
            "repeated_epoch_policy": "reservoir insertion only on first occurrence; current IDs excluded from candidates",
            "trace_file": "mir_trace.jsonl",
        },
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

    model = lora_model(model, args)
    dataset = AnswerOnlyDataset(rows, tokenizer, args.max_length)
    if len(dataset) != len(rows):
        raise ValueError(
            f"tokenization dropped {len(rows) - len(dataset)} rows; MIR index alignment would be unsafe"
        )
    collator = make_collator(tokenizer)
    parameters = trainable_parameters(model)
    optimizer = make_optimizer(parameters, args.learning_rate, args.weight_decay)
    reservoir = Reservoir(args.memory_capacity, args.seed + 3001)
    already_inserted: set[int] = set()
    row_index = {str(row["sample_id"]): index for index, row in enumerate(rows)}
    seen_rows: list[dict[str, Any]] = []
    training_log: list[dict[str, Any]] = []
    trace_path = args.output_dir / "mir_trace.jsonl"
    started = time.time()

    for stage, current_rows in stages.items():
        current_indices = [row_index[str(row["sample_id"])] for row in current_rows]
        stats = train_mir_stage(
            model,
            dataset,
            collator,
            rows,
            current_indices,
            reservoir,
            already_inserted,
            optimizer,
            device,
            args,
            stage,
            trace_path,
        )
        seen_rows.extend(current_rows)
        stage_log = {
            "stage": stage,
            "current_count": len(current_rows),
            "reservoir_count_after_stage": len(reservoir.items),
            "reservoir_seen_after_stage": reservoir.seen,
            "candidate_pool_size_requested_per_minibatch": args.candidate_pool_size,
            "replay_size_requested_per_minibatch": args.replay_size,
            **asdict(stats),
        }
        training_log.append(stage_log)
        atomic_json(args.output_dir / "training_log.json", training_log)

        if args.save_every_adapter or stage == max(stages):
            adapter_dir = args.output_dir / "adapters" / f"tau{stage}"
            model.save_pretrained(adapter_dir, safe_serialization=True)
            if stage == max(stages):
                tokenizer.save_pretrained(adapter_dir)

        if not args.no_stage_eval or stage == max(stages):
            checkpoint_metrics = evaluate_and_write(
                model,
                tokenizer,
                seen_rows,
                locality_panel,
                multihop_panel,
                device,
                args,
                checkpoint=f"tau{stage}",
                current_stage=stage,
            )
            checkpoint_metrics["train"] = stage_log
            metrics.append(checkpoint_metrics)
            atomic_json(args.output_dir / "metrics.json", metrics)

    aggregate = {
        "candidate_examples_scored": sum(row["candidate_examples_scored"] for row in training_log),
        "replay_examples_selected": sum(row["replay_examples_selected"] for row in training_log),
        "optimizer_steps": sum(row["optimizer_steps"] for row in training_log),
    }
    atomic_json(
        args.output_dir / "completed.json",
        {
            "status": "complete",
            "method": "mir_lora",
            "elapsed_seconds": time.time() - started,
            "final_stage": max(stages),
            **aggregate,
        },
    )


if __name__ == "__main__":
    main()
