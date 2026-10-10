#!/usr/bin/env python3
"""Run the directly portable WikiBigEdit baselines on Qwen3-8B-Base.

Supported methods:

* ``pre_edit``: frozen base model evaluation.
* ``lora_ft``: sequential breadth-first LoRA, current stage only.
* ``er_reservoir``: LoRA with uniformly sampled reservoir replay.
* ``semantic_replay``: LoRA with nearest replay items from the same reservoir.

LoRA-Merge and MIR deliberately live in separate runners because their update
semantics differ from ordinary sequential optimization.  Keeping them out of
this switch prevents silently relabelling an approximate implementation as a
published baseline.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch
from peft import LoraConfig, get_peft_model

from baselines.common import (
    Reservoir,
    append_jsonl,
    atomic_json,
    dense_prompt_embeddings,
    environment_metadata,
    evaluate_checkpoint,
    fixed_panel,
    group_by_stage,
    load_manifest,
    load_model,
    load_tokenizer,
    manifest_panel,
    select_semantic_replay,
    sha256_file,
    train_answer_only,
)


METHODS = ("pre_edit", "lora_ft", "er_reservoir", "semantic_replay", "relation_oracle_replay", "cumulative_replay", "entity_masked_semantic_replay", "tic_style_replay")


def entity_masked_rows(rows):
    result = []
    for row in rows:
        text = str(row['update'])
        for field, replacement in (('subject', '[SUBJECT]'), ('object', '[OBJECT]'), ('ans', '[OBJECT]')):
            value = str(row.get(field) or '').strip()
            if value:
                text = re.sub(r'(?<!\w)' + re.escape(value) + r'(?!\w)', lambda _: replacement, text, flags=re.IGNORECASE)
        result.append({**row, 'update': text})
    return result


def select_relation_replay(rows, current_indices, memory_indices, budget, seed, *, return_pairs=False):
    """Per-current-fact same-relation retrieval from the identical ER reservoir.

    Round-robin current queries, without replacement; preserve the ER example
    budget with a uniform fallback if structural candidates are exhausted.
    """
    rng = random.Random(seed)
    queries = list(current_indices)
    rng.shuffle(queries)
    candidates = {}
    for c in queries:
        current = rows[c]
        candidates[c] = [i for i in memory_indices
            if rows[i]['relation_id'] == current['relation_id']
            and (rows[i].get('subject_id') or rows[i]['subject']) != (current.get('subject_id') or current['subject'])
            and rows[i].get('fact_fingerprint', rows[i]['sample_id']) != current.get('fact_fingerprint', current['sample_id'])]
    selected, used, pairs = [], set(), []
    while len(selected) < min(budget, len(memory_indices)):
        before = len(selected)
        for c in queries:
            available = [i for i in candidates[c] if i not in used]
            if available:
                i = rng.choice(available)
                selected.append(i); used.add(i)
                pairs.append({'current_sample_id': rows[c]['sample_id'], 'replay_sample_id': rows[i]['sample_id'], 'relation_id': rows[c]['relation_id']})
            if len(selected) >= budget: break
        if len(selected) == before: break
    matched = len(selected)
    remaining = [i for i in memory_indices if i not in used]
    selected.extend(rng.sample(remaining, min(max(0, budget-len(selected)), len(remaining))))
    return (selected, matched, pairs) if return_pairs else (selected, matched)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("Qwen3-8B-Base"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="comma-separated PEFT target module suffixes",
    )
    parser.add_argument("--replay-ratio", type=float, default=1.0)
    parser.add_argument("--memory-capacity", type=int, default=1000)
    parser.add_argument(
        "--locality-panel-size",
        type=int,
        default=0,
        help="0 uses the complete panel frozen in the protocol manifest",
    )
    parser.add_argument(
        "--multihop-panel-size",
        type=int,
        default=0,
        help="0 uses the complete panel frozen in the protocol manifest",
    )
    parser.add_argument("--save-every-adapter", action="store_true")
    parser.add_argument("--no-stage-eval", action="store_true")
    parser.add_argument(
        "--skip-base-eval",
        action="store_true",
        help="reuse the separately recorded pre_edit run instead of evaluating base again",
    )
    parser.add_argument("--keep-semantic-embeddings", action="store_true")
    parser.add_argument("--match-er-admission", action="store_true", help="advance the legacy ER sampler RNG before semantic selection, preserving identical reservoir admission")
    parser.add_argument("--allow-existing-output", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def axis_accuracy(records: Sequence[dict[str, Any]], axis: str) -> float | None:
    selected = [record for record in records if record["axis"] == axis]
    if not selected:
        return None
    return sum(bool(record["match"]) for record in selected) / len(selected)


def summarize_checkpoint_records(
    records: Sequence[dict[str, Any]],
    current_stage: int | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for axis in ("update", "rephrase", "persona", "locality", "multihop"):
        result[f"{axis}_accuracy"] = axis_accuracy(records, axis)
    if current_stage is None:
        return result
    factual = [record for record in records if record["axis"] in {"update", "rephrase", "persona"}]
    current = [record for record in factual if int(record["fact_stage"]) == current_stage]
    past = [record for record in factual if int(record["fact_stage"]) < current_stage]
    for axis in ("update", "rephrase", "persona"):
        result[f"current_{axis}_accuracy"] = axis_accuracy(current, axis)
        result[f"past_{axis}_retention"] = axis_accuracy(past, axis)
    per_stage: dict[str, dict[str, float | None]] = {}
    for stage in sorted({int(record["fact_stage"]) for record in factual}):
        stage_records = [record for record in factual if int(record["fact_stage"]) == stage]
        per_stage[str(stage)] = {
            axis: axis_accuracy(stage_records, axis)
            for axis in ("update", "rephrase", "persona")
        }
    result["per_fact_stage"] = per_stage
    return result


def evaluate_and_write(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    locality_panel: Sequence[dict[str, Any]],
    multihop_panel: Sequence[dict[str, Any]],
    device: torch.device,
    args: argparse.Namespace,
    checkpoint: str,
    current_stage: int | None,
) -> dict[str, Any]:
    raw, records = evaluate_checkpoint(
        model,
        tokenizer,
        rows,
        device,
        checkpoint=checkpoint,
        axes=("update", "rephrase", "persona", "locality", "multihop"),
        batch_size=args.eval_batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        locality_panel=locality_panel,
        multihop_panel=multihop_panel,
    )
    append_jsonl(args.output_dir / "predictions.jsonl", records)
    return {
        **raw,
        **summarize_checkpoint_records(records, current_stage),
        "current_stage": current_stage,
    }


def lora_model(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    if not target_modules:
        raise ValueError("--target-modules cannot be empty")
    config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    wrapped = get_peft_model(model, config)
    wrapped.print_trainable_parameters()
    return wrapped


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.allow_existing_output:
        raise FileExistsError(
            f"non-empty output directory: {args.output_dir}; pass --allow-existing-output to reuse"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    rows, manifest_metadata = load_manifest(args.manifest)
    stages = group_by_stage(rows)
    if not stages:
        raise ValueError("empty selection manifest")
    locality_panel = manifest_panel(
        manifest_metadata, "locality", limit=args.locality_panel_size
    ) or fixed_panel(rows, "loc", "loc_ans", args.locality_panel_size, args.seed + 1001)
    multihop_panel = manifest_panel(
        manifest_metadata, "mhop", limit=args.multihop_panel_size
    ) or fixed_panel(rows, "mhop", "mhop_ans", args.multihop_panel_size, args.seed + 2001)
    tokenizer = load_tokenizer(args.model_dir)
    model = load_model(args.model_dir, device)

    config = {
        **vars(args),
        "model_dir": str(args.model_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_protocol_id": manifest_metadata.get("protocol_id"),
        "num_facts": len(rows),
        "stage_counts": {str(stage): len(stage_rows) for stage, stage_rows in stages.items()},
        "locality_panel_size_actual": len(locality_panel),
        "multihop_panel_size_actual": len(multihop_panel),
        "closed_book_evaluation": True,
        "prompt_template": "Question: <question>\\nAnswer:",
        "answer_loss_only": True,
        "reservoir_admission_matches_er": args.method in {"er_reservoir", "tic_style_replay", "relation_oracle_replay"} or args.match_er_admission,
        "stage_lr_schedule": "10% rewarm + cosine decay to 10% peak each stage; persistent Adam moments" if args.method == "tic_style_replay" else "constant",
        "replication_scope": "TiC-style schedule reference, not a reproduction of web-scale TiC-LM" if args.method == "tic_style_replay" else "shared WikiBigEdit protocol",
        "oracle_structure_access": args.method == "relation_oracle_replay",
        "replay_selection_policy": (
            "per-current-query round-robin same gold relation, different subject/fact; uniform count-matched fallback; identical ER reservoir"
            if args.method == "relation_oracle_replay" else args.method
        ),
        "budget_note": "cumulative uses all history and is not compute matched" if args.method == "cumulative_replay" else "same example budget as ER; realized token counts logged",
        **environment_metadata(),
    }
    atomic_json(args.output_dir / "run_config.json", config)

    metrics: list[dict[str, Any]] = []
    if not args.skip_base_eval:
        base_metrics = evaluate_and_write(
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
        metrics.append(base_metrics)
        atomic_json(args.output_dir / "metrics.json", metrics)
    if args.method == "pre_edit":
        atomic_json(args.output_dir / "completed.json", {"status": "complete", "method": args.method})
        return

    semantic_embeddings: torch.Tensor | None = None
    if args.method in {"semantic_replay", "entity_masked_semantic_replay"}:
        cache_path = args.output_dir / "semantic_embeddings.pt"
        semantic_embeddings = dense_prompt_embeddings(
            model,
            tokenizer,
            entity_masked_rows(rows) if args.method == "entity_masked_semantic_replay" else rows,
            device,
            batch_size=args.eval_batch_size,
            max_length=args.max_length,
        )
        torch.save(
            {
                "sample_ids": [row["sample_id"] for row in rows],
                "embeddings": semantic_embeddings.to(torch.float16),
                "encoder": "frozen Qwen3 mean-pooled final prompt states",
            },
            cache_path,
        )

    model = lora_model(model, args)
    row_index = {str(row["sample_id"]): index for index, row in enumerate(rows)}
    reservoir = Reservoir(args.memory_capacity, args.seed + 3001)
    seen_rows: list[dict[str, Any]] = []
    train_log: list[dict[str, Any]] = []
    optimizer: torch.optim.Optimizer | None = None
    started = time.time()

    for stage, current_rows in stages.items():
        current_indices = [row_index[str(row["sample_id"])] for row in current_rows]
        replay_budget = round(len(current_rows) * args.replay_ratio)
        replay_indices: list[int] = []
        relation_matched = 0
        relation_pairs = []
        if args.method in {"er_reservoir", "tic_style_replay"}:
            replay_indices = reservoir.sample(replay_budget)
        elif args.method in {"semantic_replay", "entity_masked_semantic_replay"}:
            if args.match_er_admission:
                reservoir.sample(replay_budget)
            assert semantic_embeddings is not None
            replay_indices = select_semantic_replay(
                current_indices,
                reservoir.items,
                semantic_embeddings.float(),
                replay_budget,
            )
        elif args.method == "relation_oracle_replay":
            reservoir.sample(replay_budget)
            replay_indices, relation_matched, relation_pairs = select_relation_replay(
                rows, current_indices, reservoir.items, replay_budget, args.seed + 4001 + stage, return_pairs=True
            )
        elif args.method == "cumulative_replay":
            replay_indices = [row_index[str(row['sample_id'])] for row in seen_rows]
        if any(int(rows[i]['stage']) >= stage for i in replay_indices):
            raise RuntimeError("replay includes a current or future stage")
        training_rows = list(current_rows) + [rows[index] for index in replay_indices]
        stats, optimizer = train_answer_only(
            model,
            tokenizer,
            training_rows,
            device,
            seed=args.seed + stage,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            max_length=args.max_length,
            optimizer=optimizer,
            stage_lr_schedule=args.method == "tic_style_replay",
        )
        reservoir.add_many(current_indices)
        seen_rows.extend(current_rows)
        stage_log = {
            "stage": stage,
            "current_count": len(current_rows),
            "replay_count": len(replay_indices),
            "relation_matched_count": relation_matched if args.method == "relation_oracle_replay" else None,
            "future_leak_count": 0,
            "relation_query_replay_pairs": relation_pairs,
            "replay_sample_ids": [rows[index]["sample_id"] for index in replay_indices],
            "reservoir_count_after_stage": len(reservoir.items),
            **asdict(stats),
        }
        train_log.append(stage_log)
        atomic_json(args.output_dir / "training_log.json", train_log)

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

    if args.method in {"semantic_replay", "entity_masked_semantic_replay"} and not args.keep_semantic_embeddings:
        (args.output_dir / "semantic_embeddings.pt").unlink(missing_ok=True)
    atomic_json(
        args.output_dir / "completed.json",
        {
            "status": "complete",
            "method": args.method,
            "elapsed_seconds": time.time() - started,
            "final_stage": max(stages),
        },
    )


if __name__ == "__main__":
    main()
