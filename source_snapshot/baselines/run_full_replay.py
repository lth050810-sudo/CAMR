#!/usr/bin/env python3
"""Run source-ordered full-stream LoRA baselines with bounded block replay.

This is a scale-up protocol, not the original 8 x 375-fact experiment: each
original time interval is traversed in blocks (375 facts by default), and
already trained blocks in the same interval are eligible for replay. Adam
moments persist across blocks. Evaluation uses explicitly frozen small panels.
Heavy model dependencies are imported only when training starts, so protocol
and replay/resume checks can run on a CPU without importing a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


AXES = {
    "update": ("update", "ans"),
    "rephrase": ("rephrase", "ans"),
    "persona": ("personas", "ans"),
    "locality": ("loc", "loc_ans"),
    "multihop": ("mhop", "mhop_ans"),
}
METHODS = ("lora_ft", "er_reservoir", "relation_oracle_replay")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def load_stream(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read the frozen selection, retaining numeric source order within stages."""
    metadata = json.loads(path.read_text())
    if not isinstance(metadata, dict) or not isinstance(metadata.get("records"), list):
        raise ValueError("expected a frozen JSON manifest with records and selection.sample_ids")
    selected = metadata.get("selection", {}).get("sample_ids")
    if not isinstance(selected, list) or not selected or len(selected) != len(set(selected)):
        raise ValueError("selection.sample_ids must be nonempty and unique")
    selected_ids = set(selected)
    rows = [dict(row) for row in metadata["records"] if row.get("sample_id") in selected_ids]
    if len(rows) != len(selected_ids) or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("selected IDs are missing or duplicated in manifest records")
    for row in rows:
        row["stage"] = int(row["stage"])
        if row["stage"] < 1 or row["stage"] > 8:
            raise ValueError("WikiBigEdit original stage must be in 1..8")
        if "personas" not in row and "persona" in row:
            row["personas"] = row["persona"]
        if any(not isinstance(row.get(field), str) or not row[field].strip()
               for field in ("sample_id", "update", "ans")):
            raise ValueError(f"invalid training fact {row.get('sample_id')}")
        # Lexicographic sample IDs are not a substitute for chronological row order.
        if row.get("source_global_row_idx") is None and row.get("source_row_idx") is None:
            raise ValueError(f"missing numeric source row index: {row['sample_id']}")
    def source_position(row: dict[str, Any]) -> tuple[int, int]:
        position = row.get("source_global_row_idx")
        return row["stage"], int(row["source_row_idx"] if position is None else position)
    rows.sort(key=source_position)
    positions = [source_position(row) for row in rows]
    if len(positions) != len(set(positions)):
        raise ValueError("duplicate source positions in selected stream")
    return rows, metadata


@dataclass(frozen=True)
class Block:
    index: int
    stage: int
    start: int
    end: int
    stage_end: bool


def make_blocks(rows: list[dict[str, Any]], block_size: int) -> list[Block]:
    if block_size < 1:
        raise ValueError("block size must be positive")
    blocks = []
    position = 0
    while position < len(rows):
        stage = rows[position]["stage"]
        stage_end = position + 1
        while stage_end < len(rows) and rows[stage_end]["stage"] == stage:
            stage_end += 1
        for start in range(position, stage_end, block_size):
            end = min(start + block_size, stage_end)
            blocks.append(Block(len(blocks), stage, start, end, end == stage_end))
        position = stage_end
    return blocks


def relation_selection(rows, current, memory, budget, seed):
    """Same per-query structural selector and uniform fallback as the 3K runner."""
    rng = random.Random(seed)
    queries = list(current)
    rng.shuffle(queries)
    candidates = {}
    for index in queries:
        row = rows[index]
        relation = row.get("relation_id")
        candidates[index] = [old for old in memory
            if relation and rows[old].get("relation_id") == relation
            and (rows[old].get("subject_id") or rows[old].get("subject"))
                != (row.get("subject_id") or row.get("subject"))
            and rows[old].get("fact_fingerprint", rows[old]["sample_id"])
                != row.get("fact_fingerprint", row["sample_id"])]
    selected, used, pairs = [], set(), []
    while len(selected) < min(budget, len(memory)):
        before = len(selected)
        for current_index in queries:
            available = [old for old in candidates[current_index] if old not in used]
            if available:
                old = rng.choice(available)
                selected.append(old)
                used.add(old)
                pairs.append({"current_sample_id": rows[current_index]["sample_id"],
                              "replay_sample_id": rows[old]["sample_id"],
                              "relation_id": rows[current_index]["relation_id"]})
            if len(selected) >= budget:
                break
        if before == len(selected):
            break
    remaining = [old for old in memory if old not in used]
    selected.extend(rng.sample(remaining, min(max(0, budget - len(selected)), len(remaining))))
    return selected, pairs


def _tuples(value):
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


class ReplayState:
    """Reservoir admission and stream cursor, independent of model execution."""

    def __init__(self, capacity: int, seed: int):
        if capacity < 1:
            raise ValueError("memory capacity must be positive")
        self.capacity = capacity
        self.seed = seed
        self.rng = random.Random(seed + 3001)
        self.memory: list[int] = []
        self.seen = 0
        self.next_block = 0

    def prepare(self, rows, block: Block, method: str, replay_ratio: float):
        if method not in METHODS or not math.isfinite(replay_ratio) or replay_ratio < 0:
            raise ValueError("invalid method or replay ratio")
        if block.index != self.next_block or block.start != self.seen:
            raise ValueError("block cursor does not match the trained prefix")
        if any(index >= block.start or index < 0 for index in self.memory):
            raise ValueError("reservoir contains an untrained/future fact")
        budget = round((block.end - block.start) * replay_ratio)
        # Advance the same reservoir RNG for every method. Thus Relation and ER
        # admit exactly the same facts even though retrieval differs.
        er = self.rng.sample(self.memory, min(budget, len(self.memory)))
        pairs = []
        if method == "lora_ft":
            selected = []
        elif method == "er_reservoir":
            selected = er
        else:
            selected, pairs = relation_selection(rows, range(block.start, block.end),
                self.memory, budget, self.seed + 4001 + block.index + 1)
        if len(selected) != len(set(selected)) or any(index >= block.start for index in selected):
            raise ValueError("invalid replay selection")
        return selected, pairs

    def commit(self, block: Block):
        if block.index != self.next_block or block.start != self.seen:
            raise ValueError("cannot commit a block twice or out of order")
        for index in range(block.start, block.end):
            self.seen += 1
            if len(self.memory) < self.capacity:
                self.memory.append(index)
            else:
                replacement = self.rng.randrange(self.seen)
                if replacement < self.capacity:
                    self.memory[replacement] = index
        self.next_block += 1

    def state_dict(self):
        return {"capacity": self.capacity, "seed": self.seed, "memory": list(self.memory),
                "seen": self.seen, "next_block": self.next_block, "rng": self.rng.getstate()}

    @classmethod
    def from_state_dict(cls, value):
        state = cls(value["capacity"], value["seed"])
        state.memory = list(value["memory"])
        state.seen, state.next_block = value["seen"], value["next_block"]
        if (len(state.memory) != min(state.capacity, state.seen)
                or len(state.memory) != len(set(state.memory))
                or any(index < 0 or index >= state.seen for index in state.memory)):
            raise ValueError("invalid reservoir checkpoint")
        state.rng.setstate(_tuples(value["rng"]))
        return state


def freeze_panels(rows, per_stage: int, seed: int, metadata=None):
    if per_stage < 1:
        raise ValueError("panel size must be positive; full-stream evaluation is not implied")
    from baselines.wikibigedit_protocol import normalize_answer
    panels = {}
    ambiguous_questions = {}
    masked_counts = {}
    for axis, (question, answer) in AXES.items():
        manifest_axis = {"persona": "personas", "multihop": "mhop"}.get(axis, axis)
        panel_specification = (metadata or {}).get("panels", {}).get(manifest_axis)
        allowed_ids = set(panel_specification["sample_ids"]) if panel_specification is not None else None
        def available(row):
            if allowed_ids is not None and row["sample_id"] not in allowed_ids:
                return False
            audits = [row.get("panel_audit", {}).get(manifest_axis),
                      row.get("token_length_audit", {}).get("panels", {}).get(manifest_axis)]
            audits = [audit for audit in audits if audit is not None]
            if audits and not all(audit.get("available") is True for audit in audits):
                return False
            # In a real frozen manifest, eligibility must be proved by its ID
            # panel or its row audit. Never recreate a length-masked panel from
            # the mere presence of a nonempty question/answer.
            return metadata is None or allowed_ids is not None or bool(audits)
        masked_counts[axis] = sum(not available(row) for row in rows)
        answers = {}
        for row in rows:
            if isinstance(row.get(question), str) and row[question].strip() and isinstance(row.get(answer), str) and row[answer].strip():
                key = normalize_answer(row[question])
                answers.setdefault(key, set()).add(normalize_answer(row[answer]))
        ambiguous = {key for key, values in answers.items() if len(values) > 1}
        ambiguous_questions[axis] = sorted(ambiguous)
        selected = []
        for stage in sorted({row["stage"] for row in rows}):
            valid = [index for index, row in enumerate(rows) if row["stage"] == stage
                     and available(row)
                     and isinstance(row.get(question), str) and row[question].strip()
                     and isinstance(row.get(answer), str) and row[answer].strip()
                     and normalize_answer(row[question]) not in ambiguous]
            valid.sort(key=lambda index: object_sha256([seed, axis, rows[index]["sample_id"]]))
            selected.extend(valid[:per_stage])
        panels[axis] = selected
    return {"schema_version": "full-replay-panels/v1", "seed": seed,
            "max_per_axis_per_stage": per_stage,
            "sample_ids": {axis: [rows[index]["sample_id"] for index in indices]
                           for axis, indices in panels.items()},
            "counts": {axis: len(indices) for axis, indices in panels.items()},
            "ambiguous_questions_excluded_from_single_gold_evaluation": ambiguous_questions,
            "ambiguous_question_counts": {axis: len(values) for axis, values in ambiguous_questions.items()},
            "masked_by_manifest_availability": masked_counts,
            "availability_policy": "intersection of frozen manifest panel IDs and all supplied row availability audits; includes post-screen generation guards",
            "temporal_policy": "update/rephrase/persona: trained prefix; locality/multihop: fixed full panel"}


def panel_rows(rows, panels, axis, seen):
    ids = set(panels["sample_ids"][axis])
    prefix = rows[:seen] if axis in {"update", "rephrase", "persona"} else rows
    return [row for row in prefix if row["sample_id"] in ids]


def validate_generation_budget(metadata, max_length, max_new_tokens):
    sampling = metadata.get("sampling", {})
    if max_length < int(sampling.get("max_length", max_length)):
        raise ValueError("reader context budget is smaller than the frozen token audit")
    if max_new_tokens < int(sampling.get("max_answer_tokens", max_new_tokens)):
        raise ValueError("generation budget is smaller than the frozen answer/panel audit")


def truncate_log(path: Path, expected_bytes: int):
    """Discard successful but not checkpointed blocks before deterministic replay."""
    if expected_bytes == 0 and not path.exists():
        return
    if not path.is_file() or path.stat().st_size < expected_bytes:
        raise ValueError("committed training log is missing or truncated")
    with path.open("r+b") as stream:
        stream.truncate(expected_bytes)


def cpu_tree(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def write_checkpoint(output, payload, *, block_count, stage_checkpoint=False):
    """Publish the pointer only after a complete checkpoint file is on disk."""
    import torch
    directory = output / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"block_{block_count:06d}.pt"
    temporary = path.with_suffix(f".pt.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)
    pointer = output / "latest_checkpoint.json"
    previous = json.loads(pointer.read_text()) if pointer.exists() else None
    entry = {"file": str(path.relative_to(output)), "next_block": block_count,
             "stage_checkpoint": stage_checkpoint, "bytes": path.stat().st_size,
             "config_fingerprint": payload["config_fingerprint"]}
    atomic_json(pointer, entry)
    if previous and previous["file"] != entry["file"] and not previous.get("stage_checkpoint"):
        (output / previous["file"]).unlink(missing_ok=True)
    return entry


def read_checkpoint(output, fingerprint):
    import torch
    pointer = json.loads((output / "latest_checkpoint.json").read_text())
    path = (output / pointer["file"]).resolve()
    if not path.is_relative_to((output / "checkpoints").resolve()):
        raise ValueError("checkpoint path escapes its directory")
    if path.stat().st_size != pointer["bytes"]:
        raise ValueError("checkpoint file size changed")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if pointer["config_fingerprint"] != fingerprint or value["config_fingerprint"] != fingerprint:
        raise ValueError("checkpoint configuration or manifest does not match")
    if pointer["next_block"] != value["replay_state"]["next_block"]:
        raise ValueError("checkpoint cursor does not match its pointer")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("Qwen3-8B-Base"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-size", type=int, default=375)
    parser.add_argument("--memory-capacity", type=int, default=1000)
    parser.add_argument("--replay-ratio", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--panel-size-per-stage", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=48)
    parser.add_argument("--checkpoint-every-blocks", type=int, default=10)
    parser.add_argument("--max-blocks", type=int, default=0,
                        help="stop after this many total blocks; partial run remains resumable")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if min(args.epochs, args.batch_size, args.gradient_accumulation_steps,
           args.checkpoint_every_blocks, args.eval_batch_size) < 1 or args.max_blocks < 0:
        raise ValueError("epochs, batch sizes and checkpoint frequency must be positive")
    rows, metadata = load_stream(args.manifest)
    validate_generation_budget(metadata, args.max_length, args.max_new_tokens)
    blocks = make_blocks(rows, args.block_size)
    panels = freeze_panels(rows, args.panel_size_per_stage, args.seed + 9001, metadata)
    if not blocks:
        raise ValueError("empty full stream")
    runtime = {"resume", "gpu", "max_blocks", "plan_only", "checkpoint_every_blocks"}
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key not in runtime}
    config.update({"schema_version": "full-replay/v1",
        "manifest_sha256": file_sha256(args.manifest),
        "manifest_protocol_id": metadata.get("display_protocol_id", metadata.get("protocol_id")),
        "source_protocol_scope": metadata.get("protocol_scope"),
        "training_order": "original stage then numeric source row index",
        "temporal_unit": "already trained block; earlier blocks of current stage eligible",
        "optimizer_policy": "persistent AdamW across blocks; checkpoint includes optimizer moments",
        "closed_book_evaluation": True, "parameter_update_type": "LoRA",
        "answer_loss_only": True, "num_facts": len(rows), "num_blocks": len(blocks),
        "stage_counts": {str(stage): sum(row["stage"] == stage for row in rows)
                         for stage in sorted({row["stage"] for row in rows})},
        "evaluation_scope": "frozen sampled panels, not full-data accuracy",
        "panel_fingerprint": object_sha256(panels),
        "replay_admission_matched": True,
        "oracle_structure_access": args.method == "relation_oracle_replay",
        "replication_scope": "full eligible selected stream; block-scale extension of local 3K LoRA recipe",
        "source_sha256": {name: file_sha256(Path(__file__).with_name(name))
                          for name in ["run_full_replay.py", "common.py", "run_lora_family.py"]}})
    fingerprint = object_sha256(config)
    config["config_fingerprint"] = fingerprint
    plan = {"config": config, "evaluation_panels": panels,
            "stage_block_counts": {str(stage): sum(block.stage == stage for block in blocks)
                                    for stage in sorted({block.stage for block in blocks})}}
    if args.plan_only:
        atomic_json(args.output_dir / "plan.json", plan)
        print(json.dumps({"status": "planned", "facts": len(rows), "blocks": len(blocks),
                          "panels": panels["counts"]}), flush=True)
        return
    output = args.output_dir
    if args.resume:
        old_config = json.loads((output / "run_config.json").read_text())
        if old_config.get("config_fingerprint") != fingerprint:
            raise ValueError("resume configuration changed; use the original arguments and manifest")
        if (output / "completed.json").exists():
            print("Already complete; no training repeated.", flush=True)
            return
        if (output / "latest_checkpoint.json").exists():
            saved = read_checkpoint(output, fingerprint)
            state = ReplayState.from_state_dict(saved["replay_state"])
            if state.next_block > len(blocks) or state.seen != (blocks[state.next_block].start if state.next_block < len(blocks) else len(rows)):
                raise ValueError("checkpoint trained-prefix position is inconsistent")
            truncate_log(output / "training_log.jsonl", saved["training_log_bytes"])
            metrics = saved["metrics"]
            atomic_json(output / "metrics.json", metrics)
        elif not (output / "training_log.jsonl").exists():
            # A loader failure before the initial checkpoint has not trained
            # anything. Retrying that same frozen configuration is safe.
            state, metrics, saved = ReplayState(args.memory_capacity, args.seed), [], None
        else:
            raise ValueError("training log exists without a recoverable checkpoint")
    else:
        existing = list(output.iterdir()) if output.exists() else []
        if any(path.name != "plan.json" for path in existing):
            raise FileExistsError("output has existing run artifacts; use --resume")
        state, metrics, saved = ReplayState(args.memory_capacity, args.seed), [], None
        atomic_json(output / "run_config.json", config)
        atomic_json(output / "evaluation_panels.json", panels)
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from baselines.common import (AnswerOnlyDataset, append_jsonl, evaluate_axis,
                                  load_model, load_tokenizer, train_answer_only)
    from baselines.run_lora_family import lora_model
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training; --plan-only is CPU-only")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer(args.model_dir)
    model = lora_model(load_model(args.model_dir, device), args)
    optimizer = None
    if saved is not None:
        set_peft_model_state_dict(model, saved["adapter"])
        if saved["optimizer"] is not None:
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                lr=args.learning_rate, weight_decay=args.weight_decay, fused=True)
            optimizer.load_state_dict(saved["optimizer"])
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"], device)
    fresh_start = saved is None
    elapsed_prior = float(saved.get("elapsed_seconds", 0)) if saved else 0.0
    saved = None
    started = time.monotonic()
    latest_checkpoint_block = state.next_block
    def save(stage_end=False):
        nonlocal latest_checkpoint_block
        payload = {"config_fingerprint": fingerprint,
            "adapter": cpu_tree(get_peft_model_state_dict(model)),
            "optimizer": cpu_tree(optimizer.state_dict()) if optimizer else None,
            "replay_state": state.state_dict(), "metrics": metrics,
            "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device),
            "training_log_bytes": (output / "training_log.jsonl").stat().st_size
                                  if (output / "training_log.jsonl").exists() else 0,
            "elapsed_seconds": elapsed_prior + time.monotonic() - started}
        write_checkpoint(output, payload, block_count=state.next_block, stage_checkpoint=stage_end)
        latest_checkpoint_block = state.next_block
    # An initial checkpoint also makes interruption during the first block resumable.
    if fresh_start:
        save()
    for block in blocks[state.next_block:]:
        if args.max_blocks and block.index >= args.max_blocks:
            break
        replay, pairs = state.prepare(rows, block, args.method, args.replay_ratio)
        current = rows[block.start:block.end]
        training = current + [rows[index] for index in replay]
        # The shared dataset skips examples with no surviving answer token. Never
        # silently advance the full-stream cursor over such skipped examples.
        if len(AnswerOnlyDataset(training, tokenizer, args.max_length)) != len(training):
            raise ValueError("frozen training stream includes facts with no answer tokens after truncation")
        stats, optimizer = train_answer_only(model, tokenizer, training, device,
            seed=args.seed + block.index + 1, epochs=args.epochs, batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm, max_length=args.max_length, optimizer=optimizer)
        if not math.isfinite(stats.loss) or any(not torch.isfinite(parameter).all().item()
            for parameter in model.parameters() if parameter.requires_grad):
            raise RuntimeError("nonfinite training loss/adapter; cursor was not committed")
        state.commit(block)
        log = {"block": block.index + 1, "stage": block.stage, "start": block.start,
            "end": block.end, "current_count": len(current), "replay_count": len(replay),
            "current_sample_ids": [row["sample_id"] for row in current],
            "replay_sample_ids": [rows[index]["sample_id"] for index in replay],
            "relation_query_replay_pairs": pairs, "relation_matched_count": len(pairs),
            "reservoir_count": len(state.memory), "future_leak_count": 0,
            **asdict(stats)}
        append_jsonl(output / "training_log.jsonl", [log])
        if block.stage_end:
            checkpoint = f"tau{block.stage}"
            record = {"checkpoint": checkpoint, "current_stage": block.stage,
                      "trained_facts": state.seen, "num_facts": state.seen,
                      "evaluation_scope": config["evaluation_scope"]}
            predictions = []
            for axis in AXES:
                summary, axis_predictions = evaluate_axis(model, tokenizer,
                    panel_rows(rows, panels, axis, state.seen), axis, device,
                    batch_size=args.eval_batch_size, max_length=args.max_length,
                    max_new_tokens=args.max_new_tokens, checkpoint=checkpoint)
                record[axis] = summary
                predictions.extend(axis_predictions)
            directory = output / "evaluation" / checkpoint
            atomic_json(directory / "metrics.json", record)
            prediction_path = directory / "predictions.jsonl"
            temporary = prediction_path.with_suffix(".jsonl.tmp")
            temporary.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                         for row in predictions))
            temporary.replace(prediction_path)
            metrics.append(record)
            atomic_json(output / "metrics.json", metrics)
        if block.stage_end or state.next_block % args.checkpoint_every_blocks == 0:
            save(block.stage_end)
        atomic_json(output / "progress.json", {"status": "running", "next_block": state.next_block,
            "trained_facts": state.seen, "total_facts": len(rows), "stage": block.stage,
            "checkpoint_next_block": latest_checkpoint_block,
            "last_loss": stats.loss, "elapsed_seconds": elapsed_prior + time.monotonic() - started})
        print(json.dumps({"block": state.next_block, "stage": block.stage, "trained": state.seen,
                          "total": len(rows), "loss": stats.loss}), flush=True)
    if latest_checkpoint_block != state.next_block:
        save()
    if state.next_block < len(blocks):
        atomic_json(output / "progress.json", {"status": "partial_resumable",
            "next_block": state.next_block, "trained_facts": state.seen, "total_facts": len(rows),
            "checkpoint_next_block": state.next_block})
        print("Stopped at requested block limit; resume with --resume and a larger/zero --max-blocks.", flush=True)
        return
    model.save_pretrained(output / "adapters" / "final", safe_serialization=True)
    tokenizer.save_pretrained(output / "adapters" / "final")
    atomic_json(output / "completed.json", {"status": "complete", "method": args.method,
        "num_facts": state.seen, "num_blocks": state.next_block, "final_stage": blocks[-1].stage,
        "evaluation_scope": config["evaluation_scope"], "evaluated_stages": [m["current_stage"] for m in metrics],
        "panel_counts": {axis: metrics[-1][axis]["count"] for axis in AXES},
        "elapsed_seconds": elapsed_prior + time.monotonic() - started})
    atomic_json(output / "progress.json", {"status": "complete", "trained_facts": state.seen,
        "total_facts": len(rows), "next_block": state.next_block, "checkpoint_next_block": state.next_block})


if __name__ == "__main__":
    main()
