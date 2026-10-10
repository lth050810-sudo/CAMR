#!/usr/bin/env python3
"""Full-stream TiC-style and semantic LoRA replay with frozen Qwen features.

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
METHODS = ("lora_ft", "er_reservoir", "relation_oracle_replay", "tic_style_replay",
           "semantic_replay", "entity_masked_semantic_replay")
SEMANTIC_METHODS = {"semantic_replay", "entity_masked_semantic_replay"}


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

    def prepare(self, rows, block: Block, method: str, replay_ratio: float, embeddings=None):
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
        elif method in {"er_reservoir", "tic_style_replay"}:
            selected = er
        elif method == "relation_oracle_replay":
            selected, pairs = relation_selection(rows, range(block.start, block.end),
                self.memory, budget, self.seed + 4001 + block.index + 1)
        else:
            if embeddings is None:
                raise ValueError("semantic replay requires frozen Qwen prompt embeddings")
            selected = semantic_selection(range(block.start, block.end), self.memory, embeddings, budget)
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


def resolve_evaluation_panels(args, rows, metadata):
    """Use an explicitly frozen common panel without changing training RNG seed."""
    path = getattr(args, "evaluation_panels", None)
    if path is None:
        panels = freeze_panels(rows, args.panel_size_per_stage, args.seed + 9001, metadata)
        return panels, {"file": None, "sha256": None, "seed": args.seed + 9001}
    path = Path(path)
    payload = json.loads(path.read_text())
    panels = payload.get("evaluation_panels", payload)
    if (not isinstance(panels, dict) or panels.get("schema_version") != "full-replay-panels/v1"
            or type(panels.get("seed")) is not int
            or panels.get("max_per_axis_per_stage") != args.panel_size_per_stage):
        raise ValueError("explicit evaluation panel schema, seed or per-stage budget mismatch")
    expected = freeze_panels(rows, args.panel_size_per_stage, panels["seed"], metadata)
    if panels != expected:
        raise ValueError("explicit evaluation panels differ from the frozen source selection")
    return panels, {"file": str(path.resolve()), "sha256": file_sha256(path), "seed": panels["seed"]}


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


def semantic_selection(current, memory, embeddings, budget):
    """Original max-over-current cosine ranking, without copying the full cache."""
    import torch
    if budget <= 0 or not memory:
        return []
    current, memory = list(current), list(memory)
    if not current or min(current + memory) < 0 or max(current + memory) >= len(embeddings):
        raise ValueError("semantic embedding indices are invalid")
    queries = embeddings[torch.tensor(current, dtype=torch.long)].float()
    candidates = embeddings[torch.tensor(memory, dtype=torch.long)].float()
    if not torch.isfinite(queries).all() or not torch.isfinite(candidates).all():
        raise ValueError("nonfinite semantic embeddings")
    scores = (queries @ candidates.T).max(dim=0).values
    positions = torch.topk(scores, k=min(budget, len(memory)), largest=True, sorted=True).indices.tolist()
    return [memory[position] for position in positions]


def embedding_identity(args, rows):
    """Bind a shared cache to frozen base weights, prompt transform and row order."""
    import importlib.metadata
    identity = json.loads(args.model_identity.read_text())
    if Path(identity["local_path"]).resolve() != args.model_dir.resolve():
        raise ValueError("embedding checkpoint identity names a different model")
    model_config = json.loads((args.model_dir / "config.json").read_text())
    prompt_hash = hashlib.sha256()
    fields = ("sample_id", "update", "subject", "object", "ans")
    for row in rows:
        # Answers only affect the legacy entity-masking transform. The encoder
        # receives qa_prompt(update), never an answer completion or eval label.
        prompt_hash.update(json.dumps([row.get(key) for key in fields], ensure_ascii=False,
                                      separators=(",", ":")).encode() + b"\n")
    return {"schema_version": "frozen-qwen-prompt-embeddings/v1",
            "model_dir": str(args.model_dir.resolve()),
            "model_identity_sha256": file_sha256(args.model_identity),
            "model_content_fingerprint": identity["content_fingerprint"],
            "model_files": identity["files"],
            "manifest_sha256": file_sha256(args.manifest),
            "ordered_training_inputs_sha256": prompt_hash.hexdigest(),
            "row_count": len(rows), "dimension": int(model_config["hidden_size"]),
            "transform": "entity_masked" if args.method == "entity_masked_semantic_replay" else "plain",
            "prompt": "Question: <update>\nAnswer:", "answer_completion_included": False,
            "encoder": "frozen base Qwen3 final backbone states; attention-mask mean; L2 normalized",
            "max_length": args.max_length, "embedding_batch_size": args.embedding_batch_size,
            "embedding_shard_size": args.embedding_shard_size, "storage_dtype": "float32",
            "torch_version": importlib.metadata.version("torch"),
            "transformers_version": importlib.metadata.version("transformers"),
            "source_sha256": {name: file_sha256(Path(__file__).with_name(name))
                              for name in ("run_full_replay_extended.py", "common.py", "run_lora_family.py")}}


def verify_model_identity(model_dir, identity_path):
    """Validate actual weight/tokenizer bytes before building or reusing features."""
    identity = json.loads(Path(identity_path).read_text())
    if Path(identity["local_path"]).resolve() != Path(model_dir).resolve():
        raise ValueError("frozen Qwen model path differs from its identity")
    for name, expected in identity["files"].items():
        path = Path(model_dir) / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"frozen Qwen file does not match its audited identity: {name}")


def load_or_build_embeddings(cache_root, identity, rows, compute, *, shard_size=1024, read_only=False):
    """Atomically checkpoint exact FP32 feature shards, with a shared writer lock.

    A stopped extraction reuses only committed, hash-validated shards. A reader
    never substitutes vectors from another checkpoint, transform or row order.
    """
    import fcntl
    import torch
    if shard_size <= 0 or len(rows) != identity["row_count"]:
        raise ValueError("invalid semantic cache row/shard count")
    fingerprint = object_sha256(identity)
    directory = Path(cache_root) / fingerprint
    if read_only and not directory.is_dir():
        raise ValueError("committed semantic cache directory is missing")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+") as lock:
        waiting = time.monotonic()
        while True:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - waiting >= 30:
                    print(json.dumps({"semantic_cache": str(directory), "status": "waiting_for_writer"}), flush=True)
                    waiting = time.monotonic()
                time.sleep(1)
        identity_path = directory / "identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError("semantic cache identity changed")
        else:
            if read_only:
                raise ValueError("committed semantic cache identity is missing")
            atomic_json(identity_path, identity)
        matrix = torch.empty((len(rows), identity["dimension"]), dtype=torch.float32)
        shards = []
        for start in range(0, len(rows), shard_size):
            end = min(start + shard_size, len(rows))
            path = directory / f"rows_{start:09d}_{end:09d}.pt"
            meta_path = path.with_suffix(".json")
            ids = [row["sample_id"] for row in rows[start:end]]
            expected = {"fingerprint": fingerprint, "start": start, "end": end,
                        "sample_ids_sha256": object_sha256(ids)}
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                if any(meta.get(key) != value for key, value in expected.items()):
                    raise ValueError("semantic cache shard row identity mismatch")
                if not path.is_file() or path.stat().st_size != meta["bytes"] or file_sha256(path) != meta["sha256"]:
                    raise ValueError("semantic cache shard is missing or corrupt")
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if payload.get("sample_ids") != ids or any(payload.get(key) != expected[key]
                                                          for key in ("fingerprint", "start", "end")):
                    raise ValueError("semantic cache tensor provenance mismatch")
                vectors = payload["embeddings"]
            else:
                if read_only:
                    raise ValueError("completed semantic run has a missing cache shard")
                vectors = compute(rows[start:end]).detach().cpu().float().clone()
                validate_embedding_tensor(vectors, end - start, identity["dimension"])
                payload = {**expected, "sample_ids": ids, "embeddings": vectors}
                temporary = path.with_suffix(f".pt.{os.getpid()}.tmp")
                torch.save(payload, temporary)
                with temporary.open("rb") as stream:
                    os.fsync(stream.fileno())
                temporary.replace(path)
                meta = {**expected, "file": path.name, "bytes": path.stat().st_size,
                        "sha256": file_sha256(path)}
                atomic_json(meta_path, meta)
            validate_embedding_tensor(vectors, end - start, identity["dimension"])
            matrix[start:end].copy_(vectors)
            shards.append(meta)
            if not read_only:
                atomic_json(directory / "progress.json", {"status": "building", "rows": end,
                            "total_rows": len(rows), "fingerprint": fingerprint})
                print(json.dumps({"semantic_cache": str(directory), "rows": end,
                                  "total_rows": len(rows)}), flush=True)
        complete = {"status": "complete", "fingerprint": fingerprint,
                    "rows": len(rows), "dimension": identity["dimension"], "shards": shards}
        if (directory / "complete.json").exists() and json.loads((directory / "complete.json").read_text()) != complete:
            raise ValueError("semantic cache completion manifest changed")
        if read_only and not (directory / "complete.json").is_file():
            raise ValueError("semantic cache has no completion commit")
        if not read_only:
            atomic_json(directory / "complete.json", complete)
            atomic_json(directory / "progress.json", {"status": "complete", "rows": len(rows),
                        "total_rows": len(rows), "fingerprint": fingerprint})
        return matrix, {"directory": str(directory.resolve()), "fingerprint": fingerprint,
                        "complete_sha256": file_sha256(directory / "complete.json"),
                        "rows": len(rows), "dimension": identity["dimension"], "storage_dtype": "float32"}


def validate_embedding_tensor(vectors, count, dimension):
    import torch
    if (not isinstance(vectors, torch.Tensor) or vectors.dtype != torch.float32
            or vectors.shape != (count, dimension) or not torch.isfinite(vectors).all()
            or not torch.allclose(vectors.norm(dim=1), torch.ones(count), atol=1e-4, rtol=1e-4)):
        raise ValueError("invalid frozen Qwen embedding tensor")


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
    parser.add_argument("--evaluation-panels", type=Path,
                        help="Explicit shared frozen panel or plan payload; seed is read from this file")
    parser.add_argument("--eval-batch-size", type=int, default=48)
    parser.add_argument("--checkpoint-every-blocks", type=int, default=10)
    parser.add_argument("--max-blocks", type=int, default=0,
                        help="stop after this many total blocks; partial run remains resumable")
    parser.add_argument("--embedding-cache-dir", type=Path,
                        default=Path("/mnt/yy2607004_hs/engram_full_runs_20260920/_qwen_prompt_embeddings"))
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--embedding-shard-size", type=int, default=1024)
    parser.add_argument("--model-identity", type=Path,
                        default=Path("/root/engram/runs/protocols/qwen3_8b_base_checkpoint_identity.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if min(args.epochs, args.batch_size, args.gradient_accumulation_steps,
           args.checkpoint_every_blocks, args.eval_batch_size,
           args.embedding_batch_size, args.embedding_shard_size) < 1 or args.max_blocks < 0:
        raise ValueError("epochs, batch sizes and checkpoint frequency must be positive")
    rows, metadata = load_stream(args.manifest)
    validate_generation_budget(metadata, args.max_length, args.max_new_tokens)
    blocks = make_blocks(rows, args.block_size)
    panels, panel_source = resolve_evaluation_panels(args, rows, metadata)
    if not blocks:
        raise ValueError("empty full stream")
    semantic_identity = embedding_identity(args, rows) if args.method in SEMANTIC_METHODS else None
    runtime = {"resume", "gpu", "max_blocks", "plan_only", "checkpoint_every_blocks"}
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key not in runtime}
    config.update({"schema_version": "full-replay-extended/v1",
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
        "evaluation_panel_source": panel_source,
        "replay_admission_matched": True,
        "oracle_structure_access": args.method == "relation_oracle_replay",
        "replication_scope": "full selected stream; 375-fact block extension of local 3K recipe",
        "replay_selection_policy": (
            "max cosine over current block; Top-K unique reservoir facts; frozen base-Qwen prompt states"
            if args.method in SEMANTIC_METHODS else
            "current facts only" if args.method == "lora_ft" else
            "relation-structure oracle with uniform fallback" if args.method == "relation_oracle_replay" else
            "uniform ER reservoir"),
        "stage_lr_schedule": "10% rewarm + cosine decay to 10% peak EACH BLOCK; persistent Adam moments"
            if args.method == "tic_style_replay" else "constant",
        "schedule_scope": "block-scale TiC-style reference; not original stage schedule or a TiC-LM reproduction"
            if args.method == "tic_style_replay" else "constant learning rate",
        "semantic_embedding_identity": semantic_identity,
        "semantic_embedding_fingerprint": object_sha256(semantic_identity) if semantic_identity else None,
        "semantic_cache_scope": "offline frozen base-Qwen prompt-only features; selection indexes only trained reservoir",
        "source_sha256": {name: file_sha256(Path(__file__).with_name(name))
                          for name in ["run_full_replay_extended.py", "run_full_replay.py", "common.py", "run_lora_family.py"]}})
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
    base_model = load_model(args.model_dir, device)
    embeddings = None
    if semantic_identity is not None:
        from baselines.common import dense_prompt_embeddings
        from baselines.run_lora_family import entity_masked_rows
        verify_model_identity(args.model_dir, args.model_identity)
        embedding_rows = entity_masked_rows(rows) if args.method == "entity_masked_semantic_replay" else rows
        embeddings, cache_record = load_or_build_embeddings(
            args.embedding_cache_dir, semantic_identity, embedding_rows,
            lambda batch: dense_prompt_embeddings(base_model, tokenizer, batch, device,
                batch_size=args.embedding_batch_size, max_length=args.max_length),
            shard_size=args.embedding_shard_size)
        atomic_json(output / "embedding_cache.json", cache_record)
    model = lora_model(base_model, args)
    optimizer = None
    if saved is not None:
        if saved.get("semantic_embedding_fingerprint") != config["semantic_embedding_fingerprint"]:
            raise ValueError("checkpoint frozen semantic embeddings changed")
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
            "semantic_embedding_fingerprint": config["semantic_embedding_fingerprint"],
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
        replay, pairs = state.prepare(rows, block, args.method, args.replay_ratio, embeddings)
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
            max_grad_norm=args.max_grad_norm, max_length=args.max_length, optimizer=optimizer,
            stage_lr_schedule=args.method == "tic_style_replay")
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
            "reservoir_state_sha256": object_sha256(state.state_dict()),
            "learning_rate_schedule": "block_rewarm_cosine" if args.method == "tic_style_replay" else "constant",
            "final_optimizer_learning_rates": [group["lr"] for group in optimizer.param_groups],
            "semantic_embedding_fingerprint": config["semantic_embedding_fingerprint"],
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


def validate_completed(job, output=None):
    """Supervisor callable: verify raw/selected traversal, replay and scored outputs.

    ``job`` follows the baseline queue schema (command, cwd, output, manifest).
    Return a small certificate only after all checks pass; raise on any failure.
    """
    from baselines.run_baseline_queue import job_path, command_option
    from baselines.common import answer_scores
    output = job_path(job, output or job["output"])
    sha256 = file_sha256
    directory = output.parent
    manifest = job_path(job, job.get("manifest") or command_option(job, "--manifest"))
    rows, metadata = load_stream(manifest)
    if metadata.get("protocol_scope") == "raw_all_source_records" and len(rows) != len(metadata["records"]):
        raise ValueError("raw-all replay selection omits original source records")
    if sorted({row["stage"] for row in rows}) != list(range(1, 9)):
        raise ValueError("full replay must cover all eight original stages")
    config = json.loads((directory / "run_config.json").read_text())
    done = json.loads(output.read_text())
    if config.get("manifest_sha256") != sha256(manifest) or config.get("num_facts") != len(rows):
        raise ValueError("full replay manifest hash/count mismatch")
    if config.get("config_fingerprint") != object_sha256({k: v for k, v in config.items() if k != "config_fingerprint"}):
        raise ValueError("full replay config fingerprint mismatch")
    # Bind the full recipe to the actual queue argv rather than accepting any
    # complete artifact that happens to occupy the configured output directory.
    command = [str(value).replace("{gpu}", "0") for value in job["command"]]
    try:
        start = next(i for i, value in enumerate(command)
                     if value == "baselines.run_full_replay_extended" or Path(value).name == "run_full_replay_extended.py") + 1
    except StopIteration as error:
        raise ValueError("full replay job must invoke its runner directly") from error
    arguments = parse_args(command[start:])
    runtime = {"resume", "gpu", "max_blocks", "plan_only", "checkpoint_every_blocks"}
    for key, value in vars(arguments).items():
        if key in runtime:
            continue
        expected = str(job_path(job, value)) if isinstance(value, Path) else value
        if config.get(key) != expected:
            raise ValueError(f"full replay recipe mismatch: {key}")
    blocks = make_blocks(rows, config["block_size"])
    if (done.get("status") != "complete" or done.get("num_facts") != len(rows)
            or done.get("num_blocks") != len(blocks) or done.get("final_stage") != 8
            or done.get("method") != config["method"]):
        raise ValueError("full replay completion counts/status mismatch")
    indices = {row["sample_id"]: index for index, row in enumerate(rows)}
    expected_state = ReplayState(config["memory_capacity"], config["seed"])
    total_optimizer_steps = 0
    embeddings = None
    if config["method"] in SEMANTIC_METHODS:
        cache_record = json.loads((directory / "embedding_cache.json").read_text())
        identity = config["semantic_embedding_identity"]
        if object_sha256(identity) != config["semantic_embedding_fingerprint"]:
            raise ValueError("semantic embedding identity fingerprint changed")
        cache_dir = Path(cache_record["directory"])
        if (cache_dir.name != config["semantic_embedding_fingerprint"]
                or file_sha256(cache_dir / "complete.json") != cache_record["complete_sha256"]):
            raise ValueError("semantic cache completion fingerprint changed")
        def no_recompute(_rows):
            raise ValueError("completed semantic run has a missing cache shard")
        embeddings, checked_cache = load_or_build_embeddings(cache_dir.parent, identity, rows, no_recompute,
            shard_size=config["embedding_shard_size"], read_only=True)
        if checked_cache != cache_record:
            raise ValueError("semantic cache audit record mismatch")
    seen_blocks = 0
    with (directory / "training_log.jsonl").open() as stream:
        for seen_blocks, line in enumerate(stream, 1):
            if seen_blocks > len(blocks):
                raise ValueError("full replay training log has duplicate/extra blocks")
            log = json.loads(line)
            block = blocks[seen_blocks - 1]
            if (log.get("block"), log.get("stage"), log.get("start"), log.get("end")) != (
                    seen_blocks, block.stage, block.start, block.end):
                raise ValueError("full replay log is not a continuous source prefix")
            current_ids = [row["sample_id"] for row in rows[block.start:block.end]]
            if log.get("current_sample_ids") != current_ids or log.get("current_count") != len(current_ids):
                raise ValueError("full replay current-block identities mismatch")
            replay = log.get("replay_sample_ids", [])
            expected_count = (0 if config["method"] == "lora_ft" else
                min(round(len(current_ids) * config["replay_ratio"]), block.start, config["memory_capacity"]))
            if len(replay) != expected_count or log.get("replay_count") != len(replay) or len(set(replay)) != len(replay):
                raise ValueError("full replay budget/duplicate mismatch")
            if any(sid not in indices or indices[sid] >= block.start for sid in replay):
                raise ValueError("full replay contains future/untrained facts")
            expected_replay, expected_pairs = expected_state.prepare(rows, block, config["method"],
                config["replay_ratio"], embeddings)
            if replay != [rows[index]["sample_id"] for index in expected_replay]:
                raise ValueError("full replay selector does not match the frozen method and ER admission")
            if log.get("relation_query_replay_pairs") != expected_pairs:
                raise ValueError("relation oracle replay query pairs changed")
            expected_state.commit(block)
            if log.get("reservoir_state_sha256") != object_sha256(expected_state.state_dict()):
                raise ValueError("full replay reservoir RNG/admission does not match ER")
            expected_schedule = "block_rewarm_cosine" if config["method"] == "tic_style_replay" else "constant"
            if (log.get("learning_rate_schedule") != expected_schedule
                    or log.get("semantic_embedding_fingerprint") != config["semantic_embedding_fingerprint"]):
                raise ValueError("full replay learning/embedding schedule identity mismatch")
            if log.get("future_leak_count") != 0 or log.get("reservoir_count") != min(config["memory_capacity"], block.end):
                raise ValueError("full replay reservoir/temporal audit mismatch")
            if (not isinstance(log.get("loss"), (int, float)) or not math.isfinite(log["loss"])
                    or any(log.get(key, 0) <= 0 for key in ["optimizer_steps", "sequence_tokens", "answer_tokens"])):
                raise ValueError("full replay training statistics incomplete/nonfinite")
            expected_steps = config["epochs"] * math.ceil(
                math.ceil((len(current_ids) + len(replay)) / config["batch_size"])
                / config["gradient_accumulation_steps"])
            if log["optimizer_steps"] != expected_steps:
                raise ValueError("full replay optimizer step budget mismatch")
            total_optimizer_steps += expected_steps
            from baselines.common import stage_lr_factor
            expected_lr = config["learning_rate"] * (stage_lr_factor(expected_steps - 1, expected_steps)
                if config["method"] == "tic_style_replay" else 1.0)
            rates = log.get("final_optimizer_learning_rates", [])
            if not rates or any(not math.isclose(rate, expected_lr, rel_tol=1e-12, abs_tol=1e-15) for rate in rates):
                raise ValueError("full replay block learning-rate schedule mismatch")
    if seen_blocks != len(blocks):
        raise ValueError("full replay log ends before the full selected stream")
    panels = json.loads((directory / "evaluation_panels.json").read_text())
    if arguments.evaluation_panels is not None:
        arguments.evaluation_panels = job_path(job, arguments.evaluation_panels)
    expected_panels, expected_panel_source = resolve_evaluation_panels(arguments, rows, metadata)
    if (object_sha256(panels) != config["panel_fingerprint"] or panels != expected_panels
            or config.get("evaluation_panel_source") != expected_panel_source):
        raise ValueError("full replay panel fingerprint or frozen selection mismatch")
    if job.get("common_panel_fingerprint") and config["panel_fingerprint"] != job["common_panel_fingerprint"]:
        raise ValueError("full replay does not use the supervisor's common panels")
    for axis, identifiers in panels["sample_ids"].items():
        if axis not in AXES or len(identifiers) != len(set(identifiers)) or any(sid not in indices for sid in identifiers):
            raise ValueError("full replay panel identities invalid")
        source_axis = {"persona": "personas", "multihop": "mhop"}.get(axis, axis)
        allowed = set(metadata["panels"][source_axis]["sample_ids"])
        if any(sid not in allowed or any(audit is not None and audit.get("available") is not True
                   for audit in [rows[indices[sid]].get("panel_audit", {}).get(source_axis),
                                 rows[indices[sid]].get("token_length_audit", {}).get("panels", {}).get(source_axis)])
               for sid in identifiers):
            raise ValueError("full replay panel resurrects masked source probes")
        if any(sum(rows[indices[sid]]["stage"] == stage for sid in identifiers) > config["panel_size_per_stage"]
               for stage in range(1, 9)):
            raise ValueError("full replay panel exceeds its frozen per-stage sample budget")
    metrics = json.loads((directory / "metrics.json").read_text())
    if [row.get("current_stage") for row in metrics] != list(range(1, 9)) or done.get("evaluated_stages") != list(range(1, 9)):
        raise ValueError("full replay requires eight evaluated stages")
    for stage, metric in enumerate(metrics, 1):
        trained = sum(row["stage"] <= stage for row in rows)
        if metric.get("trained_facts") != trained or metric.get("num_facts") != trained or metric.get("checkpoint") != f"tau{stage}":
            raise ValueError("full replay evaluation trained-prefix mismatch")
        if metric != json.loads((directory / "evaluation" / f"tau{stage}" / "metrics.json").read_text()):
            raise ValueError("full replay stage metric copies disagree")
        expected_ids = {axis: {sid for sid in panels["sample_ids"][axis]
                              if axis in {"locality", "multihop"} or rows[indices[sid]]["stage"] <= stage}
                        for axis in AXES}
        for axis in AXES:
            value, count = metric[axis], len(expected_ids[axis])
            if count < 1 or value.get("count") != count:
                raise ValueError("full replay sampled-panel denominator mismatch")
            for metric_name in ["match", "exact"]:
                accuracy, correct = value[metric_name + "_accuracy"], value[metric_name + "_count"]
                if not math.isfinite(accuracy) or not 0 <= correct <= count or not math.isclose(accuracy, correct / count, abs_tol=1e-12):
                    raise ValueError("full replay metric invalid/nonfinite")
        observed = {axis: set() for axis in AXES}
        recomputed = {axis: {"exact": 0, "match": 0} for axis in AXES}
        with (directory / "evaluation" / f"tau{stage}" / "predictions.jsonl").open() as predictions:
            for line in predictions:
                value = json.loads(line)
                axis, sid = value["axis"], value["sample_id"]
                if axis not in observed or sid in observed[axis] or sid not in expected_ids[axis] or value["checkpoint"] != f"tau{stage}":
                    raise ValueError("full replay prediction coverage mismatch")
                observed[axis].add(sid)
                source = rows[indices[sid]]
                question_key, answer_key = AXES[axis]
                if (value.get("question") != source[question_key] or value.get("gold") != source[answer_key]
                        or value.get("fact_stage") != source["stage"]):
                    raise ValueError("full replay prediction source QA/stage mismatch")
                scores = answer_scores(value["prediction"], source[answer_key])
                for score in ("exact", "match"):
                    if value.get(score) != scores[score]:
                        raise ValueError("full replay stored prediction score is incorrect")
                    recomputed[axis][score] += int(scores[score])
        for axis in AXES:
            if any(metric[axis][score + "_count"] != recomputed[axis][score] for score in ("exact", "match")):
                raise ValueError("full replay metrics do not equal recomputed prediction scores")
        if observed != expected_ids:
            raise ValueError("full replay missing sampled predictions")
    if done.get("panel_counts") != {axis: metrics[-1][axis]["count"] for axis in AXES}:
        raise ValueError("full replay final denominators mismatch")
    pointer = json.loads((directory / "latest_checkpoint.json").read_text())
    checkpoint = (directory / pointer["file"]).resolve()
    if (not checkpoint.is_relative_to((directory / "checkpoints").resolve())
            or pointer.get("next_block") != len(blocks) or checkpoint.stat().st_size != pointer["bytes"]
            or pointer.get("config_fingerprint") != config["config_fingerprint"]):
        raise ValueError("full replay final checkpoint missing or inconsistent")
    if not all((directory / "adapters" / "final" / name).is_file()
               for name in ["adapter_model.safetensors", "adapter_config.json"]):
        raise ValueError("full replay final adapter is missing")
    checkpoint_payload = read_checkpoint(directory, config["config_fingerprint"])
    if (checkpoint_payload.get("semantic_embedding_fingerprint") != config["semantic_embedding_fingerprint"]
            or ReplayState.from_state_dict(checkpoint_payload["replay_state"]).state_dict() != expected_state.state_dict()
            or checkpoint_payload["metrics"] != metrics
            or checkpoint_payload["training_log_bytes"] != (directory / "training_log.jsonl").stat().st_size
            or not checkpoint_payload.get("adapter") or not checkpoint_payload.get("optimizer")):
        raise ValueError("full replay checkpoint lacks complete final model/optimizer/replay state")
    import torch
    from safetensors.torch import load_file
    final_adapter = load_file(str(directory / "adapters" / "final" / "adapter_model.safetensors"))
    adapter = checkpoint_payload["adapter"]
    if final_adapter.keys() != adapter.keys() or any(not torch.equal(final_adapter[key], tensor)
            or not torch.isfinite(tensor).all() for key, tensor in adapter.items()):
        raise ValueError("final adapter differs from the committed checkpoint or is nonfinite")
    states = checkpoint_payload["optimizer"].get("state", {})
    if not states:
        raise ValueError("final checkpoint has no Adam moment state")
    for state in states.values():
        if (float(state.get("step", -1)) != total_optimizer_steps
                or any(not torch.isfinite(state[name]).all() for name in ("exp_avg", "exp_avg_sq"))):
            raise ValueError("final checkpoint optimizer steps/moments are incomplete or nonfinite")
    return {"sha256": sha256(output), "manifest_sha256": config["manifest_sha256"],
            "training_facts": len(rows), "training_blocks": len(blocks), "evaluated_stages": 8,
            "panel_counts": done["panel_counts"], "future_leaks": 0,
            "prediction_scores_recomputed": True, "selector_and_reservoir_recomputed": True}



if __name__ == "__main__":
    main()
