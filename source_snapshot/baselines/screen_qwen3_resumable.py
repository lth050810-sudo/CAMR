"""Identical greedy candidate screen with committed 64-example batch progress."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

from baselines.common import (answer_scores, atomic_json, environment_metadata,
                              generate_answers, load_model, load_tokenizer, sha256_file)
from baselines.screen_qwen3_base import parse_args


def validate_prefix(records, predictions):
    if len(predictions) > len(records):
        raise ValueError("screen prefix exceeds shard")
    counts = {"known": 0, "unknown": 0}
    for row, prediction in zip(records, predictions):
        if any(prediction[k] != row[s] for k, s in (("sample_id", "sample_id"), ("stage", "stage"),
                                                   ("question", "update"), ("gold", "ans"))):
            raise ValueError("screen prefix source mismatch")
        scores = answer_scores(prediction["prediction"], row["ans"])
        status = "known" if scores["match"] else "unknown"
        if prediction["screen_status"] != status or prediction["base_unknown"] != (status == "unknown"):
            raise ValueError("screen prefix classification mismatch")
        if any(prediction[k] != v for k, v in scores.items()):
            raise ValueError("screen prefix score mismatch")
        counts[status] += 1
    return counts


def recover_prefix(partial, progress, identity, records, batch_size):
    if not progress.exists():
        if partial.exists() and partial.stat().st_size:
            raise ValueError("partial screen has no provenance/commit marker")
        return 0, {"known": 0, "unknown": 0}, 0.0
    state = json.loads(progress.read_text())
    if state["identity"] != identity:
        raise ValueError("screen resume identity changed")
    count, offset = state["committed_count"], state["committed_bytes"]
    if count % batch_size and count != len(records):
        raise ValueError("screen checkpoint does not end at the original batch boundary")
    if not partial.exists() or partial.stat().st_size < offset:
        raise ValueError("committed screen prefix missing/truncated")
    with partial.open("r+b") as f:
        f.truncate(offset)
    predictions = [json.loads(line) for line in partial.read_text().splitlines()]
    if len(predictions) != count:
        raise ValueError("screen committed row count mismatch")
    counts = validate_prefix(records, predictions)
    if counts != state["status_counts"]:
        raise ValueError("screen committed status count mismatch")
    return count, counts, state.get("elapsed_seconds", 0.0)


def main():
    import torch
    args = parse_args()
    records = json.loads(args.manifest.read_text())["records"]
    if not records or args.batch_size < 1:
        raise ValueError("empty shard or invalid batch size")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    identity = {"manifest_sha256": sha256_file(args.manifest), "model_dir": str(args.model_dir.resolve()),
                "model_config_sha256": sha256_file(args.model_dir / "config.json"),
                "weight_index_sha256": sha256_file(args.model_dir / "model.safetensors.index.json"),
                "screen_protocol_id": "qwen3-base-greedy-update-v1", "batch_size": args.batch_size,
                "max_length": args.max_length, "max_new_tokens": args.max_new_tokens,
                "source_sha256": sha256_file(Path(__file__)),
                "common_source_sha256": sha256_file(Path(__file__).with_name("common.py"))}
    progress = args.output.with_suffix(".progress.json")
    partial = args.output.with_suffix(".jsonl.partial")
    if args.output.exists():
        metadata = json.loads(args.output.with_suffix(".metadata.json").read_text())
        if metadata.get("resume_identity") != identity:
            raise ValueError("existing screen output has different identity")
        predictions = [json.loads(line) for line in args.output.read_text().splitlines()]
        if len(predictions) != len(records):
            raise ValueError("existing screen output incomplete")
        validate_prefix(records, predictions)
        print("Verified completed shard; no generation repeated.", flush=True)
        return
    count, counts, elapsed = recover_prefix(partial, progress, identity, records, args.batch_size)
    partial.touch(exist_ok=True)
    started = time.monotonic()
    def commit():
        atomic_json(progress, {"identity": identity, "committed_count": count,
                    "committed_bytes": partial.stat().st_size, "status_counts": counts,
                    "candidate_count": len(records), "elapsed_seconds": elapsed + time.monotonic() - started})
    commit()
    if count < len(records):
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        tokenizer = load_tokenizer(args.model_dir)
        model = load_model(args.model_dir, device)
        for start in range(count, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            predictions = generate_answers(model, tokenizer, [r["update"] for r in batch], device,
                batch_size=args.batch_size, max_length=args.max_length, max_new_tokens=args.max_new_tokens)
            if len(predictions) != len(batch):
                raise ValueError("generation count mismatch")
            with partial.open("a") as f:
                for row, prediction in zip(batch, predictions):
                    scores = answer_scores(prediction, row["ans"])
                    status = "known" if scores["match"] else "unknown"
                    counts[status] += 1
                    f.write(json.dumps({"sample_id": row["sample_id"], "stage": row["stage"],
                        "screen_status": status, "base_unknown": status == "unknown",
                        "question": row["update"], "gold": row["ans"], "prediction": prediction,
                        **scores}, ensure_ascii=False, allow_nan=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            count += len(batch)
            commit()
            if count % 1024 == 0 or count == len(records):
                print(json.dumps({"screened": count, "total": len(records), "status_counts": counts}), flush=True)
    # Commit metadata first. A crash before promoting the data can reconstruct
    # the same metadata from the full committed partial without inference.
    atomic_json(args.output.with_suffix(".metadata.json"), {
        "screen_protocol_id": identity["screen_protocol_id"], "model_dir": identity["model_dir"],
        "manifest": str(args.manifest.resolve()), "manifest_sha256": identity["manifest_sha256"],
        "candidate_count": len(records), "status_counts": counts, "batch_size": args.batch_size,
        "max_length": args.max_length, "max_new_tokens": args.max_new_tokens,
        "prompt_template": "Question: <question>\\nAnswer:",
        "elapsed_seconds": elapsed + time.monotonic() - started,
        "resume_identity": identity, **environment_metadata()})
    partial.replace(args.output)


if __name__ == "__main__":
    main()
