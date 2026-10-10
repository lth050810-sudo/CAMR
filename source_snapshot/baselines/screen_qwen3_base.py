#!/usr/bin/env python3
"""Screen WikiBigEdit candidates for facts already answered by Qwen3 base."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from baselines.common import (
    answer_scores,
    atomic_json,
    environment_metadata,
    generate_answers,
    load_model,
    load_tokenizer,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("Qwen3-8B-Base"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("protocol manifest has no candidate records")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    tokenizer = load_tokenizer(args.model_dir)
    model = load_model(args.model_dir, device)
    started = time.time()
    predictions = generate_answers(
        model,
        tokenizer,
        [str(record["update"]) for record in records],
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )
    counts = {"known": 0, "unknown": 0}
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record, prediction in zip(records, predictions):
            scores = answer_scores(prediction, str(record["ans"]))
            status = "known" if scores["match"] else "unknown"
            counts[status] += 1
            row = {
                "sample_id": record["sample_id"],
                "stage": int(record["stage"]),
                "screen_status": status,
                "base_unknown": status == "unknown",
                "question": record["update"],
                "gold": record["ans"],
                "prediction": prediction,
                **scores,
            }
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(args.output)
    atomic_json(
        args.output.with_suffix(".metadata.json"),
        {
            "screen_protocol_id": "qwen3-base-greedy-update-v1",
            "model_dir": str(args.model_dir.resolve()),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "candidate_count": len(records),
            "status_counts": counts,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            "prompt_template": "Question: <question>\\nAnswer:",
            "elapsed_seconds": time.time() - started,
            **environment_metadata(),
        },
    )


if __name__ == "__main__":
    main()

