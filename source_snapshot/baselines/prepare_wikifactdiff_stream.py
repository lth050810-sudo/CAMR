#!/usr/bin/env python3
"""Freeze a reproducible 12-stage WikiFactDiff candidate stream."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


STAGES = 12
PER_STAGE = 3000
REPLACE_PER_STAGE = 300
SOURCE_REVISION = "bb17ffbff7b2d28e4cd12e251af3db50d7fa18ea"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def candidate(index: int, row: dict, seed: int) -> dict | None:
    subject = row.get("subject") or {}
    relation = row.get("relation") or {}
    prompt = row.get("update_prompt")
    if not subject.get("id") or not relation.get("id") or not isinstance(prompt, str) or not prompt.strip():
        return None
    objects = row.get("objects") or []
    new = [obj for obj in objects if obj.get("decision") == "new" and isinstance(obj.get("label"), str) and obj["label"].strip()]
    obsolete = [obj for obj in objects if obj.get("decision") == "obsolete"]
    if len(new) != 1:
        return None
    if row.get("is_replace"):
        if len(obsolete) != 1 or not isinstance(obsolete[0].get("label"), str) or not obsolete[0]["label"].strip():
            return None
        if normalized(new[0]["label"]) == normalized(obsolete[0]["label"]):
            return None
        kind = "replace"
    else:
        if obsolete:
            return None
        kind = "new"
    return {
        "source_row": index,
        "sample_id": f"wikifactdiff:legacy:{index:06d}",
        "kind": kind,
        "relation_id": relation["id"],
        "prompt_key": normalized(prompt),
        "rank": hashlib.sha256(f"{seed}:{index}".encode()).hexdigest(),
    }


def assign_stages(selected: list[dict], per_stage: int) -> None:
    grouped = defaultdict(list)
    for item in selected:
        grouped[item["relation_id"]].append(item)
    counts = [0] * STAGES
    relation_counts = [Counter() for _ in range(STAGES)]
    for relation_id, items in sorted(grouped.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        for item in sorted(items, key=lambda record: record["rank"]):
            stage = min(
                (i for i in range(STAGES) if counts[i] < per_stage),
                key=lambda i: (relation_counts[i][relation_id], counts[i], i),
            )
            item["stage"] = stage + 1
            counts[stage] += 1
            relation_counts[stage][relation_id] += 1
    if counts != [per_stage] * STAGES:
        raise RuntimeError(f"Unexpected stage sizes: {counts}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("WikiFactDiff/20210104-20230227_legacy/train.parquet"))
    parser.add_argument("--output", type=Path, default=Path("WikiFactDiff/stream_36k_12x3k_seed42"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = pq.ParquetFile(args.source)
    candidates = []
    index = 0
    fields = ["subject", "relation", "objects", "is_replace", "update_prompt"]
    for batch in source.iter_batches(columns=fields, batch_size=4096):
        for row in batch.to_pylist():
            item = candidate(index, row, args.seed)
            if item is not None:
                candidates.append(item)
            index += 1

    unique = []
    prompts = set()
    for item in sorted(candidates, key=lambda record: record["rank"]):
        if item["prompt_key"] not in prompts:
            prompts.add(item["prompt_key"])
            unique.append(item)
    replace = [item for item in unique if item["kind"] == "replace"]
    new = [item for item in unique if item["kind"] == "new"]
    replace_needed = STAGES * REPLACE_PER_STAGE
    new_needed = STAGES * (PER_STAGE - REPLACE_PER_STAGE)
    if len(replace) < replace_needed or len(new) < new_needed:
        raise RuntimeError(f"Not enough clean examples: replace={len(replace)}, new={len(new)}")
    selected = replace[:replace_needed] + new[:new_needed]
    assign_stages(selected[:replace_needed], REPLACE_PER_STAGE)
    assign_stages(selected[replace_needed:], PER_STAGE - REPLACE_PER_STAGE)
    selected.sort(key=lambda record: (record["stage"], record["rank"]))
    by_index = {item["source_row"]: item for item in selected}
    rank_by_id = {item["sample_id"]: item["rank"] for item in selected}
    manifest = [{key: item[key] for key in ("sample_id", "source_row", "stage", "kind", "relation_id")} for item in selected]
    write_jsonl(args.output / "manifest.jsonl", manifest)

    train = [[] for _ in range(STAGES)]
    evaluation = [[] for _ in range(STAGES)]
    index = 0
    for batch in source.iter_batches(batch_size=512):
        for row in batch.to_pylist():
            item = by_index.get(index)
            index += 1
            if item is None:
                continue
            new_object = next(obj for obj in row["objects"] if obj["decision"] == "new")
            old_object = next((obj for obj in row["objects"] if obj["decision"] == "obsolete"), None)
            base = {
                "sample_id": item["sample_id"],
                "stage": item["stage"],
                "kind": item["kind"],
                "subject_id": row["subject"]["id"],
                "relation_id": row["relation"]["id"],
                "update": row["update_prompt"],
                "ans": new_object["label"],
                "old_ans": old_object["label"] if old_object is not None else None,
            }
            train[item["stage"] - 1].append(base)
            evaluation[item["stage"] - 1].append({
                **base,
                "generalization_prompts": row["generalization_prompts"],
                "static_objects": [obj for obj in row["objects"] if obj["decision"] == "static"],
                "neighborhood": row["neighborhood"],
            })

    checks = []
    for stage in range(1, STAGES + 1):
        key = f"stage_{stage:02d}.jsonl"
        stage_train = sorted(train[stage - 1], key=lambda row: rank_by_id[row["sample_id"]])
        stage_eval = sorted(evaluation[stage - 1], key=lambda row: rank_by_id[row["sample_id"]])
        train_path = args.output / "train" / key
        eval_path = args.output / "evaluation" / key
        write_jsonl(train_path, stage_train)
        write_jsonl(eval_path, stage_eval)
        kinds = Counter(row["kind"] for row in stage_train)
        if len(stage_train) != PER_STAGE or len(stage_eval) != PER_STAGE or kinds != {"new": 2700, "replace": 300}:
            raise RuntimeError(f"Stage {stage} failed validation: {kinds}")
        checks.append({"stage": stage, "rows": len(stage_train), "kinds": dict(kinds), "train_sha256": file_sha256(train_path), "evaluation_sha256": file_sha256(eval_path)})

    metadata = {
        "schema_version": "wikifactdiff-12x3k/v1",
        "source": str(args.source),
        "source_revision": SOURCE_REVISION,
        "source_sha256": file_sha256(args.source),
        "seed": args.seed,
        "stages": STAGES,
        "per_stage": PER_STAGE,
        "replacement_per_stage": REPLACE_PER_STAGE,
        "eligible_rows": {"new": len(new), "replace": len(replace)},
        "manifest_sha256": file_sha256(args.output / "manifest.jsonl"),
        "stage_checks": checks,
        "note": "Stages are randomized arrivals from two Wikidata snapshots, not twelve dates. Base-model unknown screening and a WikiFactDiff runner are still required before training.",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in ("eligible_rows", "stages", "per_stage", "manifest_sha256")}, indent=2))


if __name__ == "__main__":
    main()
