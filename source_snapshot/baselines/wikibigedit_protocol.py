#!/usr/bin/env python3
"""Build and score a shared, reproducible WikiBigEdit 3K protocol.

The protocol deliberately separates model-independent sampling from optional
base-model mastery screening.  ``build`` freezes a deterministic candidate
pool and a provisional 3K selection.  ``attach-screen`` can later annotate the
same candidates and, if requested, derive a base-unknown-only 3K selection.

Only the tokenizer is needed while building the manifest; no model weights are
loaded.  Downstream baselines can import ``selected_records`` and
``panel_examples`` so every method consumes exactly the same facts and probes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import string
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "wikibigedit-protocol/v1"
DEFAULT_SEED = 42
DEFAULT_TOTAL = 3000
EXPECTED_INTERVALS = (
    "20240201_20240220",
    "20240220_20240301",
    "20240301_20240320",
    "20240320_20240401",
    "20240401_20240501",
    "20240501_20240601",
    "20240601_20240620",
    "20240620_20240701",
)
SOURCE_FIELDS = (
    "tag",
    "subject",
    "subject_id",
    "relation",
    "relation_id",
    "object",
    "object_id",
    "rephrase",
    "loc",
    "loc_ans",
    "mhop",
    "mhop_ans",
    "update",
    "personas",
    "ans",
)
CORE_REQUIRED_FIELDS = (
    "subject",
    "relation",
    "relation_id",
    "update",
    "rephrase",
    "personas",
    "ans",
)
PANEL_FIELDS: dict[str, tuple[str, str]] = {
    "update": ("update", "ans"),
    "rephrase": ("rephrase", "ans"),
    "personas": ("personas", "ans"),
    "locality": ("loc", "loc_ans"),
    "mhop": ("mhop", "mhop_ans"),
}
DIRECT_PANELS = ("update", "rephrase", "personas")
SCREEN_STATUSES = frozenset({"known", "unknown", "ambiguous", "error"})


def qa_prompt(question: str) -> str:
    """Canonical completion prompt used by all closed-book baselines."""

    return f"Question: {question.strip()}\nAnswer:"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalized_key(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def stable_sample_id(interval: str, source_row_idx: int) -> str:
    """ID stable for a fixed WikiBigEdit source revision and row ordering."""

    return f"wikibigedit:{interval}:{source_row_idx:06d}"


def _fact_fingerprint(row: Mapping[str, Any]) -> str:
    subject = row.get("subject_id") if _text(row.get("subject_id")) else row.get("subject")
    relation = row.get("relation_id") if _text(row.get("relation_id")) else row.get("relation")
    obj = row.get("object_id") if _text(row.get("object_id")) else row.get("ans")
    return _digest(
        {
            "subject": _normalized_key(subject),
            "relation": _normalized_key(relation),
            "object": _normalized_key(obj),
        }
    )


def _sample_fingerprint(row: Mapping[str, Any]) -> str:
    return _digest({field: row.get(field) for field in SOURCE_FIELDS})


def _rank_key(seed: int, sample_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{sample_id}".encode("utf-8")).digest()


def _targets(total: int, stage_count: int) -> list[int]:
    if total < 1:
        raise ValueError("total must be positive")
    if stage_count < 1:
        raise ValueError("stage_count must be positive")
    base, remainder = divmod(total, stage_count)
    return [base + (index < remainder) for index in range(stage_count)]


def _basic_invalid_reason(row: Any, allowed_tags: set[str]) -> str | None:
    if not isinstance(row, dict):
        return "not_an_object"
    if _normalized_key(row.get("tag")) not in allowed_tags:
        return "tag"
    for field in CORE_REQUIRED_FIELDS:
        if not _text(row.get(field)):
            return f"missing_{field}"
    return None


class TokenLengthFilter:
    """Tokenizer-aware length audit for the five WikiBigEdit panels."""

    def __init__(self, tokenizer: Any, max_length: int, max_answer_tokens: int):
        if max_length < 4:
            raise ValueError("max_length must be at least 4")
        if max_answer_tokens < 1:
            raise ValueError("max_answer_tokens must be positive")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_answer_tokens = max_answer_tokens

    def _lengths(self, texts: Sequence[str], add_special_tokens: bool) -> list[int]:
        if not texts:
            return []
        encoded = self.tokenizer(
            list(texts),
            add_special_tokens=add_special_tokens,
            padding=False,
            truncation=False,
        )
        if isinstance(encoded, Mapping):
            input_ids = encoded["input_ids"]
        else:
            input_ids = encoded.input_ids
        if input_ids and isinstance(input_ids[0], int):
            input_ids = [input_ids]
        return [len(item) for item in input_ids]

    def audit(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = [
            {"core_valid": True, "panels": {}} for _ in rows
        ]
        for panel, (question_field, answer_field) in PANEL_FIELDS.items():
            usable_indices = [
                index
                for index, row in enumerate(rows)
                if _text(row.get(question_field)) and _text(row.get(answer_field))
            ]
            answers = [str(rows[index][answer_field]).strip() for index in usable_indices]
            sequences = [
                qa_prompt(str(rows[index][question_field])) + " " + str(rows[index][answer_field]).strip()
                for index in usable_indices
            ]
            answer_lengths = self._lengths(answers, add_special_tokens=False)
            sequence_lengths = self._lengths(sequences, add_special_tokens=True)
            usable = set(usable_indices)
            for index in range(len(rows)):
                if index not in usable:
                    result[index]["panels"][panel] = {
                        "available": False,
                        "reason": "missing_question_or_answer",
                    }
                    if panel in DIRECT_PANELS:
                        result[index]["core_valid"] = False
            for index, answer_length, sequence_length in zip(
                usable_indices, answer_lengths, sequence_lengths
            ):
                answer_ok = 0 < answer_length <= self.max_answer_tokens
                # Training adds EOS, so keep one token of headroom.
                sequence_ok = sequence_length < self.max_length
                available = answer_ok and sequence_ok
                reason = None
                if not answer_ok:
                    reason = "answer_length"
                elif not sequence_ok:
                    reason = "sequence_length"
                result[index]["panels"][panel] = {
                    "available": available,
                    "answer_tokens": answer_length,
                    "sequence_tokens": sequence_length,
                    "reason": reason,
                }
                if panel in DIRECT_PANELS and not available:
                    result[index]["core_valid"] = False
        return result


def _load_source_descriptors(
    data_dir: Path, verify_hashes: bool = False
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_manifest_path = data_dir / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"missing source manifest: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    files = source_manifest.get("files")
    if not isinstance(files, list) or len(files) != 8:
        raise ValueError("full_hf_viewer manifest must describe exactly eight files")
    descriptors = sorted(files, key=lambda item: int(item.get("stage", 0)))
    intervals = tuple(str(item.get("interval", "")) for item in descriptors)
    if intervals != EXPECTED_INTERVALS:
        raise ValueError(f"unexpected interval order: {intervals}")
    for expected_stage, descriptor in enumerate(descriptors, start=1):
        if int(descriptor.get("stage", -1)) != expected_stage:
            raise ValueError(f"invalid stage descriptor: {descriptor}")
        path = data_dir / str(descriptor.get("path", ""))
        if not path.is_file():
            raise FileNotFoundError(path)
        if verify_hashes:
            expected = descriptor.get("sha256_reconstructed")
            if expected and sha256_file(path) != expected:
                raise ValueError(f"SHA-256 mismatch for {path}")
    source_summary = {
        "dataset_id": source_manifest.get("dataset_id"),
        "revision": source_manifest.get("revision"),
        "license": source_manifest.get("license"),
        "content_complete": source_manifest.get("content_complete"),
        "total_rows": source_manifest.get("total_rows"),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "files": [
            {
                "stage": int(item["stage"]),
                "interval": item["interval"],
                "path": item["path"],
                "rows": int(item["rows"]),
                "sha256_reconstructed": item.get("sha256_reconstructed"),
                "source_global_row_start": int(item["source_global_row_start"]),
                "source_global_row_end": int(item["source_global_row_end"]),
            }
            for item in descriptors
        ],
    }
    return source_summary, descriptors


def _tokenizer_metadata(tokenizer: Any, tokenizer_source: str | Path) -> dict[str, Any]:
    source = Path(tokenizer_source)
    files: dict[str, str] = {}
    if source.is_dir():
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
            path = source / name
            if path.is_file():
                files[name] = sha256_file(path)
    return {
        "source": source.name if source.is_dir() else str(tokenizer_source),
        "class": type(tokenizer).__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "file_sha256": files,
    }


def _materialize_record(
    row: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    source_row_idx: int,
    candidate_rank: int,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    interval = str(descriptor["interval"])
    global_index = int(descriptor["source_global_row_start"]) + source_row_idx
    record = {field: row.get(field) for field in SOURCE_FIELDS}
    record.update(
        {
            "sample_id": stable_sample_id(interval, source_row_idx),
            "stage": int(descriptor["stage"]),
            "interval": interval,
            "source_file": descriptor["path"],
            "source_row_idx": source_row_idx,
            "source_global_row_idx": global_index,
            "candidate_rank": candidate_rank,
            "fact_fingerprint": _fact_fingerprint(row),
            "source_record_fingerprint": _sample_fingerprint(row),
            "panel_audit": audit["panels"],
        }
    )
    return record


def _record_map(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("manifest.records must be a list")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = record.get("sample_id") if isinstance(record, dict) else None
        if not isinstance(sample_id, str) or sample_id in result:
            raise ValueError(f"invalid or duplicate sample_id: {sample_id!r}")
        result[sample_id] = record
    return result


def _choose_selected_ids(
    manifest: Mapping[str, Any],
    accepted_screen_statuses: set[str] | None = None,
    allow_shortfall: bool = False,
) -> dict[int, list[str]]:
    records = _record_map(manifest)
    max_per_relation = int(manifest["sampling"].get("max_per_relation", 0))
    selected: dict[int, list[str]] = {}
    globally_seen_facts: set[str] = set()
    for stage_info in manifest["stages"]:
        stage = int(stage_info["stage"])
        target = int(stage_info["target_count"])
        candidate_ids = list(stage_info["candidate_sample_ids"])
        relation_counts: Counter[str] = Counter()
        chosen: list[str] = []

        def eligible(sample_id: str, enforce_relation_cap: bool) -> bool:
            record = records[sample_id]
            if accepted_screen_statuses is not None:
                status = record.get("base_screen", {}).get("status")
                if status not in accepted_screen_statuses:
                    return False
            if record["fact_fingerprint"] in globally_seen_facts:
                return False
            relation = str(record.get("relation_id") or record.get("relation") or "")
            if enforce_relation_cap and max_per_relation > 0:
                if relation_counts[relation] >= max_per_relation:
                    return False
            return True

        for enforce_cap in ((True, False) if max_per_relation > 0 else (False,)):
            for sample_id in candidate_ids:
                if sample_id in chosen or not eligible(sample_id, enforce_cap):
                    continue
                record = records[sample_id]
                relation = str(record.get("relation_id") or record.get("relation") or "")
                chosen.append(sample_id)
                relation_counts[relation] += 1
                globally_seen_facts.add(record["fact_fingerprint"])
                if len(chosen) == target:
                    break
            if len(chosen) == target:
                break
        if len(chosen) < target and not allow_shortfall:
            qualifier = (
                f" with statuses {sorted(accepted_screen_statuses)}"
                if accepted_screen_statuses is not None
                else ""
            )
            raise RuntimeError(
                f"stage {stage}: selected {len(chosen)}/{target}{qualifier}; "
                "increase --candidate-multiplier or provide more screen results"
            )
        selected[stage] = chosen
    return selected


def _refresh_selection(
    manifest: dict[str, Any],
    selected_by_stage: Mapping[int, Sequence[str]],
    policy: str,
    allow_shortfall: bool,
) -> None:
    records = _record_map(manifest)
    all_selected: list[str] = []
    for stage_info in manifest["stages"]:
        stage = int(stage_info["stage"])
        ids = list(selected_by_stage.get(stage, ()))
        stage_info["selected_sample_ids"] = ids
        stage_info["selected_count"] = len(ids)
        all_selected.extend(ids)
    panels: dict[str, Any] = {}
    for panel in PANEL_FIELDS:
        ids = [
            sample_id
            for sample_id in all_selected
            if records[sample_id]["panel_audit"][panel]["available"]
        ]
        by_stage = Counter(str(records[sample_id]["stage"]) for sample_id in ids)
        panels[panel] = {
            "question_field": PANEL_FIELDS[panel][0],
            "answer_field": PANEL_FIELDS[panel][1],
            "sample_ids": ids,
            "count": len(ids),
            "count_by_stage": dict(sorted(by_stage.items(), key=lambda item: int(item[0]))),
        }
    manifest["selection"] = {
        "policy": policy,
        "allow_shortfall": bool(allow_shortfall),
        "sample_ids": all_selected,
        "total": len(all_selected),
    }
    manifest["panels"] = panels
    _refresh_identity(manifest)


def _identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    screening = manifest.get("screening", {})
    return {
        "schema_version": manifest["schema_version"],
        "source_manifest_sha256": manifest["source"]["source_manifest_sha256"],
        "source_revision": manifest["source"].get("revision"),
        "sampling": manifest["sampling"],
        "selection_policy": manifest.get("selection", {}).get("policy"),
        "selected_sample_ids": manifest.get("selection", {}).get("sample_ids", []),
        "screening_source_sha256": (
            screening.get("results_sha256")
            if screening.get("applied_to_selection")
            else None
        ),
    }


def _refresh_identity(manifest: dict[str, Any]) -> None:
    digest = _digest(_identity_payload(manifest))
    manifest["selection_sha256"] = digest
    manifest["protocol_id"] = f"wikibigedit-3k-{digest[:16]}"


def build_manifest(
    data_dir: Path,
    tokenizer: Any,
    tokenizer_source: str | Path,
    *,
    total: int = DEFAULT_TOTAL,
    seed: int = DEFAULT_SEED,
    candidate_multiplier: float = 2.0,
    max_length: int = 128,
    max_answer_tokens: int = 16,
    allowed_tags: Iterable[str] = ("new",),
    max_per_relation: int = 16,
    verify_source_hashes: bool = False,
) -> dict[str, Any]:
    """Create a deterministic candidate pool and provisional shared selection."""

    if candidate_multiplier < 1.0:
        raise ValueError("candidate_multiplier must be at least 1.0")
    if max_per_relation < 0:
        raise ValueError("max_per_relation cannot be negative")
    allowed = {_normalized_key(tag) for tag in allowed_tags if str(tag).strip()}
    if not allowed:
        raise ValueError("allowed_tags cannot be empty")
    source, descriptors = _load_source_descriptors(data_dir, verify_source_hashes)
    stage_targets = _targets(total, len(descriptors))
    length_filter = TokenLengthFilter(tokenizer, max_length, max_answer_tokens)
    records: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []

    for descriptor, target in zip(descriptors, stage_targets):
        path = data_dir / descriptor["path"]
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"source file is not a JSON list: {path}")
        if len(rows) != int(descriptor["rows"]):
            raise ValueError(
                f"row-count mismatch for {path}: {len(rows)} != {descriptor['rows']}"
            )
        invalid_reasons: Counter[str] = Counter()
        ranked: list[tuple[bytes, int, dict[str, Any]]] = []
        interval = str(descriptor["interval"])
        for source_row_idx, row in enumerate(rows):
            reason = _basic_invalid_reason(row, allowed)
            if reason is not None:
                invalid_reasons[reason] += 1
                continue
            sample_id = stable_sample_id(interval, source_row_idx)
            ranked.append((_rank_key(seed, sample_id), source_row_idx, row))
        ranked.sort(key=lambda item: (item[0], item[1]))

        candidate_target = max(target, int(math.ceil(target * candidate_multiplier)))
        accepted: list[dict[str, Any]] = []
        length_rejections: Counter[str] = Counter()
        examined = 0
        batch_size = 512
        for start in range(0, len(ranked), batch_size):
            batch = ranked[start : start + batch_size]
            audits = length_filter.audit([item[2] for item in batch])
            for (_, source_row_idx, row), audit in zip(batch, audits):
                examined += 1
                if not audit["core_valid"]:
                    for panel in DIRECT_PANELS:
                        panel_audit = audit["panels"][panel]
                        if not panel_audit["available"]:
                            length_rejections[f"{panel}:{panel_audit['reason']}"] += 1
                    continue
                accepted.append(
                    _materialize_record(
                        row,
                        descriptor,
                        source_row_idx,
                        len(accepted) + 1,
                        audit,
                    )
                )
                if len(accepted) == candidate_target:
                    break
            if len(accepted) == candidate_target:
                break
        if len(accepted) < candidate_target:
            raise RuntimeError(
                f"stage {descriptor['stage']}: only {len(accepted)} token-valid candidates; "
                f"need {candidate_target}"
            )
        candidate_ids = [record["sample_id"] for record in accepted]
        records.extend(accepted)
        stages.append(
            {
                "stage": int(descriptor["stage"]),
                "interval": interval,
                "source_file": descriptor["path"],
                "source_rows": len(rows),
                "target_count": target,
                "candidate_count": len(candidate_ids),
                "candidate_sample_ids": candidate_ids,
                "selected_count": 0,
                "selected_sample_ids": [],
                "basic_valid_count": len(ranked),
                "basic_invalid_counts": dict(sorted(invalid_reasons.items())),
                "length_examined_count": examined,
                "length_rejection_counts": dict(sorted(length_rejections.items())),
            }
        )
        print(
            f"stage {descriptor['stage']}: source={len(rows)} basic_valid={len(ranked)} "
            f"examined={examined} candidates={len(candidate_ids)} target={target}",
            file=sys.stderr,
            flush=True,
        )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "sampling": {
            "seed": seed,
            "target_total": total,
            "target_allocation": "balanced_across_eight_chronological_intervals",
            "candidate_multiplier": candidate_multiplier,
            "allowed_tags": sorted(allowed),
            "max_per_relation": max_per_relation,
            "deduplication": "exact_fact_fingerprint_across_final_selection",
            "max_length": max_length,
            "max_answer_tokens": max_answer_tokens,
            "prompt_template": "Question: {question}\\nAnswer:",
            "reserve_one_eos_token": True,
            "tokenizer": _tokenizer_metadata(tokenizer, tokenizer_source),
        },
        "screening": {
            "status": "pending",
            "applied_to_selection": False,
            "policy_note": (
                "Base mastery is checkpoint/prompt/scorer dependent, so it is an overlay rather "
                "than part of model-independent source sampling. Official rephrase/personas are "
                "reserved for evaluation; screen with update-derived prompts only."
            ),
        },
        "panel_policy": {
            "update": "training/reliability and optional base screening",
            "rephrase": "held-out generalization; never use for screening or training",
            "personas": "held-out persona generalization; never use for screening or training",
            "locality": "evaluate only rows with token-valid loc/loc_ans",
            "mhop": "evaluate only rows with token-valid mhop/mhop_ans",
        },
        "stages": stages,
        "records": records,
    }
    selected = _choose_selected_ids(manifest)
    _refresh_selection(
        manifest,
        selected,
        policy="model_independent_seeded_candidates",
        allow_shortfall=False,
    )
    validate_manifest(manifest)
    return manifest


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.casefold() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                rows.append(row)
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "predictions", "rows"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"cannot find a row list in {path}")


def _screen_status(row: Mapping[str, Any]) -> str:
    explicit = row.get("screen_status", row.get("status"))
    if isinstance(explicit, str):
        status = explicit.strip().casefold().replace("base_", "")
        aliases = {
            "mastered": "known",
            "not_mastered": "unknown",
            "uncertain": "ambiguous",
            "failed": "error",
        }
        status = aliases.get(status, status)
        if status in SCREEN_STATUSES:
            return status
    if isinstance(row.get("base_unknown"), bool):
        return "unknown" if row["base_unknown"] else "known"
    if isinstance(row.get("base_known"), bool):
        return "known" if row["base_known"] else "unknown"
    raise ValueError(
        "screen row needs screen_status/status or a boolean base_unknown/base_known"
    )


def attach_screen_results(
    manifest: Mapping[str, Any],
    screen_rows: Sequence[Mapping[str, Any]],
    *,
    results_sha256: str,
    model_id: str,
    screen_protocol_id: str,
    accepted_statuses: set[str] | None = None,
    allow_shortfall: bool = False,
) -> dict[str, Any]:
    """Attach screen annotations and optionally derive a screened selection."""

    result = copy.deepcopy(manifest)
    records = _record_map(result)
    candidate_ids = {
        sample_id
        for stage in result["stages"]
        for sample_id in stage["candidate_sample_ids"]
    }
    seen: set[str] = set()
    status_counts: Counter[str] = Counter()
    ignored_unknown_ids = 0
    for row in screen_rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str):
            raise ValueError("screen result is missing string sample_id")
        if sample_id in seen:
            raise ValueError(f"duplicate screen result for {sample_id}")
        seen.add(sample_id)
        if sample_id not in candidate_ids:
            ignored_unknown_ids += 1
            continue
        status = _screen_status(row)
        status_counts[status] += 1
        compact: dict[str, Any] = {
            "status": status,
            "result_fingerprint": _digest(row),
        }
        for key in (
            "prompt_count",
            "correct_count",
            "match_count",
            "gold_nll",
            "gold_logprob",
            "prediction",
        ):
            value = row.get(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                if key in row:
                    compact[key] = value
        records[sample_id]["base_screen"] = compact

    screened_count = sum(status_counts.values())
    result["screening"] = {
        "status": "complete" if screened_count == len(candidate_ids) else "partial",
        "applied_to_selection": accepted_statuses is not None,
        "model_id": model_id,
        "screen_protocol_id": screen_protocol_id,
        "results_sha256": results_sha256,
        "candidate_count": len(candidate_ids),
        "screened_candidate_count": screened_count,
        "missing_candidate_count": len(candidate_ids) - screened_count,
        "ignored_non_candidate_result_count": ignored_unknown_ids,
        "status_counts": dict(sorted(status_counts.items())),
        "accepted_statuses": sorted(accepted_statuses) if accepted_statuses else None,
        "policy_note": (
            "Screening is an overlay tied to the recorded base checkpoint and protocol. "
            "Rephrase/personas remain held out."
        ),
    }
    if accepted_statuses is not None:
        invalid = accepted_statuses - SCREEN_STATUSES
        if invalid:
            raise ValueError(f"invalid accepted screen statuses: {sorted(invalid)}")
        selected = _choose_selected_ids(
            result,
            accepted_screen_statuses=accepted_statuses,
            allow_shortfall=allow_shortfall,
        )
        policy = "base_screen:" + "+".join(sorted(accepted_statuses))
        _refresh_selection(result, selected, policy, allow_shortfall)
    else:
        _refresh_identity(result)
    validate_manifest(result)
    return result


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def selected_records(
    manifest: Mapping[str, Any], stage: int | None = None
) -> list[dict[str, Any]]:
    records = _record_map(manifest)
    ids = manifest["selection"]["sample_ids"]
    return [
        records[sample_id]
        for sample_id in ids
        if stage is None or int(records[sample_id]["stage"]) == stage
    ]


def candidate_records(
    manifest: Mapping[str, Any], stage: int | None = None
) -> list[dict[str, Any]]:
    records = _record_map(manifest)
    ids = [
        sample_id
        for stage_info in manifest["stages"]
        if stage is None or int(stage_info["stage"]) == stage
        for sample_id in stage_info["candidate_sample_ids"]
    ]
    return [records[sample_id] for sample_id in ids]


def panel_examples(
    manifest: Mapping[str, Any], panel: str, stage: int | None = None
) -> list[dict[str, Any]]:
    if panel not in PANEL_FIELDS:
        raise KeyError(f"unknown panel {panel!r}; choose from {sorted(PANEL_FIELDS)}")
    records = _record_map(manifest)
    question_field, answer_field = PANEL_FIELDS[panel]
    result = []
    for sample_id in manifest["panels"][panel]["sample_ids"]:
        record = records[sample_id]
        if stage is not None and int(record["stage"]) != stage:
            continue
        result.append(
            {
                "sample_id": sample_id,
                "stage": int(record["stage"]),
                "interval": record["interval"],
                "panel": panel,
                "question": record[question_field],
                "answer": record[answer_field],
                "prompt": qa_prompt(record[question_field]),
                "subject": record.get("subject"),
                "subject_id": record.get("subject_id"),
                "relation": record.get("relation"),
                "relation_id": record.get("relation_id"),
                "source_row_idx": record["source_row_idx"],
                "source_global_row_idx": record["source_global_row_idx"],
            }
        )
    return result


def normalize_answer(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    value = value.splitlines()[0] if value.splitlines() else value
    value = "".join(" " if char in string.punctuation else char for char in value)
    return " ".join(value.split())


def answer_scores(prediction: Any, gold: Any) -> tuple[bool, bool]:
    pred = normalize_answer(prediction)
    target = normalize_answer(gold)
    if not target:
        return False, False
    exact = pred == target
    pred_tokens = pred.split()
    target_tokens = target.split()
    contains = any(
        pred_tokens[index : index + len(target_tokens)] == target_tokens
        for index in range(max(len(pred_tokens) - len(target_tokens) + 1, 0))
    )
    return exact, contains


def score_predictions(
    manifest: Mapping[str, Any],
    prediction_rows: Sequence[Mapping[str, Any]],
    panels: Iterable[str] | None = None,
    default_panel: str | None = None,
) -> dict[str, Any]:
    requested = list(panels) if panels is not None else list(PANEL_FIELDS)
    invalid_panels = set(requested) - set(PANEL_FIELDS)
    if invalid_panels:
        raise ValueError(f"unknown panels: {sorted(invalid_panels)}")
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    for panel in requested:
        for example in panel_examples(manifest, panel):
            expected[(panel, example["sample_id"])] = example
    predictions: dict[tuple[str, str], Any] = {}
    unexpected = 0
    for row in prediction_rows:
        panel = row.get("panel", default_panel)
        sample_id = row.get("sample_id")
        if not isinstance(panel, str) or not isinstance(sample_id, str):
            raise ValueError("each prediction needs panel and sample_id")
        key = (panel, sample_id)
        if key in predictions:
            raise ValueError(f"duplicate prediction: panel={panel} sample_id={sample_id}")
        if key not in expected:
            unexpected += 1
            continue
        if "prediction" not in row:
            raise ValueError(f"prediction field missing for {key}")
        predictions[key] = row["prediction"]

    def summarize(keys: Sequence[tuple[str, str]]) -> dict[str, Any]:
        exact_count = 0
        match_count = 0
        supplied = 0
        for key in keys:
            if key not in predictions:
                continue
            supplied += 1
            exact, match = answer_scores(predictions[key], expected[key]["answer"])
            exact_count += int(exact)
            match_count += int(match)
        count = len(keys)
        return {
            "count": count,
            "supplied": supplied,
            "missing": count - supplied,
            "coverage": supplied / count if count else None,
            "exact_count": exact_count,
            "match_count": match_count,
            # Missing predictions are failures in the primary denominator.
            "exact_accuracy": exact_count / count if count else None,
            "match_accuracy": match_count / count if count else None,
            "match_accuracy_supplied_only": match_count / supplied if supplied else None,
        }

    report: dict[str, Any] = {
        "schema_version": "wikibigedit-score/v1",
        "protocol_id": manifest["protocol_id"],
        "selection_sha256": manifest["selection_sha256"],
        "unexpected_prediction_count": unexpected,
        "panels": {},
    }
    for panel in requested:
        panel_keys = [key for key in expected if key[0] == panel]
        by_stage = {
            str(stage): summarize(
                [key for key in panel_keys if expected[key]["stage"] == stage]
            )
            for stage in range(1, 9)
            if any(expected[key]["stage"] == stage for key in panel_keys)
        }
        report["panels"][panel] = {
            "overall": summarize(panel_keys),
            "by_stage": by_stage,
        }
    return report


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {manifest.get('schema_version')}")
    if not isinstance(manifest.get("stages"), list) or len(manifest["stages"]) != 8:
        raise ValueError("manifest must have eight stages")
    records = _record_map(manifest)
    candidate_ids: set[str] = set()
    selected_ids: list[str] = []
    for expected_stage, stage_info in enumerate(manifest["stages"], start=1):
        if int(stage_info["stage"]) != expected_stage:
            raise ValueError("stages are not in chronological order")
        candidates = stage_info["candidate_sample_ids"]
        selected = stage_info["selected_sample_ids"]
        if len(candidates) != len(set(candidates)):
            raise ValueError(f"duplicate candidate ID in stage {expected_stage}")
        if not set(selected).issubset(candidates):
            raise ValueError(f"selected IDs outside candidate pool in stage {expected_stage}")
        if any(sample_id not in records for sample_id in candidates):
            raise ValueError(f"candidate record missing in stage {expected_stage}")
        if candidate_ids.intersection(candidates):
            raise ValueError("candidate ID appears in multiple stages")
        candidate_ids.update(candidates)
        selected_ids.extend(selected)
    if set(records) != candidate_ids:
        raise ValueError("records and candidate IDs differ")
    if selected_ids != manifest["selection"]["sample_ids"]:
        raise ValueError("stage selections do not match selection.sample_ids")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("selected sample IDs are not unique")
    for panel, specification in manifest["panels"].items():
        if panel not in PANEL_FIELDS:
            raise ValueError(f"unknown panel in manifest: {panel}")
        ids = specification["sample_ids"]
        if not set(ids).issubset(selected_ids):
            raise ValueError(f"panel {panel} contains unselected rows")
        if any(not records[sample_id]["panel_audit"][panel]["available"] for sample_id in ids):
            raise ValueError(f"panel {panel} contains invalid rows")
    expected_digest = _digest(_identity_payload(manifest))
    if manifest.get("selection_sha256") != expected_digest:
        raise ValueError("selection_sha256 does not match manifest contents")
    if manifest.get("protocol_id") != f"wikibigedit-3k-{expected_digest[:16]}":
        raise ValueError("protocol_id does not match selection digest")


def _parse_panels(value: str | None) -> list[str]:
    if value is None or value.strip().casefold() == "all":
        return list(PANEL_FIELDS)
    result = [item.strip() for item in value.split(",") if item.strip()]
    invalid = set(result) - set(PANEL_FIELDS)
    if invalid:
        raise ValueError(f"unknown panels: {sorted(invalid)}")
    return result


def _load_tokenizer(source: str, local_files_only: bool) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("build requires transformers; use the project venv") from error
    return AutoTokenizer.from_pretrained(source, local_files_only=local_files_only)


def _build_command(args: argparse.Namespace) -> None:
    tokenizer = _load_tokenizer(args.tokenizer, args.local_files_only)
    manifest = build_manifest(
        args.data_dir,
        tokenizer,
        args.tokenizer,
        total=args.total,
        seed=args.seed,
        candidate_multiplier=args.candidate_multiplier,
        max_length=args.max_length,
        max_answer_tokens=args.max_answer_tokens,
        allowed_tags=args.tags.split(","),
        max_per_relation=args.max_per_relation,
        verify_source_hashes=args.verify_source_hashes,
    )
    _atomic_json(args.output, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "protocol_id": manifest["protocol_id"],
                "selected": manifest["selection"]["total"],
                "candidates": len(manifest["records"]),
                "panels": {
                    panel: value["count"] for panel, value in manifest["panels"].items()
                },
            },
            ensure_ascii=False,
        )
    )


def _attach_command(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    rows = _read_rows(args.screen_results)
    accepted = None
    if args.select_status:
        accepted = {item.strip().casefold() for item in args.select_status.split(",") if item.strip()}
        if not args.screen_model_id or not args.screen_protocol_id:
            raise ValueError(
                "--screen-model-id and --screen-protocol-id are required when reselection is requested"
            )
    result = attach_screen_results(
        manifest,
        rows,
        results_sha256=sha256_file(args.screen_results),
        model_id=args.screen_model_id or "unspecified",
        screen_protocol_id=args.screen_protocol_id or "unspecified",
        accepted_statuses=accepted,
        allow_shortfall=args.allow_shortfall,
    )
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "protocol_id": result["protocol_id"],
                "selection_policy": result["selection"]["policy"],
                "selected": result["selection"]["total"],
                "screening": result["screening"],
            },
            ensure_ascii=False,
        )
    )


def _export_command(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    panels = _parse_panels(args.panels)
    if args.scope == "selected":
        rows = []
        for panel in panels:
            rows.extend(panel_examples(manifest, panel))
    else:
        candidates = candidate_records(manifest)
        rows = []
        for record in candidates:
            for panel in panels:
                if not record["panel_audit"][panel]["available"]:
                    continue
                question_field, answer_field = PANEL_FIELDS[panel]
                rows.append(
                    {
                        "sample_id": record["sample_id"],
                        "stage": record["stage"],
                        "interval": record["interval"],
                        "panel": panel,
                        "question": record[question_field],
                        "answer": record[answer_field],
                        "prompt": qa_prompt(record[question_field]),
                        "subject": record.get("subject"),
                        "subject_id": record.get("subject_id"),
                        "relation": record.get("relation"),
                        "relation_id": record.get("relation_id"),
                        "source_row_idx": record["source_row_idx"],
                        "source_global_row_idx": record["source_global_row_idx"],
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(rows)}))


def _score_command(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    rows = _read_rows(args.predictions)
    report = score_predictions(
        manifest,
        rows,
        panels=_parse_panels(args.panels),
        default_panel=args.default_panel,
    )
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "protocol_id": report["protocol_id"]}))


def _validate_command(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    print(
        json.dumps(
            {
                "valid": True,
                "protocol_id": manifest["protocol_id"],
                "selected": manifest["selection"]["total"],
                "candidates": len(manifest["records"]),
            }
        )
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build the shared candidate/3K manifest")
    build.add_argument("--data-dir", type=Path, default=Path("WikiBigEdit/full_hf_viewer"))
    build.add_argument("--tokenizer", default="Qwen3-8B-Base")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--total", type=int, default=DEFAULT_TOTAL)
    build.add_argument("--seed", type=int, default=DEFAULT_SEED)
    build.add_argument("--candidate-multiplier", type=float, default=2.0)
    build.add_argument("--max-length", type=int, default=128)
    build.add_argument("--max-answer-tokens", type=int, default=16)
    build.add_argument("--tags", default="new")
    build.add_argument(
        "--max-per-relation",
        type=int,
        default=16,
        help="soft per-stage relation cap; a second pass fills any shortfall",
    )
    build.add_argument("--verify-source-hashes", action="store_true")
    build.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    build.set_defaults(func=_build_command)

    attach = subparsers.add_parser(
        "attach-screen", help="attach base-screen results and optionally select unknown facts"
    )
    attach.add_argument("--manifest", type=Path, required=True)
    attach.add_argument("--screen-results", type=Path, required=True)
    attach.add_argument("--output", type=Path, required=True)
    attach.add_argument("--screen-model-id")
    attach.add_argument("--screen-protocol-id")
    attach.add_argument(
        "--select-status",
        help="comma-separated statuses to retain (normally: unknown); omit to annotate only",
    )
    attach.add_argument("--allow-shortfall", action="store_true")
    attach.set_defaults(func=_attach_command)

    export = subparsers.add_parser("export", help="export panel examples as JSONL")
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--scope", choices=("selected", "candidates"), default="selected")
    export.add_argument("--panels", default="all")
    export.set_defaults(func=_export_command)

    score = subparsers.add_parser("score", help="score JSON/JSONL predictions")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--panels", default="all")
    score.add_argument("--default-panel", choices=tuple(PANEL_FIELDS))
    score.set_defaults(func=_score_command)

    validate = subparsers.add_parser("validate", help="validate a protocol manifest")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.set_defaults(func=_validate_command)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
