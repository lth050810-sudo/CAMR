#!/usr/bin/env python3
"""Run temporally valid open-book WikiBigEdit RAG baselines on frozen Qwen3.

This runner intentionally lives outside the closed-book editing runners.  It
never constructs an optimizer and freezes every reader parameter.  Two modes
are supported:

* ``dense_rag`` retrieves from only the edit facts available at the current
  WikiBigEdit stage with exact cosine search over a frozen encoder.
* ``oracle_rag`` inserts the corresponding gold edit fact as a diagnostic
  context.  It is not a retrieval baseline or a formal main-table method.

Embedding and per-axis generation caches are sharded/atomic so an interrupted
run can resume without recomputing completed work.  The canonical artifacts
(``run_config.json``, ``metrics.json``, ``predictions.jsonl``, and
``completed.json``) match the other baseline runners.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
from collections import Counter

import numpy as np
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from baselines.common import (
    answer_scores,
    atomic_json,
    axis_fields,
    group_by_stage,
    load_manifest,
    manifest_panel,
    sha256_file,
)


METHODS = ("dense_rag", "oracle_rag", "bm25_rag", "hybrid_rag")
AXES = ("update", "rephrase", "persona", "locality", "multihop")
DOCUMENT_TEMPLATE = "Question: <update>\nAnswer: <answer>"
RAG_PROMPT_TEMPLATE = (
    "Use the provided WikiBigEdit facts to answer the question. "
    "When a fact is relevant, prefer it over prior knowledge.\n"
    "Context:\n<context>\n\nQuestion: <question>\nAnswer:"
)
TEMPORAL_POLICY = "a fact is retrievable iff fact.stage <= evaluation_stage"


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "evaluation_panels" or value is not None
    }


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def fact_document(row: Mapping[str, Any]) -> str:
    """Canonical indexed/context document for one newly available fact."""

    return f"Question: {str(row['update']).strip()}\nAnswer: {str(row['ans']).strip()}"


def rag_prompt(question: str, documents: Sequence[str]) -> str:
    """Build a reader prompt while keeping the shared outer QA completion form."""

    if documents:
        # Rank 1 is intentionally closest to the final question.  With left
        # truncation this preserves the strongest passage first.
        ranked = [
            f"[{rank}] {document.strip()}"
            for rank, document in reversed(list(enumerate(documents, start=1)))
        ]
        context = "\n\n".join(ranked)
    else:
        context = "(no retrieved facts)"
    return (
        "Use the provided WikiBigEdit facts to answer the question. "
        "When a fact is relevant, prefer it over prior knowledge.\n"
        f"Context:\n{context}\n\n"
        f"Question: {question.strip()}\nAnswer:"
    )


def temporal_rows(
    rows: Sequence[dict[str, Any]], evaluation_stage: int
) -> list[dict[str, Any]]:
    """Return the exact stage prefix visible to retrieval/evaluation."""

    return [row for row in rows if int(row["stage"]) <= evaluation_stage]


def validate_no_future(
    retrieved_rows: Sequence[Mapping[str, Any]], evaluation_stage: int
) -> None:
    future = [
        str(row.get("sample_id"))
        for row in retrieved_rows
        if int(row["stage"]) > evaluation_stage
    ]
    if future:
        raise RuntimeError(
            f"future-fact leakage at stage {evaluation_stage}: {future[:3]}"
        )


@dataclass(frozen=True)
class RetrievalHit:
    row: dict[str, Any]
    score: float


class TextEncoder(Protocol):
    provenance: Mapping[str, Any]

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        """Return one normalized float embedding per input text."""


def _local_model_provenance(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        return {"local_directory": None}
    patterns = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    files: list[dict[str, Any]] = []
    named = {item.name: item for item in path.iterdir() if item.is_file()}
    selected = [named[name] for name in patterns if name in named]
    selected.extend(sorted(path.glob("model-*.safetensors")))
    selected.extend(sorted(path.glob("pytorch_model-*.bin")))
    for item in dict.fromkeys(selected):
        files.append(
            {
                "name": item.name,
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    return {
        "local_directory": str(path.resolve()),
        "files": files,
        "tree_fingerprint": _canonical_digest(files),
    }


class HFMeanPoolEncoder:
    """Small dependency-free Hugging Face dense encoder wrapper."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: torch.device,
        batch_size: int,
        max_length: int,
        query_prefix: str,
        document_prefix: str,
        local_files_only: bool,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            trust_remote_code=False,
            use_fast=True,
        )
        model_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "trust_remote_code": False,
        }
        if device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        self.model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
        self.model.to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        source = Path(model_name_or_path).expanduser()
        self.provenance = {
            "model_name_or_path": model_name_or_path,
            "model_class": type(self.model).__name__,
            "tokenizer_class": type(self.tokenizer).__name__,
            "config_name_or_path": getattr(self.model.config, "_name_or_path", None),
            "commit_hash": getattr(self.model.config, "_commit_hash", None),
            "hidden_size": getattr(self.model.config, "hidden_size", None),
            "pooling": "attention-mask mean pooling over last_hidden_state",
            "normalize": "L2",
            "max_length": max_length,
            "query_prefix": query_prefix,
            "document_prefix": document_prefix,
            "local_files_only": local_files_only,
            **_local_model_provenance(source),
        }
        self.provenance = {
            **self.provenance,
            "provenance_sha256": _canonical_digest(self.provenance),
        }

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            outputs = self.model(**encoded, return_dict=True)
            hidden = outputs.last_hidden_state.float()
            mask = encoded.attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            chunks.append(F.normalize(pooled, dim=-1).cpu())
        if not chunks:
            hidden_size = int(getattr(self.model.config, "hidden_size", 0))
            return torch.empty((0, hidden_size), dtype=torch.float32)
        return torch.cat(chunks, dim=0)

    def encode_queries(self, texts: Sequence[str]) -> torch.Tensor:
        return self.encode([self.query_prefix + text for text in texts])

    def encode_documents(self, texts: Sequence[str]) -> torch.Tensor:
        return self.encode([self.document_prefix + text for text in texts])


def encode_with_shard_cache(
    encoder: TextEncoder,
    texts: Sequence[str],
    identifiers: Sequence[str],
    *,
    cache_root: Path,
    namespace: str,
    shard_size: int,
    kind: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Encode texts with atomic shards that can be reused fully offline."""

    if len(texts) != len(identifiers):
        raise ValueError("texts and identifiers must have equal length")
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    identity = {
        "namespace": namespace,
        "kind": kind,
        "identifiers": list(identifiers),
        "texts_sha256": _canonical_digest(list(texts)),
        "encoder": dict(encoder.provenance),
        "count": len(texts),
        "shard_size": shard_size,
    }
    signature = _canonical_digest(identity)
    cache_dir = cache_root / f"{namespace}-{signature[:16]}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("signature") != signature:
            raise ValueError(f"embedding cache identity mismatch: {metadata_path}")
    else:
        atomic_json(metadata_path, {**identity, "signature": signature})

    chunks: list[torch.Tensor] = []
    encoded_shards = 0
    reused_shards = 0
    for start in range(0, len(texts), shard_size):
        end = min(start + shard_size, len(texts))
        shard_path = cache_dir / f"shard_{start:06d}_{end:06d}.pt"
        tensor: torch.Tensor | None = None
        if shard_path.is_file():
            try:
                loaded = torch.load(shard_path, map_location="cpu", weights_only=True)
                if (
                    isinstance(loaded, torch.Tensor)
                    and loaded.ndim == 2
                    and int(loaded.shape[0]) == end - start
                ):
                    tensor = loaded.float()
                    reused_shards += 1
            except (OSError, RuntimeError, ValueError, TypeError):
                tensor = None
        if tensor is None:
            method_name = "encode_queries" if kind == "query" else "encode_documents"
            method = getattr(encoder, method_name, encoder.encode)
            tensor = method(texts[start:end]).detach().cpu().float()
            if tensor.ndim != 2 or int(tensor.shape[0]) != end - start:
                raise ValueError("encoder returned an invalid embedding shape")
            tensor = F.normalize(tensor, dim=-1)
            # Keep float32 as the canonical cache representation so a resumed
            # run cannot change a close top-k ordering through quantization.
            _atomic_torch_save(shard_path, tensor.to(torch.float32))
            encoded_shards += 1
        chunks.append(F.normalize(tensor, dim=-1))
    embeddings = (
        torch.cat(chunks, dim=0)
        if chunks
        else torch.empty((0, 0), dtype=torch.float32)
    )
    completed = {
        "signature": signature,
        "cache_dir": str(cache_dir.resolve()),
        "count": len(texts),
        "dimension": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
        "encoded_shards_this_run": encoded_shards,
        "reused_shards_this_run": reused_shards,
    }
    atomic_json(cache_dir / "completed.json", completed)
    return embeddings, {**identity, **completed}


class TemporalDenseIndex:
    """Exact cosine index with an enforced WikiBigEdit stage boundary."""

    def __init__(
        self, rows: Sequence[dict[str, Any]], embeddings: torch.Tensor
    ) -> None:
        if embeddings.ndim != 2 or int(embeddings.shape[0]) != len(rows):
            raise ValueError("document embedding count does not match rows")
        self.rows = list(rows)
        self.embeddings = F.normalize(embeddings.detach().cpu().float(), dim=-1)

    def search(
        self,
        query_embeddings: torch.Tensor,
        *,
        evaluation_stage: int,
        top_k: int,
        batch_size: int = 1024,
    ) -> list[list[RetrievalHit]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        eligible = [
            index
            for index, row in enumerate(self.rows)
            if int(row["stage"]) <= evaluation_stage
        ]
        if not eligible:
            raise ValueError(f"no documents are available at stage {evaluation_stage}")
        eligible_tensor = torch.tensor(eligible, dtype=torch.long)
        documents = self.embeddings.index_select(0, eligible_tensor)
        queries = F.normalize(query_embeddings.detach().cpu().float(), dim=-1)
        result: list[list[RetrievalHit]] = []
        count = min(top_k, len(eligible))
        for start in range(0, int(queries.shape[0]), batch_size):
            scores = queries[start : start + batch_size] @ documents.T
            values, positions = torch.topk(
                scores, k=count, dim=1, largest=True, sorted=True
            )
            for row_values, row_positions in zip(values, positions):
                hits = [
                    RetrievalHit(
                        row=self.rows[eligible[int(position)]],
                        score=float(score),
                    )
                    for score, position in zip(row_values.tolist(), row_positions.tolist())
                ]
                validate_no_future([hit.row for hit in hits], evaluation_stage)
                result.append(hits)
        return result


class TemporalBM25Index:
    """Okapi BM25 with corpus statistics recomputed on each visible prefix."""
    def __init__(self, rows, k1=1.5, b=0.75):
        self.rows = list(rows)
        self.k1, self.b = k1, b
        self.tokens = [Counter(self.tokenize(fact_document(r))) for r in rows]
        self.cache = {}

    @staticmethod
    def tokenize(text):
        return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)

    def search(self, queries, *, evaluation_stage, top_k):
        if evaluation_stage not in self.cache:
            ids = [i for i,r in enumerate(self.rows) if int(r['stage']) <= evaluation_stage]
            lengths = np.array([sum(self.tokens[i].values()) for i in ids], dtype=float)
            avg = max(float(lengths.mean()), 1.0)
            postings = {}
            for j,i in enumerate(ids):
                for token, tf in self.tokens[i].items():
                    postings.setdefault(token, []).append((j, tf))
            self.cache[evaluation_stage] = ids, lengths, avg, postings
        ids, lengths, avg, postings = self.cache[evaluation_stage]
        results = []
        for query in queries:
            scores = np.zeros(len(ids), dtype=float)
            for token in set(self.tokenize(query)):
                posting = postings.get(token, [])
                if not posting: continue
                positions, tf = np.array(posting).T
                idf = np.log(1 + (len(ids)-len(posting)+0.5)/(len(posting)+0.5))
                scores[positions] += idf * tf*(self.k1+1)/(tf+self.k1*(1-self.b+self.b*lengths[positions]/avg))
            order = np.argsort(-scores, kind='stable')[:top_k]
            hits = [RetrievalHit(self.rows[ids[i]], float(scores[i])) for i in order]
            validate_no_future([h.row for h in hits], evaluation_stage)
            results.append(hits)
        return results


def reciprocal_rank_fusion(dense, sparse, top_k, constant=60):
    scores, records = {}, {}
    for ranking in (dense, sparse):
        for rank, hit in enumerate(ranking, 1):
            sid = str(hit.row['sample_id'])
            records[sid] = hit.row
            scores[sid] = scores.get(sid, 0.0) + 1.0/(constant+rank)
    ranked = sorted(scores, key=lambda sid: (-scores[sid], sid))[:top_k]
    return [RetrievalHit(records[sid], scores[sid]) for sid in ranked]


def paper_prompt(question, hits):
    context = ' '.join(f"Q: {h.row['update'].strip()} A: {h.row['ans'].strip()}" for h in hits)
    return f"{context} Q: {question.strip()} A:".strip()


def oracle_hits(
    rows: Sequence[dict[str, Any]], evaluation_stage: int
) -> list[list[RetrievalHit]]:
    output: list[list[RetrievalHit]] = []
    for row in rows:
        if int(row["stage"]) > evaluation_stage:
            raise ValueError(
                f"oracle fact {row['sample_id']} is from future stage {row['stage']}"
            )
        output.append([RetrievalHit(row=row, score=1.0)])
    return output


def freeze_reader(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("reader freezing failed")


def load_reader(model_dir: Path, device: torch.device) -> tuple[Any, torch.nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "local_files_only": True,
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
        "attn_implementation": "flash_attention_2",
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    except Exception:
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    model.to(device)
    freeze_reader(model)
    return tokenizer, model


@torch.inference_mode()
def generate_from_prompts(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    device: torch.device,
    *,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
) -> list[str]:
    predictions: list[str] = []
    previous_padding = tokenizer.padding_side
    previous_truncation = tokenizer.truncation_side
    previous_cache = model.config.use_cache
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    model.config.use_cache = True
    model.eval()
    try:
        for start in range(0, len(prompts), batch_size):
            encoded = tokenizer(
                list(prompts[start : start + batch_size]),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            ).to(device)
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            continuation = generated[:, encoded.input_ids.shape[1] :]
            predictions.extend(
                tokenizer.batch_decode(continuation, skip_special_tokens=True)
            )
    finally:
        tokenizer.padding_side = previous_padding
        tokenizer.truncation_side = previous_truncation
        model.config.use_cache = previous_cache
    return predictions


def valid_axis_rows(
    rows: Sequence[dict[str, Any]], axis: str
) -> list[dict[str, Any]]:
    question_key, answer_key = axis_fields(axis)
    return [
        row
        for row in rows
        if isinstance(row.get(question_key), str)
        and str(row[question_key]).strip()
        and isinstance(row.get(answer_key), str)
        and str(row[answer_key]).strip()
    ]


def build_prediction_records(
    rows: Sequence[dict[str, Any]],
    axis: str,
    hits: Sequence[Sequence[RetrievalHit]],
    predictions: Sequence[str],
    *,
    checkpoint: str,
    evaluation_stage: int,
    method: str,
) -> list[dict[str, Any]]:
    if not (len(rows) == len(hits) == len(predictions)):
        raise ValueError("rows, retrieval hits, and predictions must align")
    question_key, answer_key = axis_fields(axis)
    records: list[dict[str, Any]] = []
    for row, row_hits, prediction in zip(rows, hits, predictions):
        retrieved_rows = [hit.row for hit in row_hits]
        validate_no_future(retrieved_rows, evaluation_stage)
        retrieved_ids = [str(hit.row["sample_id"]) for hit in row_hits]
        retrieval_evaluable = axis != "locality"
        scores = answer_scores(prediction, str(row[answer_key]))
        context = [fact_document(hit.row) for hit in row_hits]
        records.append(
            {
                "checkpoint": checkpoint,
                "current_stage": evaluation_stage,
                "sample_id": str(row["sample_id"]),
                "fact_stage": int(row["stage"]),
                "axis": axis,
                "question": str(row[question_key]),
                "gold": str(row[answer_key]),
                "prediction": prediction,
                **scores,
                "retrieval_method": method,
                "retrieved_sample_ids": retrieved_ids,
                "retrieved_fact_stages": [int(hit.row["stage"]) for hit in row_hits],
                "retrieval_scores": [float(hit.score) for hit in row_hits],
                "retrieval_evaluable": retrieval_evaluable,
                "retrieval_hit": (
                    str(row["sample_id"]) in retrieved_ids
                    if retrieval_evaluable
                    else None
                ),
                "oracle_supports_axis": method == "oracle_rag" and axis != "locality",
                "context": context,
                "reader_prompt_sha256": _canonical_digest(
                    rag_prompt(str(row[question_key]), context)
                ),
            }
        )
    return records


def _accuracy(records: Sequence[dict[str, Any]], key: str) -> float | None:
    if not records:
        return None
    return sum(bool(record[key]) for record in records) / len(records)


def summarize_rag_records(
    records: Sequence[dict[str, Any]],
    *,
    checkpoint: str,
    current_stage: int,
    num_facts: int,
    index_count: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "checkpoint": checkpoint,
        "current_stage": current_stage,
        "num_facts": num_facts,
    }
    for axis in AXES:
        selected = [record for record in records if record["axis"] == axis]
        exact_count = sum(bool(record["exact"]) for record in selected)
        match_count = sum(bool(record["match"]) for record in selected)
        result[axis] = {
            "count": len(selected),
            "exact_count": exact_count,
            "match_count": match_count,
            "exact_accuracy": exact_count / len(selected) if selected else None,
            "match_accuracy": match_count / len(selected) if selected else None,
        }
        result[f"{axis}_accuracy"] = _accuracy(selected, "match")

    factual = [
        record
        for record in records
        if record["axis"] in {"update", "rephrase", "persona"}
    ]
    current = [record for record in factual if int(record["fact_stage"]) == current_stage]
    past = [record for record in factual if int(record["fact_stage"]) < current_stage]
    for axis in ("update", "rephrase", "persona"):
        result[f"current_{axis}_accuracy"] = _accuracy(
            [record for record in current if record["axis"] == axis], "match"
        )
        result[f"past_{axis}_retention"] = _accuracy(
            [record for record in past if record["axis"] == axis], "match"
        )
    per_stage: dict[str, dict[str, float | None]] = {}
    for fact_stage in sorted({int(record["fact_stage"]) for record in factual}):
        stage_records = [
            record for record in factual if int(record["fact_stage"]) == fact_stage
        ]
        per_stage[str(fact_stage)] = {
            axis: _accuracy(
                [record for record in stage_records if record["axis"] == axis],
                "match",
            )
            for axis in ("update", "rephrase", "persona")
        }
    result["per_fact_stage"] = per_stage

    retrieval_records = [
        record for record in records if bool(record.get("retrieval_evaluable"))
    ]
    hit_count = sum(bool(record.get("retrieval_hit")) for record in retrieval_records)
    future_leak_count = sum(
        any(int(stage) > current_stage for stage in record["retrieved_fact_stages"])
        for record in records
    )
    result["retrieval"] = {
        "index_count": index_count,
        "evaluation_stage": current_stage,
        "temporal_policy": TEMPORAL_POLICY,
        "evaluable_query_count": len(retrieval_records),
        "hit_count": hit_count,
        "recall_at_k": (
            hit_count / len(retrieval_records) if retrieval_records else None
        ),
        "future_leak_count": future_leak_count,
        "per_axis": {
            axis: {
                "count": len(selected := [
                    record
                    for record in retrieval_records
                    if record["axis"] == axis
                ]),
                "hit_count": sum(bool(record.get("retrieval_hit")) for record in selected),
                "recall_at_k": (
                    sum(bool(record.get("retrieval_hit")) for record in selected)
                    / len(selected)
                    if selected
                    else None
                ),
            }
            for axis in AXES
        },
    }
    if future_leak_count:
        raise AssertionError("retrieval metrics detected future-fact leakage")
    return result


def _axis_cache_paths(
    output_dir: Path, checkpoint: str, axis: str
) -> tuple[Path, Path]:
    directory = output_dir / "cache" / "evaluation" / checkpoint
    return directory / f"{axis}.jsonl", directory / f"{axis}.meta.json"


def load_axis_cache(
    output_dir: Path,
    checkpoint: str,
    axis: str,
    *,
    run_fingerprint: str,
    expected_count: int,
) -> list[dict[str, Any]] | None:
    predictions_path, metadata_path = _axis_cache_paths(output_dir, checkpoint, axis)
    if not predictions_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("run_fingerprint") != run_fingerprint:
            return None
        if int(metadata.get("count", -1)) != expected_count:
            return None
        if metadata.get("predictions_sha256") != sha256_file(predictions_path):
            return None
        records = load_jsonl(predictions_path)
        if len(records) != expected_count:
            return None
        return records
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def write_axis_cache(
    output_dir: Path,
    checkpoint: str,
    axis: str,
    records: Sequence[dict[str, Any]],
    *,
    run_fingerprint: str,
) -> None:
    predictions_path, metadata_path = _axis_cache_paths(output_dir, checkpoint, axis)
    atomic_jsonl(predictions_path, records)
    atomic_json(
        metadata_path,
        {
            "run_fingerprint": run_fingerprint,
            "checkpoint": checkpoint,
            "axis": axis,
            "count": len(records),
            "predictions_sha256": sha256_file(predictions_path),
        },
    )


def _stage_rows(
    all_rows: Sequence[dict[str, Any]],
    panel_rows: Mapping[str, Sequence[dict[str, Any]]],
    stage: int,
    *,
    evaluation_panels: Mapping[str, Sequence[dict[str, Any]]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if evaluation_panels is not None:
        # Frozen query panels do not restrict the retrieval corpus. Unlike the
        # closed-book replay runner, RAG only asks probes whose fact has arrived.
        return {
            axis: valid_axis_rows(temporal_rows(evaluation_panels[axis], stage), axis)
            for axis in AXES
        }
    seen = temporal_rows(all_rows, stage)
    return {
        "update": valid_axis_rows(seen, "update"),
        "rephrase": valid_axis_rows(seen, "rephrase"),
        "persona": valid_axis_rows(seen, "persona"),
        "locality": valid_axis_rows(temporal_rows(panel_rows["locality"], stage), "locality"),
        "multihop": valid_axis_rows(temporal_rows(panel_rows["multihop"], stage), "multihop"),
    }


def load_evaluation_panels(
    path: Path,
    rows: Sequence[dict[str, Any]],
    manifest: Mapping[str, Any],
    *,
    manifest_sha256: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Resolve full-replay sampled queries without changing the index rows."""
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("evaluation panels must be a JSON object")
    specification = payload.get("evaluation_panels", payload)
    if not isinstance(specification, dict):
        raise ValueError("evaluation_panels must be a JSON object")
    supplied_config = payload.get("config", {})
    declared_counts = specification.get("counts", {})
    if not isinstance(supplied_config, dict) or not isinstance(declared_counts, dict):
        raise ValueError("evaluation panel config/counts must be JSON objects")
    supplied_manifest_hash = supplied_config.get("manifest_sha256")
    if supplied_manifest_hash is None:
        supplied_manifest_hash = specification.get("manifest_sha256")
    if supplied_manifest_hash is not None and supplied_manifest_hash != manifest_sha256:
        raise ValueError("evaluation panels were frozen for a different manifest")
    supplied_panel_hash = supplied_config.get("panel_fingerprint")
    if supplied_panel_hash is not None and supplied_panel_hash != _canonical_digest(specification):
        raise ValueError("evaluation panels disagree with the full-replay plan panel fingerprint")
    identifiers = specification.get("sample_ids")
    if not isinstance(identifiers, dict) or set(identifiers) != set(AXES):
        raise ValueError(f"evaluation panel sample_ids must contain exactly these axes: {AXES}")
    row_map = {str(row["sample_id"]): row for row in rows}
    selection = manifest.get("selection", {}).get("sample_ids")
    if (not isinstance(selection, list) or len(selection) != len(set(selection))
            or set(selection) != set(row_map) or len(row_map) != len(rows)):
        raise ValueError("sampled evaluation requires the entire unique manifest selection as its index")
    if any(isinstance(row.get("stage"), bool) or not isinstance(row.get("stage"), int)
           or not 1 <= row["stage"] <= 8 for row in rows):
        raise ValueError("manifest rows must retain their original integer stages 1 through 8")
    resolved: dict[str, list[dict[str, Any]]] = {}
    counts_by_stage: dict[str, dict[str, int]] = {}
    for axis in AXES:
        ids = identifiers[axis]
        if (not isinstance(ids, list) or any(not isinstance(sid, str) or not sid for sid in ids)
                or len(ids) != len(set(ids))):
            raise ValueError(f"evaluation panel {axis} must contain unique string sample IDs")
        missing = set(ids) - set(row_map)
        if missing:
            raise ValueError(f"evaluation panel {axis} contains IDs outside the manifest selection: {sorted(missing)[:3]}")
        manifest_axis = {"persona": "personas", "multihop": "mhop"}.get(axis, axis)
        frozen = manifest.get("panels", {}).get(manifest_axis)
        if not isinstance(frozen, dict) or not isinstance(frozen.get("sample_ids"), list):
            raise ValueError(f"manifest has no frozen availability panel for {axis}")
        if not set(ids).issubset(set(frozen["sample_ids"])):
            raise ValueError(f"evaluation panel {axis} contains IDs masked by manifest availability")
        question_key, answer_key = axis_fields(axis)
        for sid in ids:
            row = row_map[sid]
            audits = [row.get("panel_audit", {}).get(manifest_axis),
                      row.get("token_length_audit", {}).get("panels", {}).get(manifest_axis)]
            if any(audit is not None and (not isinstance(audit, dict) or audit.get("available") is not True)
                   for audit in audits):
                raise ValueError(f"evaluation panel {axis} contains an unavailable row: {sid}")
            if not all(isinstance(row.get(key), str) and row[key].strip()
                       for key in (question_key, answer_key)):
                raise ValueError(f"evaluation panel {axis} has a missing question/answer: {sid}")
        declared_count = declared_counts.get(axis)
        if declared_count is not None and declared_count != len(ids):
            raise ValueError(f"evaluation panel {axis} count disagrees with its IDs")
        # Match full-replay's actual evaluation order: manifest stream order,
        # independent of the order in which panel membership was sampled.
        id_set = set(ids)
        resolved[axis] = [row for row in rows if str(row["sample_id"]) in id_set]
        counts_by_stage[axis] = dict(sorted(Counter(str(row["stage"]) for row in resolved[axis]).items()))
    provenance = {
        "source": str(path.resolve()),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "panel_payload_sha256": _canonical_digest(specification),
        "manifest_sha256": manifest_sha256,
        "counts": {axis: len(resolved[axis]) for axis in AXES},
        "counts_by_stage": counts_by_stage,
        "sample_ids_sha256": {
            axis: _canonical_digest([str(row["sample_id"]) for row in resolved[axis]])
            for axis in AXES
        },
        "evaluation_scope": "complete selected fact index with frozen sampled five-axis evaluation; not full-data five-axis accuracy",
        "query_temporal_policy": "every axis only evaluates panel facts with fact.stage <= evaluation_stage",
        "source_panel_temporal_policy": specification.get("temporal_policy"),
        "query_order": "manifest stream order",
    }
    return resolved, provenance


def bind_evaluation_panels(
    run_identity: dict[str, Any], provenance: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Leave the legacy run identity untouched unless panels were requested."""
    if provenance is None:
        return run_identity
    return {**run_identity, "evaluation_panels": {
        key: value for key, value in provenance.items() if key != "source"
    }}


def validate_evaluation_panel_resume(output_dir: Path, provenance: Mapping[str, Any]) -> None:
    """Reject a different query panel before building a large embedding index."""
    config_path = output_dir / "run_config.json"
    if not config_path.is_file():
        return
    existing = json.loads(config_path.read_text(encoding="utf-8"))
    recorded = existing.get("evaluation_panels_provenance")
    if (not isinstance(recorded, dict)
            or bind_evaluation_panels({}, recorded) != bind_evaluation_panels({}, provenance)):
        raise ValueError("--resume output has different frozen evaluation panels")


def sampled_evaluation_counts(rows_by_axis: Mapping[str, Sequence[dict[str, Any]]]) -> dict[str, Any]:
    counts = {axis: len(rows_by_axis[axis]) for axis in AXES}
    return {
        "evaluated_count": sum(counts.values()),
        "evaluated_count_unit": "axis/sample_id query pairs at this checkpoint",
        "evaluated_count_by_axis": counts,
        "evaluated_unique_fact_count": len({str(row["sample_id"]) for axis in AXES for row in rows_by_axis[axis]}),
    }


def _rebuild_canonical_artifacts(
    output_dir: Path, stages: Sequence[int]
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    for stage in stages:
        checkpoint = f"tau{stage}"
        metric_path = output_dir / "cache" / "evaluation" / checkpoint / "metrics.json"
        if not metric_path.is_file():
            continue
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
        metrics.append(metric)
        for axis in AXES:
            prediction_path, _ = _axis_cache_paths(output_dir, checkpoint, axis)
            if prediction_path.is_file():
                all_predictions.extend(load_jsonl(prediction_path))
    atomic_json(output_dir / "metrics.json", metrics)
    atomic_jsonl(output_dir / "predictions.jsonl", all_predictions)
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("Qwen3-8B-Base"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--evaluation-panels", type=Path,
        help="full-replay evaluation_panels.json or plan.json; index all selected facts and evaluate only these frozen panel IDs",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--recipe", choices=("shared", "wikibigedit"), default="shared")
    parser.add_argument("--rrf-depth", type=int, default=100)
    parser.add_argument("--rrf-constant", type=int, default=60)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="HF encoder ID or local directory; ignored for oracle_rag",
    )
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--embedding-shard-size", type=int, default=256)
    parser.add_argument("--embedding-max-length", type=int, default=256)
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        help="default: <output-dir>/cache/embeddings",
    )
    parser.add_argument(
        "--allow-embedding-download",
        action="store_true",
        help="permit HF network access; default requires an already cached/local encoder",
    )
    parser.add_argument(
        "--query-prefix",
        default="",
    )
    parser.add_argument("--document-prefix", default="")
    parser.add_argument("--retrieval-batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--reader-max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-stages", type=int, default=0)
    parser.add_argument("--examples-per-stage", type=int, default=0)
    parser.add_argument("--locality-panel-size", type=int, default=0)
    parser.add_argument("--multihop-panel-size", type=int, default=0)
    parser.add_argument("--no-stage-eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.evaluation_panels is not None and any((
        args.max_stages, args.examples_per_stage, args.locality_panel_size, args.multihop_panel_size,
    )):
        parser.error("--evaluation-panels requires the full manifest index and exact frozen panels; do not combine it with stage/example/panel truncation flags")
    return args


def _select_rows(
    rows: Sequence[dict[str, Any]], max_stages: int, examples_per_stage: int
) -> list[dict[str, Any]]:
    grouped = group_by_stage(rows)
    selected: list[dict[str, Any]] = []
    for stage, stage_rows in grouped.items():
        if max_stages > 0 and stage > max_stages:
            continue
        selected.extend(
            stage_rows[:examples_per_stage]
            if examples_per_stage > 0
            else stage_rows
        )
    return selected


def _reader_provenance(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    tokenizer_path = model_dir / "tokenizer_config.json"
    index_path = model_dir / "model.safetensors.index.json"
    return {
        "path": str(model_dir.resolve()),
        "config_sha256": sha256_file(config_path) if config_path.is_file() else None,
        "tokenizer_config_sha256": (
            sha256_file(tokenizer_path) if tokenizer_path.is_file() else None
        ),
        "weight_index_sha256": sha256_file(index_path) if index_path.is_file() else None,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen3 reader")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"non-empty output directory: {args.output_dir}; pass --resume to reuse caches"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    all_rows, manifest_metadata = load_manifest(args.manifest)
    rows = _select_rows(all_rows, args.max_stages, args.examples_per_stage)
    if not rows:
        raise ValueError("manifest selection is empty after stage/example limits")
    stages = group_by_stage(rows)
    selected_ids = {str(row["sample_id"]) for row in rows}
    locality_panel = manifest_panel(manifest_metadata, "locality", limit=0)
    multihop_panel = manifest_panel(manifest_metadata, "mhop", limit=0)
    if locality_panel is None or multihop_panel is None:
        raise ValueError("RAG requires the frozen v1 locality and multihop panels")
    selected_locality = [
        row for row in locality_panel if str(row["sample_id"]) in selected_ids
    ]
    selected_multihop = [
        row for row in multihop_panel if str(row["sample_id"]) in selected_ids
    ]
    panels = {
        "locality": (
            selected_locality[: args.locality_panel_size]
            if args.locality_panel_size > 0
            else selected_locality
        ),
        "multihop": (
            selected_multihop[: args.multihop_panel_size]
            if args.multihop_panel_size > 0
            else selected_multihop
        ),
    }
    evaluation_panels: dict[str, list[dict[str, Any]]] | None = None
    evaluation_panels_provenance: dict[str, Any] | None = None
    if args.evaluation_panels is not None:
        evaluation_panels, evaluation_panels_provenance = load_evaluation_panels(
            args.evaluation_panels, rows, manifest_metadata,
            manifest_sha256=sha256_file(args.manifest),
        )
        validate_evaluation_panel_resume(args.output_dir, evaluation_panels_provenance)
        panels = {axis: evaluation_panels[axis] for axis in ("locality", "multihop")}
    cache_root = args.embedding_cache_dir or args.output_dir / "cache" / "embeddings"

    encoder_provenance: dict[str, Any] | None = None
    embedding_cache_provenance: dict[str, Any] = {}
    dense_index: TemporalDenseIndex | None = None
    query_embeddings: dict[str, dict[str, torch.Tensor]] = {}
    if args.method in {"dense_rag", "hybrid_rag"}:
        embedding_device = torch.device(args.embedding_device)
        encoder = HFMeanPoolEncoder(
            args.embedding_model,
            device=embedding_device,
            batch_size=args.embedding_batch_size,
            max_length=args.embedding_max_length,
            query_prefix=args.query_prefix,
            document_prefix=args.document_prefix,
            local_files_only=not args.allow_embedding_download,
        )
        encoder_provenance = dict(encoder.provenance)
        documents = [str(row["update"]) if args.recipe == "wikibigedit" else fact_document(row) for row in rows]
        document_embeddings, document_cache = encode_with_shard_cache(
            encoder,
            documents,
            [str(row["sample_id"]) for row in rows],
            cache_root=cache_root,
            namespace="documents",
            shard_size=args.embedding_shard_size,
            kind="document",
        )
        embedding_cache_provenance["documents"] = document_cache
        dense_index = TemporalDenseIndex(rows, document_embeddings)

        all_axis_rows = evaluation_panels if evaluation_panels is not None else {
            "update": valid_axis_rows(rows, "update"),
            "rephrase": valid_axis_rows(rows, "rephrase"),
            "persona": valid_axis_rows(rows, "persona"),
            "locality": valid_axis_rows(panels["locality"], "locality"),
            "multihop": valid_axis_rows(panels["multihop"], "multihop"),
        }
        for axis, axis_rows in all_axis_rows.items():
            question_key, _ = axis_fields(axis)
            texts = [str(row[question_key]) for row in axis_rows]
            identifiers = [f"{axis}:{row['sample_id']}" for row in axis_rows]
            embeddings, cache_info = encode_with_shard_cache(
                encoder,
                texts,
                identifiers,
                cache_root=cache_root,
                namespace=f"queries-{axis}",
                shard_size=args.embedding_shard_size,
                kind="query",
            )
            query_embeddings[axis] = {
                str(row["sample_id"]): embeddings[index]
                for index, row in enumerate(axis_rows)
            }
            embedding_cache_provenance[f"queries_{axis}"] = cache_info
        del encoder
        gc.collect()
        if embedding_device.type == "cuda":
            torch.cuda.empty_cache()

    sparse_index = TemporalBM25Index(rows) if args.method in {"bm25_rag", "hybrid_rag"} else None
    stage_counts = {str(stage): len(stage_rows) for stage, stage_rows in stages.items()}
    index_provenance = {
        "source": "the frozen manifest selection only",
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_protocol_id": manifest_metadata.get("protocol_id"),
        "selected_sample_ids_sha256": _canonical_digest(
            [str(row["sample_id"]) for row in rows]
        ),
        "document_count": len(rows),
        "stage_counts": stage_counts,
        "temporal_policy": TEMPORAL_POLICY,
        "document_template": "<update>" if args.recipe == "wikibigedit" else DOCUMENT_TEMPLATE,
        "search_backend": (
            "exact normalized dot product (cosine), torch CPU"
            if args.method == "dense_rag"
            else ("BM25 + dense RRF" if args.method == "hybrid_rag" else "BM25 Okapi" if args.method == "bm25_rag" else "oracle corresponding-fact lookup")
        ),
    }
    if evaluation_panels is not None:
        index_provenance.update({
            "manifest_protocol_id": manifest_metadata.get("display_protocol_id", manifest_metadata.get("protocol_id")),
            "source_protocol_scope": manifest_metadata.get("protocol_scope"),
        })
    run_identity = {
        "method": args.method,
        "reader": _reader_provenance(args.model_dir),
        "manifest_sha256": index_provenance["manifest_sha256"],
        "selected_sample_ids_sha256": index_provenance["selected_sample_ids_sha256"],
        "top_k": args.top_k if args.method != "oracle_rag" else 1,
        "encoder": encoder_provenance,
        "query_prefix": args.query_prefix if args.method == "dense_rag" else None,
        "document_prefix": args.document_prefix if args.method == "dense_rag" else None,
        "reader_max_length": args.reader_max_length,
        "max_new_tokens": args.max_new_tokens,
        "rag_prompt_template": "<Q: question A: answer contexts> Q: <question> A:" if args.recipe == "wikibigedit" else RAG_PROMPT_TEMPLATE,
        "recipe": args.recipe,
        "rrf_depth": args.rrf_depth if args.method == "hybrid_rag" else None,
        "rrf_constant": args.rrf_constant if args.method == "hybrid_rag" else None,
        "index": index_provenance,
    }
    run_identity = bind_evaluation_panels(run_identity, evaluation_panels_provenance)
    run_fingerprint = _canonical_digest(run_identity)
    config = {
        **_jsonable_args(args),
        "model_dir": str(args.model_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "embedding_cache_dir": str(cache_root.resolve()),
        "method_family": "external retrieval / open-book",
        "method_variant": args.method,
        "manifest_sha256": index_provenance["manifest_sha256"],
        "manifest_protocol_id": manifest_metadata.get("protocol_id"),
        "num_facts": len(rows),
        "stage_counts": stage_counts,
        "locality_panel_size_actual": len(panels["locality"]),
        "multihop_panel_size_actual": len(panels["multihop"]),
        "closed_book_evaluation": False,
        "open_book_evaluation": True,
        "parameter_updates": False,
        "reader_frozen": True,
        "optimizer": None,
        "prompt_template": run_identity["rag_prompt_template"],
        "qa_completion_suffix": "Question: <question>\\nAnswer:",
        "context_order": "rank 1 first" if args.recipe == "wikibigedit" else "descending rank so rank 1 is closest to the question",
        "reader_truncation_side": "left",
        "retrieval_top_k": args.top_k if args.method != "oracle_rag" else 1,
        "embedding_model_provenance": encoder_provenance,
        "embedding_cache_provenance": embedding_cache_provenance,
        "index_provenance": index_provenance,
        "run_fingerprint": run_fingerprint,
        "formal_baseline_eligible": args.method != "oracle_rag",
        "diagnostic_reference": args.method == "oracle_rag",
        "upstream_reference": {
            "repository": "ExplainableML/WikiBigEdit",
            "commit": "66b428560254c38755ab70068cb0b0179aa442e1",
            "implementation": "EasyEdit_Experiments/easyeditor/models/rag/rag_main.py",
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
            "alignment": (
                "Qwen3 shared generated-answer evaluation; paper recipe indexes questions and directly prepends Q/A; "
                "this runner enforces normalized exact cosine and temporal stage masks"
            ),
        },
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "hostname": os.uname().nodename,
    }
    if evaluation_panels is not None:
        assert evaluation_panels_provenance is not None
        config.update({
            "index_count": len(rows),
            "manifest_protocol_id": manifest_metadata.get("display_protocol_id", manifest_metadata.get("protocol_id")),
            "evaluation_scope": evaluation_panels_provenance["evaluation_scope"],
            "evaluation_panels_provenance": evaluation_panels_provenance,
            **sampled_evaluation_counts(evaluation_panels),
        })
    existing_config_path = args.output_dir / "run_config.json"
    if existing_config_path.is_file():
        existing_config = json.loads(existing_config_path.read_text(encoding="utf-8"))
        if existing_config.get("run_fingerprint") != run_fingerprint:
            raise ValueError("--resume output has a different RAG run fingerprint")
    atomic_json(existing_config_path, config)

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    tokenizer, reader = load_reader(args.model_dir, device)
    evaluation_stages = (
        [max(stages)] if args.no_stage_eval else sorted(stages)
    )
    started = time.time()
    resumed_axis_count = 0
    generated_axis_count = 0
    for stage in evaluation_stages:
        checkpoint = f"tau{stage}"
        rows_by_axis = _stage_rows(rows, panels, stage, evaluation_panels=evaluation_panels)
        stage_records: list[dict[str, Any]] = []
        for axis in AXES:
            axis_rows = rows_by_axis[axis]
            if evaluation_panels is not None:
                validate_no_future(axis_rows, stage)
            cached = load_axis_cache(
                args.output_dir,
                checkpoint,
                axis,
                run_fingerprint=run_fingerprint,
                expected_count=len(axis_rows),
            ) if args.resume else None
            if cached is not None:
                if evaluation_panels is not None:
                    if ([record.get("sample_id") for record in cached] != [str(row["sample_id"]) for row in axis_rows]
                            or any(record.get("axis") != axis or record.get("current_stage") != stage
                                   or record.get("fact_stage") != row["stage"]
                                   for record, row in zip(cached, axis_rows))):
                        raise ValueError("sampled evaluation cache does not match its frozen query panel")
                stage_records.extend(cached)
                resumed_axis_count += 1
                continue
            if args.method in {"dense_rag", "hybrid_rag"}:
                assert dense_index is not None
                embeddings = torch.stack(
                    [query_embeddings[axis][str(row["sample_id"])] for row in axis_rows]
                ) if axis_rows else torch.empty((0, dense_index.embeddings.shape[1]))
                hit_lists = dense_index.search(
                    embeddings,
                    evaluation_stage=stage,
                    top_k=max(args.top_k, args.rrf_depth) if args.method == "hybrid_rag" else args.top_k,
                    batch_size=args.retrieval_batch_size,
                ) if axis_rows else []
            elif args.method == "bm25_rag":
                question_key, _ = axis_fields(axis)
                hit_lists = sparse_index.search([str(r[question_key]) for r in axis_rows], evaluation_stage=stage, top_k=args.top_k)
            else:
                hit_lists = oracle_hits(axis_rows, stage)
            if args.method == "hybrid_rag":
                question_key, _ = axis_fields(axis)
                sparse = sparse_index.search([str(r[question_key]) for r in axis_rows], evaluation_stage=stage, top_k=max(args.top_k,args.rrf_depth))
                hit_lists = [reciprocal_rank_fusion(d, b, args.top_k, args.rrf_constant) for d,b in zip(hit_lists,sparse)]
            question_key, _ = axis_fields(axis)
            contexts = [
                [fact_document(hit.row) for hit in row_hits]
                for row_hits in hit_lists
            ]
            prompts = [
                rag_prompt(str(row[question_key]), context)
                for row, context in zip(axis_rows, contexts)
            ]
            if args.recipe == "wikibigedit":
                prompts = [paper_prompt(str(r[question_key]), h) for r,h in zip(axis_rows,hit_lists)]
            predictions = generate_from_prompts(
                reader,
                tokenizer,
                prompts,
                device,
                batch_size=args.eval_batch_size,
                max_length=args.reader_max_length,
                max_new_tokens=args.max_new_tokens,
            )
            records = build_prediction_records(
                axis_rows,
                axis,
                hit_lists,
                predictions,
                checkpoint=checkpoint,
                evaluation_stage=stage,
                method=args.method,
            )
            for record, prompt in zip(records, prompts):
                record["reader_prompt_sha256"] = _canonical_digest(prompt)
            write_axis_cache(
                args.output_dir,
                checkpoint,
                axis,
                records,
                run_fingerprint=run_fingerprint,
            )
            stage_records.extend(records)
            generated_axis_count += 1

        index_count = len(temporal_rows(rows, stage))
        metrics = summarize_rag_records(
            stage_records,
            checkpoint=checkpoint,
            current_stage=stage,
            num_facts=index_count,
            index_count=index_count,
        )
        if evaluation_panels is not None:
            assert evaluation_panels_provenance is not None
            metrics.update({
                "index_count": index_count,
                "evaluation_scope": evaluation_panels_provenance["evaluation_scope"],
                "evaluation_panels_file_sha256": evaluation_panels_provenance["file_sha256"],
                **sampled_evaluation_counts(rows_by_axis),
            })
        metrics["retrieval"]["index_sample_ids_sha256"] = _canonical_digest(
            [str(row["sample_id"]) for row in temporal_rows(rows, stage)]
        )
        metrics["retrieval"]["max_retrieved_fact_stage"] = max(
            (
                int(fact_stage)
                for record in stage_records
                for fact_stage in record["retrieved_fact_stages"]
            ),
            default=None,
        )
        metrics["run_fingerprint"] = run_fingerprint
        atomic_json(
            args.output_dir / "cache" / "evaluation" / checkpoint / "metrics.json",
            metrics,
        )
        _rebuild_canonical_artifacts(args.output_dir, evaluation_stages)

    final_metrics = _rebuild_canonical_artifacts(args.output_dir, evaluation_stages)
    if not final_metrics or int(final_metrics[-1]["current_stage"]) != max(stages):
        raise RuntimeError("final-stage RAG metrics are missing")
    completion = {
        "status": "complete",
        "method": args.method,
        "final_stage": max(stages),
        "num_facts": len(rows),
        "elapsed_seconds": time.time() - started,
        "resumed_axis_count": resumed_axis_count,
        "generated_axis_count": generated_axis_count,
        "parameter_updates": 0,
        "future_leak_count": final_metrics[-1]["retrieval"]["future_leak_count"],
        "run_fingerprint": run_fingerprint,
    }
    if evaluation_panels is not None:
        assert evaluation_panels_provenance is not None
        completion.update({
            "index_count": len(rows),
            "evaluation_scope": evaluation_panels_provenance["evaluation_scope"],
            "evaluation_panels_file_sha256": evaluation_panels_provenance["file_sha256"],
            **sampled_evaluation_counts(evaluation_panels),
            "evaluated_total_across_stages": sum(metric["evaluated_count"] for metric in final_metrics),
        })
    atomic_json(args.output_dir / "completed.json", completion)


if __name__ == "__main__":
    main()
