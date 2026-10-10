#!/usr/bin/env python3
"""Collect checkpoint-specific Qwen3 activation moments for MEMIT.

The formal EasyEdit Qwen2.5 recipe uses the non-centered second moment of the
inputs to every edited MLP ``down_proj``, estimated from 100,000 documents in
``wikimedia/wikipedia`` (``20231101.en``).  These statistics are model- and
module-specific, so this collector writes a separately audited Qwen3 artifact
for each layer.  All requested layers are accumulated during one corpus/model
pass, with a shared resumable checkpoint; completed tensors are still written
as independent atomic layer artifacts.

Hugging Face streaming is supported when the optional ``datasets`` package is
installed.  A local JSONL/JSON/text corpus can be used without that package.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch

from baselines.common import atomic_json, environment_metadata, load_model, load_tokenizer
from baselines.run_memit import get_module, model_stats_key, parse_int_range, tokenizer_padding


COLLECTOR_SCHEMA = "engram-memit-mom2/v1"
PROGRESS_SCHEMA = "engram-memit-mom2-progress/v2"
DEFAULT_HF_DATASET = "wikimedia/wikipedia"
DEFAULT_HF_CONFIG = "20231101.en"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("Qwen3-8B-Base"))
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=Path("/mnt/yy2607004_hs/engram_memit_stats"),
    )
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--layers", type=parse_int_range, default=parse_int_range("4-8"))
    parser.add_argument("--mode", choices=("full", "diagonal"), default="full")
    parser.add_argument("--sample-size", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--document-batch-size", type=int, default=1)
    parser.add_argument(
        "--checkpoint-every-docs",
        type=int,
        default=5000,
        help=(
            "durably checkpoint dense accumulators after this many additional corpus "
            "documents; 0 disables the document-count trigger"
        ),
    )
    parser.add_argument(
        "--checkpoint-every-seconds",
        type=float,
        default=3600.0,
        help=(
            "durably checkpoint dense accumulators after this many seconds; 0 disables "
            "the wall-clock trigger"
        ),
    )
    parser.add_argument(
        "--heartbeat-every-docs",
        type=int,
        default=100,
        help="write a small progress/ETA JSON without copying dense tensors",
    )
    parser.add_argument("--corpus-source", choices=("hf", "local"), default="hf")
    parser.add_argument("--local-corpus", type=Path, default=None)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--hf-dataset", type=str, default=DEFAULT_HF_DATASET)
    parser.add_argument("--hf-config", type=str, default=DEFAULT_HF_CONFIG)
    parser.add_argument("--hf-split", type=str, default="train")
    parser.add_argument("--hf-shuffle-buffer", type=int, default=10000)
    parser.add_argument(
        "--model-fingerprint-mode",
        choices=("content", "metadata"),
        default="content",
        help="content hashes every model artifact; metadata hashes names and sizes",
    )
    parser.add_argument(
        "--local-corpus-fingerprint-mode",
        choices=("content", "metadata"),
        default="content",
        help=(
            "content is required for formal MEMIT provenance; metadata is retained "
            "only for explicitly diagnostic collections"
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_paths(paths: Sequence[Path], root: Path, *, content: bool) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        stat = path.stat()
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        if content:
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def model_artifact_files(model_dir: Path) -> list[Path]:
    patterns = (
        "*.safetensors",
        "*.bin",
        "config.json",
        "generation_config.json",
        "tokenizer*.json",
        "*.model",
        "*.txt",
    )
    files = {
        path
        for pattern in patterns
        for path in model_dir.glob(pattern)
        if path.is_file()
    }
    if not files:
        raise FileNotFoundError(f"no model artifacts found under {model_dir}")
    return sorted(files)


def model_fingerprint(model_dir: Path, mode: str) -> str:
    return sha256_paths(
        model_artifact_files(model_dir),
        model_dir,
        content=mode == "content",
    )


def local_corpus_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    allowed = {".jsonl", ".json", ".txt"}
    files = sorted(candidate for candidate in path.rglob("*") if candidate.suffix in allowed)
    if not files:
        raise FileNotFoundError(f"no JSONL/JSON/text files under {path}")
    return files


def _extract_text(record: Any, text_field: str) -> str | None:
    if isinstance(record, str):
        text = record
    elif isinstance(record, dict):
        text = record.get(text_field)
    else:
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    return text.strip()


def iter_local_documents(path: Path, text_field: str) -> Iterator[str]:
    for file_path in local_corpus_files(path):
        if file_path.suffix == ".jsonl":
            with file_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    line = line.strip()
                    if not line:
                        continue
                    text = _extract_text(json.loads(line), text_field)
                    if text is not None:
                        yield text
        elif file_path.suffix == ".json":
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for key in ("train", "records", "rows", "data"):
                    if isinstance(payload.get(key), list):
                        payload = payload[key]
                        break
            if not isinstance(payload, list):
                raise TypeError(f"JSON corpus must contain a list: {file_path}")
            for record in payload:
                text = _extract_text(record, text_field)
                if text is not None:
                    yield text
        else:
            with file_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        yield line.strip()


def reservoir_indices(length: int, sample_size: int, seed: int) -> list[int]:
    if sample_size <= 0 or sample_size >= length:
        return list(range(length))
    rng = random.Random(seed)
    selected = list(range(sample_size))
    for index in range(sample_size, length):
        replacement = rng.randint(0, index)
        if replacement < sample_size:
            selected[replacement] = index
    return sorted(selected)


def write_jsonl_atomic(path: Path, texts: Iterable[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for text in texts:
            stream.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def prepare_local_cache(args: argparse.Namespace, cache_path: Path) -> tuple[int, dict[str, Any]]:
    assert args.local_corpus is not None
    corpus_root = args.local_corpus.resolve()
    files = local_corpus_files(corpus_root)
    fingerprint_root = corpus_root if corpus_root.is_dir() else corpus_root.parent
    source_fingerprint = sha256_paths(
        files,
        fingerprint_root,
        content=args.local_corpus_fingerprint_mode == "content",
    )
    total = sum(1 for _ in iter_local_documents(corpus_root, args.text_field))
    selected = set(reservoir_indices(total, args.sample_size, args.seed))

    def chosen() -> Iterator[str]:
        for index, text in enumerate(iter_local_documents(corpus_root, args.text_field)):
            if index in selected:
                yield text

    count = write_jsonl_atomic(cache_path, chosen())
    return count, {
        "source": "local",
        "path": str(corpus_root),
        "source_files": [str(path) for path in files],
        "source_fingerprint": source_fingerprint,
        "source_fingerprint_mode": args.local_corpus_fingerprint_mode,
        "available_documents": total,
    }


def prepare_hf_cache(args: argparse.Namespace, cache_path: Path) -> tuple[int, dict[str, Any]]:
    if args.sample_size <= 0:
        raise ValueError("HF streaming collection requires a positive --sample-size")
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face streaming requires `datasets`; install it in the editor environment "
            "or use --corpus-source local"
        ) from error
    dataset = load_dataset(
        args.hf_dataset,
        args.hf_config,
        split=args.hf_split,
        streaming=True,
    )
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.hf_shuffle_buffer)

    def selected() -> Iterator[str]:
        for record in dataset.take(args.sample_size):
            text = _extract_text(record, args.text_field)
            if text is not None:
                yield text

    count = write_jsonl_atomic(cache_path, selected())
    identity = {
        "dataset": args.hf_dataset,
        "config": args.hf_config,
        "split": args.hf_split,
        "seed": args.seed,
        "shuffle_buffer": args.hf_shuffle_buffer,
    }
    source_fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return count, {
        "source": "huggingface_streaming",
        **identity,
        "source_fingerprint": source_fingerprint,
    }


def corpus_cache_identity(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "corpus_source": args.corpus_source,
        "sample_size_requested": args.sample_size,
        "seed": args.seed,
        "text_field": args.text_field,
        "local_corpus": str(args.local_corpus.resolve()) if args.local_corpus else None,
        "hf_dataset": args.hf_dataset,
        "hf_config": args.hf_config,
        "hf_split": args.hf_split,
        "hf_shuffle_buffer": args.hf_shuffle_buffer,
    }


def prepare_corpus_cache(
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[Path, dict[str, Any]]:
    identity = corpus_cache_identity(args)
    identity_hash = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    cache_path = output_root / f"corpus_sample_{identity_hash}.jsonl"
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.exists() and metadata_path.exists() and not args.force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("identity") != identity:
            raise ValueError(f"corpus cache metadata mismatch: {metadata_path}")
        current_cache_sha256 = file_sha256(cache_path)
        if metadata.get("cache_sha256") != current_cache_sha256:
            raise ValueError(
                f"corpus cache content changed since it was built: {cache_path}; "
                "pass --force to rebuild it"
            )
        if args.corpus_source == "local":
            assert args.local_corpus is not None
            corpus_root = args.local_corpus.resolve()
            files = local_corpus_files(corpus_root)
            fingerprint_root = corpus_root if corpus_root.is_dir() else corpus_root.parent
            current_fingerprint = sha256_paths(
                files,
                fingerprint_root,
                content=args.local_corpus_fingerprint_mode == "content",
            )
            cached_fingerprint = metadata.get("source", {}).get("source_fingerprint")
            if current_fingerprint != cached_fingerprint:
                raise ValueError(
                    "local corpus changed since the sample cache was built; pass --force "
                    "to rebuild it"
                )
        return cache_path, metadata
    if args.corpus_source == "local":
        if args.local_corpus is None:
            raise ValueError("--local-corpus is required for --corpus-source local")
        count, source = prepare_local_cache(args, cache_path)
    else:
        count, source = prepare_hf_cache(args, cache_path)
    metadata = {
        "schema_version": "engram-memit-corpus-cache/v1",
        "identity": identity,
        "documents": count,
        "source": source,
        "cache_sha256": file_sha256(cache_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(metadata_path, metadata)
    return cache_path, metadata


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_cached_documents(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            yield str(record["text"])


class StopForward(RuntimeError):
    pass


class SecondMomentAccumulator:
    def __init__(self, dimension: int, mode: str, device: torch.device) -> None:
        self.dimension = dimension
        self.mode = mode
        shape = (dimension, dimension) if mode == "full" else (dimension,)
        self.total = torch.zeros(shape, dtype=torch.float32, device=device)
        self.tokens = 0

    def add(self, values: torch.Tensor) -> None:
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(
                f"expected [tokens, {self.dimension}], got {tuple(values.shape)}"
            )
        values = values.float()
        if self.mode == "full":
            self.total.add_(values.T @ values)
        else:
            self.total.add_((values * values).sum(dim=0))
        self.tokens += int(values.shape[0])

    def moment(self) -> torch.Tensor:
        if self.tokens <= 0:
            raise ValueError("cannot finalize an empty second moment")
        return self.total / self.tokens

    def state_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "mode": self.mode,
            "tokens": self.tokens,
            "total": self.total.detach().cpu(),
        }

    def load_state_dict(self, payload: dict[str, Any], device: torch.device) -> None:
        if int(payload["dimension"]) != self.dimension or payload["mode"] != self.mode:
            raise ValueError("progress accumulator shape/mode mismatch")
        tokens = int(payload["tokens"])
        if tokens < 0:
            raise ValueError("progress accumulator has a negative token count")
        total = payload.get("total")
        expected_shape = (
            (self.dimension, self.dimension)
            if self.mode == "full"
            else (self.dimension,)
        )
        if not isinstance(total, torch.Tensor) or tuple(total.shape) != expected_shape:
            raise ValueError(
                f"progress accumulator shape {getattr(total, 'shape', None)} "
                f"does not match {expected_shape}"
            )
        if not torch.isfinite(total).all():
            raise ValueError("progress accumulator contains non-finite values")
        self.tokens = tokens
        self.total = total.to(device=device, dtype=torch.float32)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def batch_documents(documents: Iterable[str], batch_size: int) -> Iterator[list[str]]:
    if batch_size <= 0:
        raise ValueError("document batch size must be positive")
    batch: list[str] = []
    for document in documents:
        batch.append(document)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def layer_collection_expected(
    *,
    layer: int,
    model_metadata: dict[str, Any],
    corpus_metadata: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Return the stable compatibility/provenance fields for one output layer."""

    return {
        "schema_version": COLLECTOR_SCHEMA,
        "model_fingerprint": model_metadata["model_fingerprint"],
        "module_name": f"model.layers.{layer}.mlp.down_proj",
        "layer": int(layer),
        "mode": args.mode,
        "sample_size_requested": int(args.sample_size),
        "corpus_cache_sha256": corpus_metadata["cache_sha256"],
        "max_length": int(args.max_length),
    }


def validate_reusable_output(
    output_path: Path,
    expected: dict[str, Any],
    *,
    force: bool,
) -> dict[str, Any] | None:
    """Validate an existing atomic tensor+JSON pair, or mark it for recompute."""

    metadata_path = output_path.with_suffix(".json")
    if force or not (output_path.exists() and metadata_path.exists()):
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"existing stats metadata mismatch for layer {expected['layer']}: {mismatches}"
        )
    expected_size = metadata.get("artifact_size_bytes")
    if expected_size is not None and output_path.stat().st_size != int(expected_size):
        return None
    expected_sha256 = metadata.get("artifact_sha256")
    if expected_sha256 is not None:
        if file_sha256(output_path) != expected_sha256:
            return None
    else:
        # Upgrade old v1 artifacts safely. mmap keeps the ~576 MiB tensor storage
        # out of resident RAM while we validate its embedded provenance and shape.
        try:
            payload = torch.load(
                output_path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        except TypeError:
            payload = torch.load(output_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
            return None
        embedded = payload["metadata"]
        if any(embedded.get(key) != value for key, value in expected.items()):
            return None
        tensor_key = "moment" if expected["mode"] == "full" else "diag_moment"
        tensor = payload.get(tensor_key)
        feature_dimension = int(embedded.get("feature_dimension", -1))
        expected_shape = (
            (feature_dimension, feature_dimension)
            if expected["mode"] == "full"
            else (feature_dimension,)
        )
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_shape:
            return None
        if not torch.isfinite(tensor).all():
            return None
        metadata = {
            **embedded,
            "artifact_size_bytes": output_path.stat().st_size,
            "artifact_sha256": file_sha256(output_path),
        }
        atomic_json(metadata_path, metadata)
    return {**metadata, "status": "reused"}


def collection_progress_identity(
    *,
    layers: Sequence[int],
    output_paths: dict[int, Path],
    expected_by_layer: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Identity fields that must match before a dense progress file is resumed."""

    return {
        "schema_version": PROGRESS_SCHEMA,
        "layers": [int(layer) for layer in layers],
        "outputs": {str(layer): str(output_paths[layer].resolve()) for layer in layers},
        "expected_by_layer": {
            str(layer): expected_by_layer[layer] for layer in layers
        },
        # Batch grouping changes floating-point reduction order, so it belongs in
        # the resumability identity even though the mathematical moment is the same.
        "document_batch_size": int(args.document_batch_size),
    }


def collection_progress_path(
    *,
    layers: Sequence[int],
    output_paths: dict[int, Path],
    identity: dict[str, Any],
) -> Path:
    """Use the legacy filename for one layer and a collision-safe group filename."""

    if len(layers) == 1:
        return output_paths[int(layers[0])].with_suffix(".progress.pt")
    parents = {path.resolve().parent for path in output_paths.values()}
    if len(parents) != 1:
        raise ValueError("multi-layer outputs must share one directory")
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    layer_label = "-".join(str(layer) for layer in layers)
    return parents.pop() / f"layers_{layer_label}_{argsafe_mode(identity)}_{digest}.progress.pt"


def argsafe_mode(identity: dict[str, Any]) -> str:
    """Extract a filesystem-safe mode label from a progress identity."""

    expected = identity.get("expected_by_layer", {})
    first = next(iter(expected.values()), {})
    mode = str(first.get("mode", "unknown"))
    return "".join(character for character in mode if character.isalnum() or character in "-_")


def _load_layer_progress(
    progress_path: Path,
    expected: dict[str, Any],
    accumulator: SecondMomentAccumulator,
    device: torch.device,
    document_batch_size: int,
) -> int | None:
    """Load a standalone v1 or v2 layer checkpoint into a grouped run."""

    if not progress_path.exists():
        return None
    progress = torch.load(progress_path, map_location="cpu", weights_only=False)
    if progress.get("schema_version") == PROGRESS_SCHEMA:
        identity = progress.get("identity", {})
        layer_key = str(expected["layer"])
        if identity.get("expected_by_layer", {}).get(layer_key) != expected:
            raise ValueError(f"progress metadata mismatch: {progress_path}")
        if int(identity.get("document_batch_size", -1)) != int(document_batch_size):
            raise ValueError(
                f"progress batch size mismatch: {progress_path}; resume with "
                f"--document-batch-size {identity.get('document_batch_size')}"
            )
        states = progress.get("accumulators", {})
        offsets = progress.get("documents_done", {})
        if layer_key not in states or layer_key not in offsets:
            raise ValueError(f"progress is missing layer {layer_key}: {progress_path}")
        accumulator.load_state_dict(states[layer_key], device)
        return int(offsets[layer_key])
    if progress.get("expected") != expected:
        raise ValueError(f"progress metadata mismatch: {progress_path}")
    accumulator.load_state_dict(progress["accumulator"], device)
    return int(progress["documents_done"])


def _progress_payload(
    *,
    identity: dict[str, Any],
    accumulators: dict[int, SecondMomentAccumulator],
    documents_done: dict[int, int],
    invocation_started: float,
    previous_wall_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": PROGRESS_SCHEMA,
        "identity": identity,
        "documents_done": {
            str(layer): int(documents_done[layer]) for layer in accumulators
        },
        "accumulators": {
            str(layer): accumulator.state_dict()
            for layer, accumulator in accumulators.items()
        },
        "wall_seconds_accumulated": (
            float(previous_wall_seconds) + time.perf_counter() - invocation_started
        ),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }


def _heartbeat_payload(
    *,
    identity: dict[str, Any],
    accumulators: dict[int, SecondMomentAccumulator],
    documents_done: dict[int, int],
    resumed_from: dict[int, int],
    invocation_started: float,
    previous_wall_seconds: float,
    last_durable_documents: dict[int, int],
    corpus_documents: int | None,
) -> dict[str, Any]:
    elapsed = max(time.perf_counter() - invocation_started, 1e-9)
    newly_completed = sum(
        documents_done[layer] - resumed_from[layer] for layer in accumulators
    )
    layer_document_rate = newly_completed / elapsed
    synchronized_rate = layer_document_rate / max(len(accumulators), 1)
    common_done = min(documents_done.values()) if documents_done else 0
    remaining = (
        max(int(corpus_documents) - common_done, 0)
        if corpus_documents is not None
        else None
    )
    return {
        "schema_version": PROGRESS_SCHEMA,
        "status": "running",
        "layers": identity["layers"],
        "documents_done": {
            str(layer): int(documents_done[layer]) for layer in accumulators
        },
        "last_durable_documents": {
            str(layer): int(last_durable_documents.get(layer, 0))
            for layer in accumulators
        },
        "tokens_observed": {
            str(layer): int(accumulator.tokens)
            for layer, accumulator in accumulators.items()
        },
        "wall_seconds_this_invocation": elapsed,
        "wall_seconds_accumulated": float(previous_wall_seconds) + elapsed,
        "effective_synchronized_documents_per_second": synchronized_rate,
        "estimated_remaining_seconds": (
            remaining / synchronized_rate if remaining is not None and synchronized_rate > 0 else None
        ),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _masked_inputs(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.ndim != mask.ndim + 1 or tuple(values.shape[:-1]) != tuple(mask.shape):
        raise ValueError(
            "module inputs and attention mask disagree: "
            f"inputs={tuple(values.shape)}, mask={tuple(mask.shape)}"
        )
    return values[mask]


def collect_layers(
    model: torch.nn.Module,
    tokenizer: Any,
    corpus_cache: Path,
    corpus_metadata: dict[str, Any],
    model_metadata: dict[str, Any],
    layers: Sequence[int],
    output_root: Path,
    device: torch.device,
    args: argparse.Namespace,
    *,
    output_paths: dict[int, Path] | None = None,
) -> list[dict[str, Any]]:
    """Collect several ``down_proj`` moments in one corpus/model pass.

    Dense Qwen3 full moments are roughly 576 MiB per layer.  The original
    implementation replayed the entire 100K-document corpus once per layer
    and rewrote one cumulative matrix every 100 documents.  This collector
    keeps one accumulator per requested unfinished layer, updates all active
    layers during the same forward, stops at the deepest active layer, and
    writes a single infrequent atomic recovery snapshot.

    A resumed group can contain legacy per-layer checkpoints at different
    document offsets.  Layers behind the leading offset are caught up first;
    once offsets coincide, all layers again share each forward.
    """

    ordered_layers = list(dict.fromkeys(int(layer) for layer in layers))
    if not ordered_layers:
        return []
    if int(args.document_batch_size) <= 0:
        raise ValueError("document batch size must be positive")
    if output_paths is None:
        output_paths = {
            layer: output_root / f"layer_{layer}_mom2.pt" for layer in ordered_layers
        }
    else:
        output_paths = {int(layer): Path(path) for layer, path in output_paths.items()}
    missing_paths = [layer for layer in ordered_layers if layer not in output_paths]
    if missing_paths:
        raise ValueError(f"missing output paths for layers: {missing_paths}")

    modules: dict[int, torch.nn.Linear] = {}
    expected_by_layer: dict[int, dict[str, Any]] = {}
    completed: dict[int, dict[str, Any]] = {}
    for layer in ordered_layers:
        module_name = f"model.layers.{layer}.mlp.down_proj"
        module = get_module(model, module_name)
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"{module_name} is not nn.Linear")
        modules[layer] = module
        expected = layer_collection_expected(
            layer=layer,
            model_metadata=model_metadata,
            corpus_metadata=corpus_metadata,
            args=args,
        )
        expected_by_layer[layer] = expected
        reusable = validate_reusable_output(
            output_paths[layer], expected, force=bool(args.force)
        )
        if reusable is not None:
            completed[layer] = reusable

    identity = collection_progress_identity(
        layers=ordered_layers,
        output_paths=output_paths,
        expected_by_layer=expected_by_layer,
        args=args,
    )
    progress_path = collection_progress_path(
        layers=ordered_layers,
        output_paths=output_paths,
        identity=identity,
    )
    heartbeat_path = progress_path.with_suffix(".json")
    pending = [layer for layer in ordered_layers if layer not in completed]
    if not pending:
        # A process can finish every atomic layer artifact and die just before
        # removing the final recovery snapshot.  Do not strand ~3 GiB forever.
        progress_path.unlink(missing_ok=True)
        heartbeat_path.unlink(missing_ok=True)
        for layer in ordered_layers:
            legacy_path = output_paths[layer].with_suffix(".progress.pt")
            if legacy_path != progress_path:
                legacy_path.unlink(missing_ok=True)
        return [completed[layer] for layer in ordered_layers]
    if bool(args.force):
        progress_path.unlink(missing_ok=True)
        heartbeat_path.unlink(missing_ok=True)

    accumulators = {
        layer: SecondMomentAccumulator(modules[layer].in_features, args.mode, device)
        for layer in pending
    }
    documents_done = {layer: 0 for layer in pending}
    previous_wall_seconds = 0.0
    loaded_group_progress = False
    if progress_path.exists() and not bool(args.force):
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("schema_version") == PROGRESS_SCHEMA:
            if progress.get("identity") != identity:
                raise ValueError(f"progress metadata mismatch: {progress_path}")
            states = progress.get("accumulators", {})
            offsets = progress.get("documents_done", {})
            for layer in pending:
                key = str(layer)
                if key not in states or key not in offsets:
                    raise ValueError(
                        f"progress is missing unfinished layer {layer}: {progress_path}"
                    )
                accumulators[layer].load_state_dict(states[key], device)
                documents_done[layer] = int(offsets[key])
            previous_wall_seconds = float(progress.get("wall_seconds_accumulated", 0.0))
            loaded_group_progress = True
        elif len(ordered_layers) == 1:
            layer = pending[0]
            resumed = _load_layer_progress(
                progress_path,
                expected_by_layer[layer],
                accumulators[layer],
                device,
                int(args.document_batch_size),
            )
            if resumed is not None:
                documents_done[layer] = resumed

    if not loaded_group_progress and len(ordered_layers) > 1 and not bool(args.force):
        # Migrate any v1 checkpoints left by the old layer-at-a-time loop.  Their
        # offsets can differ; the forward scheduler below safely catches them up.
        for layer in pending:
            legacy_path = output_paths[layer].with_suffix(".progress.pt")
            resumed = _load_layer_progress(
                legacy_path,
                expected_by_layer[layer],
                accumulators[layer],
                device,
                int(args.document_batch_size),
            )
            if resumed is not None:
                documents_done[layer] = resumed

    corpus_documents_raw = corpus_metadata.get("documents")
    corpus_documents = (
        int(corpus_documents_raw) if corpus_documents_raw is not None else None
    )
    if any(offset < 0 for offset in documents_done.values()):
        raise ValueError("progress contains a negative document offset")
    if corpus_documents is not None and any(
        offset > corpus_documents for offset in documents_done.values()
    ):
        raise ValueError("progress is beyond the end of the cached corpus")
    for layer, offset in documents_done.items():
        tokens = accumulators[layer].tokens
        if (offset == 0) != (tokens == 0):
            raise ValueError(
                f"progress document/token counts disagree for layer {layer}: "
                f"documents={offset}, tokens={tokens}"
            )

    resumed_from = dict(documents_done)
    invocation_started = time.perf_counter()
    last_checkpoint_time = invocation_started
    last_checkpoint_position = min(documents_done.values())
    last_heartbeat_position = last_checkpoint_position
    last_durable_documents = dict(documents_done) if progress_path.exists() else {
        layer: 0 for layer in pending
    }
    checkpoint_every_docs = int(getattr(args, "checkpoint_every_docs", 5000))
    checkpoint_every_seconds = float(
        getattr(args, "checkpoint_every_seconds", 3600.0)
    )
    heartbeat_every_docs = int(getattr(args, "heartbeat_every_docs", 100))

    hook_state: dict[str, Any] = {
        "active": set(),
        "captured": set(),
        "mask": None,
        "stop_layer": None,
    }
    handles = []

    def make_hook(layer: int):
        def capture(_module, inputs):
            if layer not in hook_state["active"]:
                return None
            if layer in hook_state["captured"]:
                raise RuntimeError(f"captured layer {layer} more than once in one forward")
            mask = hook_state["mask"]
            if not isinstance(mask, torch.Tensor):
                raise RuntimeError("attention mask was not installed before forward")
            accumulators[layer].add(_masked_inputs(inputs[0], mask))
            hook_state["captured"].add(layer)
            if layer == hook_state["stop_layer"]:
                raise StopForward()
            return None

        return capture

    for layer in pending:
        handles.append(modules[layer].register_forward_pre_hook(make_hook(layer)))

    documents = iter_cached_documents(corpus_cache)
    position = min(documents_done.values())
    for _ in range(position):
        try:
            next(documents)
        except StopIteration as error:
            raise ValueError("progress is beyond the end of the cached corpus") from error

    try:
        while True:
            active = [layer for layer in pending if documents_done[layer] == position]
            if not active:
                raise RuntimeError(
                    f"no layer is ready at corpus offset {position}: {documents_done}"
                )
            future_offsets = [
                offset for offset in documents_done.values() if offset > position
            ]
            batch_limit = int(args.document_batch_size)
            if future_offsets:
                batch_limit = min(batch_limit, min(future_offsets) - position)
            batch = list(itertools.islice(documents, batch_limit))
            if not batch:
                if future_offsets:
                    raise ValueError("progress is beyond the end of the cached corpus")
                break

            with tokenizer_padding(tokenizer, "right"):
                encoded = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    add_special_tokens=True,
                )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            mask = encoded.get("attention_mask")
            if not isinstance(mask, torch.Tensor):
                raise ValueError("tokenizer output must include an attention_mask tensor")
            hook_state["active"] = set(active)
            hook_state["captured"] = set()
            hook_state["mask"] = mask.bool()
            hook_state["stop_layer"] = max(active)
            try:
                with torch.inference_mode():
                    model(**encoded, use_cache=False)
            except StopForward:
                pass
            if hook_state["captured"] != set(active):
                missing = set(active) - hook_state["captured"]
                raise RuntimeError(f"failed to capture layers: {sorted(missing)}")

            for layer in active:
                documents_done[layer] += len(batch)
            position += len(batch)
            common_done = min(documents_done.values())
            now = time.perf_counter()
            documents_due = (
                checkpoint_every_docs > 0
                and common_done - last_checkpoint_position >= checkpoint_every_docs
            )
            seconds_due = (
                checkpoint_every_seconds > 0
                and now - last_checkpoint_time >= checkpoint_every_seconds
            )
            if documents_due or seconds_due:
                atomic_torch_save(
                    progress_path,
                    _progress_payload(
                        identity=identity,
                        accumulators=accumulators,
                        documents_done=documents_done,
                        invocation_started=invocation_started,
                        previous_wall_seconds=previous_wall_seconds,
                    ),
                )
                last_checkpoint_time = time.perf_counter()
                last_checkpoint_position = common_done
                last_durable_documents = dict(documents_done)

            if (
                heartbeat_every_docs > 0
                and common_done - last_heartbeat_position >= heartbeat_every_docs
            ):
                atomic_json(
                    heartbeat_path,
                    _heartbeat_payload(
                        identity=identity,
                        accumulators=accumulators,
                        documents_done=documents_done,
                        resumed_from=resumed_from,
                        invocation_started=invocation_started,
                        previous_wall_seconds=previous_wall_seconds,
                        last_durable_documents=last_durable_documents,
                        corpus_documents=corpus_documents,
                    ),
                )
                last_heartbeat_position = common_done
    finally:
        for handle in handles:
            handle.remove()

    if len(set(documents_done.values())) != 1:
        raise RuntimeError(f"layers ended at different corpus offsets: {documents_done}")

    compute_elapsed = time.perf_counter() - invocation_started
    # Persist the completed raw sums before writing several independent final
    # artifacts.  If finalization is interrupted, restart only re-emits the
    # missing atomic layer files and does not replay the corpus.
    final_documents = next(iter(documents_done.values()))
    if any(
        last_durable_documents.get(layer, 0) != final_documents for layer in pending
    ):
        atomic_torch_save(
            progress_path,
            _progress_payload(
                identity=identity,
                accumulators=accumulators,
                documents_done=documents_done,
                invocation_started=invocation_started,
                previous_wall_seconds=previous_wall_seconds,
            ),
        )
        last_durable_documents = dict(documents_done)

    elapsed = time.perf_counter() - invocation_started
    computed: dict[int, dict[str, Any]] = {}
    for layer in pending:
        accumulator = accumulators[layer]
        if accumulator.tokens <= 0:
            raise ValueError("cannot finalize an empty second moment")
        # Divide after the device-to-host copy to avoid allocating another
        # 576 MiB dense tensor on the GPU for Qwen3-8B.
        moment = accumulator.total.detach().cpu()
        moment.div_(accumulator.tokens)
        if not torch.isfinite(moment).all():
            raise ValueError(f"layer {layer} second moment contains non-finite values")
        metadata = {
            **expected_by_layer[layer],
            **model_metadata,
            "kind": (
                "full_second_moment"
                if args.mode == "full"
                else "diagonal_second_moment"
            ),
            "formal_memit_eligible": args.mode == "full",
            "feature_dimension": modules[layer].in_features,
            "documents_observed": documents_done[layer],
            "tokens_observed": accumulator.tokens,
            "moment_dtype": str(moment.dtype),
            "corpus": corpus_metadata,
            "collection_strategy": "single_pass_multi_layer",
            "layers_collected_together": ordered_layers,
            "document_batch_size": int(args.document_batch_size),
            "checkpoint_every_docs": checkpoint_every_docs,
            "checkpoint_every_seconds": checkpoint_every_seconds,
            "documents_resumed_from": resumed_from[layer],
            "wall_seconds_previous_invocations": previous_wall_seconds,
            "compute_wall_seconds_this_invocation": compute_elapsed,
            "effective_synchronized_documents_per_second": (
                (final_documents - min(resumed_from.values())) / compute_elapsed
                if compute_elapsed > 0
                else None
            ),
            "wall_seconds_this_invocation": elapsed,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        tensor_key = "moment" if args.mode == "full" else "diag_moment"
        atomic_torch_save(
            output_paths[layer],
            {
                tensor_key: moment,
                "metadata": metadata,
            },
        )
        metadata = {
            **metadata,
            "artifact_size_bytes": output_paths[layer].stat().st_size,
            "artifact_sha256": file_sha256(output_paths[layer]),
        }
        atomic_json(output_paths[layer].with_suffix(".json"), metadata)
        computed[layer] = {**metadata, "status": "computed"}
        del moment

    progress_path.unlink(missing_ok=True)
    heartbeat_path.unlink(missing_ok=True)
    for layer in pending:
        legacy_path = output_paths[layer].with_suffix(".progress.pt")
        if legacy_path != progress_path:
            legacy_path.unlink(missing_ok=True)
    return [completed.get(layer, computed.get(layer)) for layer in ordered_layers]


def collect_layer(
    model: torch.nn.Module,
    tokenizer: Any,
    corpus_cache: Path,
    corpus_metadata: dict[str, Any],
    model_metadata: dict[str, Any],
    layer: int,
    output_path: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return collect_layers(
        model,
        tokenizer,
        corpus_cache,
        corpus_metadata,
        model_metadata,
        [layer],
        output_path.parent,
        device,
        args,
        output_paths={int(layer): output_path},
    )[0]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive for reproducible formal statistics")
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    if args.document_batch_size <= 0:
        raise ValueError("--document-batch-size must be positive")
    if args.checkpoint_every_docs < 0 or args.checkpoint_every_seconds < 0:
        raise ValueError("checkpoint intervals must be non-negative")
    if args.heartbeat_every_docs < 0:
        raise ValueError("--heartbeat-every-docs must be non-negative")
    args.model_dir = args.model_dir.resolve()
    output_root = args.stats_dir.resolve() / model_stats_key(args.model_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    corpus_cache, corpus_metadata = prepare_corpus_cache(args, output_root)
    fingerprint_started = time.perf_counter()
    fingerprint = model_fingerprint(args.model_dir, args.model_fingerprint_mode)
    config_path = args.model_dir / "config.json"
    model_metadata = {
        "model_dir": str(args.model_dir),
        "model_name": model_stats_key(args.model_dir),
        "model_fingerprint": fingerprint,
        "model_fingerprint_mode": args.model_fingerprint_mode,
        "model_config_sha256": file_sha256(config_path),
        "model_fingerprint_wall_seconds": time.perf_counter() - fingerprint_started,
    }
    atomic_json(output_root / "model_identity.json", model_metadata)

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    tokenizer = load_tokenizer(args.model_dir)
    model = load_model(args.model_dir, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    if getattr(model.config, "model_type", None) != "qwen3":
        raise ValueError(f"collector is scoped to Qwen3, got {model.config.model_type!r}")

    for layer in args.layers:
        if layer < 0 or layer >= int(model.config.num_hidden_layers):
            raise ValueError(f"layer {layer} is outside this Qwen3 checkpoint")
    atomic_json(
        output_root / "collection_summary.json",
        {
            "schema_version": COLLECTOR_SCHEMA,
            "status": "running",
            "collection_strategy": "single_pass_multi_layer",
            "layers_requested": list(args.layers),
            "results": [],
            **environment_metadata(),
        },
    )
    results = collect_layers(
        model,
        tokenizer,
        corpus_cache,
        corpus_metadata,
        model_metadata,
        args.layers,
        output_root,
        device,
        args,
    )

    atomic_json(
        output_root / "collection_summary.json",
        {
            "schema_version": COLLECTOR_SCHEMA,
            "status": "complete",
            "collection_strategy": "single_pass_multi_layer",
            "layers_requested": list(args.layers),
            "layers_finished": [entry["layer"] for entry in results],
            "results": results,
            **environment_metadata(),
        },
    )


if __name__ == "__main__":
    main()
