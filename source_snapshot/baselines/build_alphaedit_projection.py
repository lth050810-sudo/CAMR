#!/usr/bin/env python3
"""Collect Qwen3 moments and build provenance-checked AlphaEdit null spaces.

The official AlphaEdit/EasyEdit Qwen2.5 preset derives each projector from a
100,000-document English-Wikipedia input second moment.  Neither that moment
nor its null-space projector is transferable to Qwen3.  This command reuses
the resumable MEMIT moment collector, then atomically writes a Qwen3-specific
orthonormal null basis for every requested ``mlp.down_proj`` layer.

Hugging Face streaming and local JSONL/JSON/text corpora are supported.  A
local corpus is diagnostic by default; ``--publication-corpus`` is an
explicit provenance assertion, not an automatic quality judgment.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from baselines.alphaedit_core import (
    EASYEDIT_COMMIT,
    EASYEDIT_REPOSITORY,
    LICENSE_PROVENANCE,
    PROJECTION_SCHEMA,
    UPSTREAM_COMMIT,
    UPSTREAM_QWEN25_PRESET,
    UPSTREAM_REPOSITORY,
    compute_nullspace_basis,
    load_projection_payload,
    module_name_for_layer,
    orthogonality_probe_error,
    projection_filename,
)
from baselines.collect_memit_stats import (
    COLLECTOR_SCHEMA,
    atomic_torch_save,
    collect_layers,
    file_sha256,
    model_fingerprint,
    prepare_corpus_cache,
)
from baselines.common import atomic_json, environment_metadata, load_model, load_tokenizer
from baselines.run_memit import get_module, load_moment_file, model_stats_key, parse_int_range


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("Qwen3-8B-Base"))
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=Path("/mnt/yy2607004_hs/engram_alphaedit_stats"),
    )
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--layers", type=parse_int_range, default=parse_int_range("4-8"))
    parser.add_argument("--sample-size", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--document-batch-size", type=int, default=1)
    parser.add_argument("--checkpoint-every-docs", type=int, default=5000)
    parser.add_argument("--checkpoint-every-seconds", type=float, default=3600.0)
    parser.add_argument("--heartbeat-every-docs", type=int, default=100)
    parser.add_argument("--nullspace-threshold", type=float, default=2e-2)
    parser.add_argument(
        "--decomposition-device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="device used for the symmetric eigendecomposition",
    )
    parser.add_argument("--corpus-source", choices=("hf", "local"), default="hf")
    parser.add_argument("--local-corpus", type=Path, default=None)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--hf-dataset", type=str, default="wikimedia/wikipedia")
    parser.add_argument("--hf-config", type=str, default="20231101.en")
    parser.add_argument("--hf-split", type=str, default="train")
    parser.add_argument("--hf-shuffle-buffer", type=int, default=10000)
    parser.add_argument(
        "--model-fingerprint-mode", choices=("content", "metadata"), default="content"
    )
    parser.add_argument(
        "--local-corpus-fingerprint-mode",
        choices=("content", "metadata"),
        default="content",
    )
    parser.add_argument(
        "--publication-corpus",
        action="store_true",
        help="assert that a custom/local source is suitable old-task Wikipedia data",
    )
    parser.add_argument(
        "--preset-label",
        default="qwen25-exact-layer-indices-final-block-transfer-untuned",
    )
    parser.add_argument("--force-stats", action="store_true")
    parser.add_argument("--force-projection", action="store_true")
    return parser.parse_args()


def formal_corpus_eligible(args: argparse.Namespace, corpus: dict[str, Any]) -> bool:
    source = corpus.get("source") or {}
    source_fingerprint = source.get("source_fingerprint")
    canonical_hf = (
        source.get("source") == "huggingface_streaming"
        and source.get("dataset") == "wikimedia/wikipedia"
        and source.get("config") == "20231101.en"
        and source.get("split") == "train"
        and isinstance(source_fingerprint, str)
        and bool(source_fingerprint)
    )
    asserted_local = (
        args.publication_corpus
        and source.get("source") == "local"
        and source.get("source_fingerprint_mode") == "content"
        and isinstance(source_fingerprint, str)
        and bool(source_fingerprint)
    )
    return bool(
        (canonical_hf or asserted_local)
        and int(corpus.get("documents", 0)) >= 100000
        and args.sample_size >= 100000
    )


def projection_metadata(
    *,
    args: argparse.Namespace,
    layer: int,
    moment_path: Path,
    moment_metadata: dict[str, Any],
    model_metadata: dict[str, Any],
    eigenvalues: torch.Tensor,
    basis: torch.Tensor,
    elapsed: float,
    formal_eligible: bool,
) -> dict[str, Any]:
    return {
        "schema_version": PROJECTION_SCHEMA,
        "method": "AlphaEdit",
        "representation": "orthonormal_nullspace_basis",
        "projector_definition": "basis @ basis.T",
        "selection_rule": "abs(eigenvalue) < absolute nullspace_threshold",
        "decomposition": "torch.linalg.eigh(symmetrized_second_moment)",
        "model_type": "qwen3",
        **model_metadata,
        "layer": int(layer),
        "module_name": module_name_for_layer(layer),
        "feature_dimension": int(basis.shape[0]),
        "nullity": int(basis.shape[1]),
        "basis_orthogonality_probe_max_error": orthogonality_probe_error(basis),
        "nullspace_threshold": float(args.nullspace_threshold),
        "sample_size_requested": int(args.sample_size),
        "eigenvalue_min": float(eigenvalues.min()),
        "eigenvalue_max": float(eigenvalues.max()),
        "selected_eigenvalue_max": (
            float(
                eigenvalues[eigenvalues.abs() < args.nullspace_threshold]
                .abs()
                .max()
            )
            if basis.shape[1]
            else None
        ),
        "source_stats_schema": moment_metadata.get("schema_version"),
        "statistics_definition": "non-centered token-level second moment E[k k^T]",
        "stats_max_length": moment_metadata.get("max_length"),
        "source_moment_path": str(moment_path.resolve()),
        "source_moment_sha256": (
            moment_metadata.get("artifact_sha256") or file_sha256(moment_path)
        ),
        "corpus": moment_metadata.get("corpus"),
        "publication_corpus_asserted": bool(args.publication_corpus),
        "corpus_source_fingerprint_mode": (
            (moment_metadata.get("corpus") or {}).get("source", {}).get(
                "source_fingerprint_mode"
            )
        ),
        "documents_observed": moment_metadata.get("documents_observed"),
        "tokens_observed": moment_metadata.get("tokens_observed"),
        "formal_baseline_eligible": bool(formal_eligible),
        "qualification": (
            "Qwen3-specific full 100K English-Wikipedia moment"
            if formal_eligible
            else "diagnostic/custom projection; not a formal AlphaEdit baseline cache"
        ),
        "preset_label": args.preset_label,
        "qwen3_layer_selection_status": "untuned transfer; no official Qwen3-8B preset",
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "easyedit_repository": EASYEDIT_REPOSITORY,
        "easyedit_commit": EASYEDIT_COMMIT,
        "upstream_qwen25_preset": UPSTREAM_QWEN25_PRESET,
        "license_provenance": LICENSE_PROVENANCE,
        "decomposition_device": args.decomposition_device,
        "sampling_note": (
            "HF streaming buffer shuffle is deterministic from the recorded seed and sample "
            "hash but is not claimed sample-identical to EasyEdit's in-memory shuffle"
        ),
        "decomposition_wall_seconds": elapsed,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to collect Qwen3 AlphaEdit activations")
    if args.sample_size <= 0 or args.max_length <= 0 or args.document_batch_size <= 0:
        raise ValueError("sample size, max length, and document batch size must be positive")
    if args.checkpoint_every_docs < 0 or args.checkpoint_every_seconds < 0:
        raise ValueError("checkpoint intervals must be non-negative")
    if args.heartbeat_every_docs < 0:
        raise ValueError("heartbeat interval must be non-negative")
    if args.nullspace_threshold < 0:
        raise ValueError("null-space threshold must be non-negative")
    if args.corpus_source == "local" and args.local_corpus is None:
        raise ValueError("--local-corpus is required for --corpus-source local")

    args.model_dir = args.model_dir.resolve()
    output_root = args.stats_dir.resolve() / model_stats_key(args.model_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Helpers imported from collect_memit_stats intentionally consume this
    # compatible namespace.  AlphaEdit always needs the full second moment.
    args.mode = "full"
    args.force = bool(args.force_stats)
    corpus_cache, corpus = prepare_corpus_cache(args, output_root)
    fingerprint_started = time.perf_counter()
    fingerprint = model_fingerprint(args.model_dir, args.model_fingerprint_mode)
    config_path = args.model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_metadata = {
        "model_dir": str(args.model_dir),
        "model_name": model_stats_key(args.model_dir),
        "model_fingerprint": fingerprint,
        "model_fingerprint_mode": args.model_fingerprint_mode,
        "model_config_sha256": file_sha256(config_path),
        "model_fingerprint_wall_seconds": time.perf_counter() - fingerprint_started,
        "num_hidden_layers": config.get("num_hidden_layers"),
        "hidden_size": config.get("hidden_size"),
        "intermediate_size": config.get("intermediate_size"),
    }
    if config.get("model_type") != "qwen3":
        raise ValueError(f"AlphaEdit builder is scoped to Qwen3, got {config.get('model_type')!r}")
    atomic_json(output_root / "model_identity.json", model_metadata)

    torch.cuda.set_device(args.gpu)
    model_device = torch.device(f"cuda:{args.gpu}")
    decomposition_device = (
        model_device if args.decomposition_device == "cuda" else torch.device("cpu")
    )
    tokenizer = load_tokenizer(args.model_dir)
    model = load_model(args.model_dir, model_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    results: list[dict[str, Any]] = []
    corpus_is_formal = formal_corpus_eligible(args, corpus)
    for layer in args.layers:
        if layer < 0 or layer >= int(model.config.num_hidden_layers):
            raise ValueError(f"layer {layer} is outside this Qwen3 checkpoint")
    moment_results = collect_layers(
        model,
        tokenizer,
        corpus_cache,
        corpus,
        model_metadata,
        args.layers,
        output_root,
        model_device,
        args,
    )
    moment_results_by_layer = {int(entry["layer"]): entry for entry in moment_results}
    if model_device.type == "cuda":
        torch.cuda.empty_cache()

    for layer in args.layers:
        moment_path = output_root / f"layer_{layer}_mom2.pt"
        moment_metadata = moment_results_by_layer[int(layer)]
        moment, loaded_metadata = load_moment_file(moment_path)
        provenance_keys = (
            "schema_version",
            "model_fingerprint",
            "model_config_sha256",
            "module_name",
            "layer",
            "mode",
            "sample_size_requested",
            "corpus_cache_sha256",
            "max_length",
            "feature_dimension",
            "documents_observed",
            "tokens_observed",
        )
        mismatches = {
            key: (moment_metadata.get(key), loaded_metadata.get(key))
            for key in provenance_keys
            if moment_metadata.get(key) != loaded_metadata.get(key)
        }
        if mismatches:
            raise ValueError(
                f"moment sidecar/payload metadata mismatch for layer {layer}: {mismatches}"
            )
        moment_metadata = {**moment_metadata, **loaded_metadata}
        if moment.ndim != 2:
            raise ValueError("AlphaEdit requires a full second moment, not a diagonal proxy")
        rewrite_module = get_module(model, module_name_for_layer(layer))
        expected_shape = (rewrite_module.in_features, rewrite_module.in_features)
        if tuple(moment.shape) != expected_shape:
            raise ValueError(
                f"AlphaEdit moment shape {tuple(moment.shape)} does not match "
                f"{module_name_for_layer(layer)} input shape {expected_shape}"
            )
        projection_path = output_root / projection_filename(layer)
        if projection_path.exists() and not args.force_projection:
            basis, metadata = load_projection_payload(
                projection_path,
                layer=layer,
                module_name=module_name_for_layer(layer),
                input_dimension=moment.shape[0],
                nullspace_threshold=args.nullspace_threshold,
                model_fingerprint=fingerprint,
                model_config_sha256=model_metadata["model_config_sha256"],
                sample_size=args.sample_size,
                allow_diagnostic=True,
            )
            source_moment_sha256 = (
                moment_metadata.get("artifact_sha256") or file_sha256(moment_path)
            )
            if metadata.get("source_moment_sha256") != source_moment_sha256:
                raise ValueError(
                    f"projection source moment changed for layer {layer}; "
                    "pass --force-projection to rebuild"
                )
            atomic_json(projection_path.with_suffix(".json"), metadata)
            results.append({**metadata, "status": "reused"})
            del basis, moment
        else:
            started = time.perf_counter()
            eigenvalues, basis = compute_nullspace_basis(
                moment.to(decomposition_device), args.nullspace_threshold
            )
            elapsed = time.perf_counter() - started
            formal = bool(
                corpus_is_formal
                and moment_metadata.get("kind") == "full_second_moment"
                and int(moment_metadata.get("documents_observed", 0)) >= args.sample_size
                and args.model_fingerprint_mode == "content"
            )
            metadata = projection_metadata(
                args=args,
                layer=layer,
                moment_path=moment_path,
                moment_metadata=moment_metadata,
                model_metadata=model_metadata,
                eigenvalues=eigenvalues,
                basis=basis,
                elapsed=elapsed,
                formal_eligible=formal,
            )
            atomic_torch_save(
                projection_path,
                {
                    "basis": basis.detach().float().cpu(),
                    "eigenvalues": eigenvalues.detach().float().cpu(),
                    "metadata": metadata,
                },
            )
            atomic_json(projection_path.with_suffix(".json"), metadata)
            results.append({**metadata, "status": "computed"})
            del eigenvalues, basis, moment
            if decomposition_device.type == "cuda":
                torch.cuda.empty_cache()
        atomic_json(
            output_root / "alphaedit_projection_summary.json",
            {
                "schema_version": PROJECTION_SCHEMA,
                "status": "running",
                "layers_requested": list(args.layers),
                "layers_finished": [entry["layer"] for entry in results],
                "results": results,
                **environment_metadata(),
            },
        )

    atomic_json(
        output_root / "alphaedit_projection_summary.json",
        {
            "schema_version": PROJECTION_SCHEMA,
            "source_stats_schema": COLLECTOR_SCHEMA,
            "status": "complete",
            "layers_requested": list(args.layers),
            "layers_finished": [entry["layer"] for entry in results],
            "formal_baseline_eligible": all(
                entry.get("formal_baseline_eligible") is True for entry in results
            ),
            "results": results,
            **environment_metadata(),
        },
    )


if __name__ == "__main__":
    main()
