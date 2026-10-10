#!/usr/bin/env python3
"""Generate Qwen3-specific KFAC and projection caches for CrispEdit.

The official Qwen2.5 cache is architecture- and weight-dependent and must not
be reused for Qwen3.  This script stores enough provenance next to every cache
to reject model, layer, corpus, or energy-threshold mismatches later.

For a publication run, use the upstream corpus choice (English Wikipedia
20220301) and 1,000 documents.  ``--manifest-corpus`` exists only for a tiny
offline diagnostic and is explicitly marked non-publication in metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from baselines.common import atomic_json, load_manifest, load_model, load_tokenizer, qa_prompt
from baselines.crispedit_core import (
    PROJECTION_FORMAT,
    STATS_FORMAT,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    collect_kfac_stats,
    low_curvature_projection,
    model_identity,
    module_name_for_layer,
    parse_layers,
    projection_filename,
    resolve_modules,
    stats_filename,
    validate_projection_payload,
    validate_stats_payload,
)


class TextDataset(Dataset):
    def __init__(self, texts: Sequence[str], tokenizer: Any, max_length: int) -> None:
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": torch.tensor(encoded.input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(encoded.input_ids), dtype=torch.long),
        }


def collate_texts(examples: Sequence[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(example["input_ids"].numel() for example in examples)
    input_ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), width), dtype=torch.long)
    for index, example in enumerate(examples):
        length = example["input_ids"].numel()
        input_ids[index, :length] = example["input_ids"]
        attention_mask[index, :length] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("Qwen3-8B-Base"))
    parser.add_argument("--stats-dir", type=Path, required=True)
    parser.add_argument(
        "--layers",
        type=str,
        required=True,
        help="comma-separated Qwen3 layer indices; do not assume Qwen2.5 layers transfer",
    )
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--energy-threshold", type=float, default=0.99)
    parser.add_argument(
        "--projection-storage-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16",
        help="upstream converts eigenvectors to the edited model dtype before use",
    )
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--preset-label",
        type=str,
        default="unvalidated-qwen3-layer-selection",
        help="audit label only; this script never declares a transfer preset validated",
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hf-dataset", type=str)
    source.add_argument("--corpus-jsonl", type=Path)
    source.add_argument("--corpus-json", type=Path)
    source.add_argument("--manifest-corpus", type=Path)
    parser.add_argument("--hf-config", type=str, default=None)
    parser.add_argument("--hf-split", type=str, default="train")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--corpus-audit", type=Path, help="Hash-bound cross-source audit for a local dated Wikipedia clone")
    return parser.parse_args()


def _json_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sample_texts(texts: Iterable[str], sample_size: int, seed: int) -> list[str]:
    cleaned = [str(text).strip() for text in texts if isinstance(text, str) and text.strip()]
    if not cleaned:
        raise ValueError("corpus has no non-empty text records")
    rng = random.Random(seed)
    if sample_size > len(cleaned):
        raise ValueError(f"requested {sample_size} documents from a corpus of {len(cleaned)}")
    indices = rng.sample(range(len(cleaned)), sample_size)
    return [cleaned[index] for index in indices]


def load_corpus(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    if args.manifest_corpus is not None:
        rows, metadata = load_manifest(args.manifest_corpus)
        texts = [f"{qa_prompt(str(row['update']))} {str(row['ans']).strip()}" for row in rows]
        sampled = _sample_texts(texts, args.sample_size, args.seed)
        return sampled, {
            "kind": "wikibigedit-manifest-diagnostic",
            "publication_ready": False,
            "warning": "Edit facts are not the upstream Wikipedia old-task corpus.",
            "path": str(args.manifest_corpus.resolve()),
            "protocol_id": metadata.get("protocol_id"),
            "selection_sha256": metadata.get("selection_sha256"),
        }

    if args.corpus_jsonl is not None:
        records = []
        with args.corpus_jsonl.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if args.text_field not in record:
                    raise KeyError(f"line {line_number} has no {args.text_field!r} field")
                records.append(record[args.text_field])
        sampled = _sample_texts(records, args.sample_size, args.seed)
        metadata = {
            "kind": "local-jsonl",
            "publication_ready": None,
            "path": str(args.corpus_jsonl.resolve()),
            "text_field": args.text_field,
        }
        audit_path = getattr(args, "corpus_audit", None)
        if audit_path is not None:
            audit = json.loads(audit_path.read_text())
            digest = hashlib.sha256(args.corpus_jsonl.read_bytes()).hexdigest()
            if audit.get("file_sha256", {}).get(args.corpus_jsonl.name) != digest:
                raise ValueError("Local corpus does not match the cross-source audit hash")
            if audit.get("status") != "candidate_downloaded_and_cross_checked" or audit.get("mismatches") != []:
                raise ValueError("Corpus cross-source verification did not pass")
            if audit.get("overlap_articles", 0) < 10 or not audit.get("source_revision_unchanged_after_download"):
                raise ValueError("Insufficient independent dated-corpus evidence")
            sources = audit.get("sources", {})
            normalization = audit.get("comparison_normalization")
            if audit.get("raw_byte_mismatches"):
                if (not normalization or normalization.get("rule") != "first10k.text.replace('\\n', ' ') == validation.text"
                        or normalization.get("matching_articles") != audit["overlap_articles"]):
                    raise ValueError("Unexplained cross-source byte differences")
            if sources.get("validation", {}).get("dataset") != "closji/wikipedia-20220301.en-0.005-validation":
                raise ValueError("Unexpected dated-corpus source")
            if sources.get("first10k", {}).get("dataset") != "kaitchup/wikipedia-20220301-en-sample-10k":
                raise ValueError("Unexpected independent comparison source")
            metadata.update({
                "kind": "verified-Wikipedia20220301-clone-subset",
                "publication_ready": False,
                "verified_dated_corpus_transfer": True,
                "source_config": "20220301.en",
                "source_revision": sources["validation"]["revision"],
                "source_subset": f"{args.corpus_jsonl.name}; parent clone validation contains 32,293 original articles",
                "original_author_sample_reproduced": False,
                "reproduction_exact": False,
                "corpus_transfer_label": "20220301 sampled-clone, whitespace-normalized",
                "source_audit": str(audit_path.resolve()),
                "source_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                "source_file_sha256": digest,
                "cross_source_overlap_articles": audit["overlap_articles"],
                "comparison_normalization": normalization,
                "qualification": "Date-matched corpus transfer; retain clone/subset/sampling and newline-to-space disclosure",
            })
        return sampled, metadata

    if args.corpus_json is not None:
        payload = json.loads(args.corpus_json.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else payload.get("records")
        if not isinstance(records, list):
            raise TypeError("--corpus-json must contain a list or an object with records")
        texts = [record[args.text_field] if isinstance(record, dict) else record for record in records]
        sampled = _sample_texts(texts, args.sample_size, args.seed)
        return sampled, {
            "kind": "local-json",
            "publication_ready": None,
            "path": str(args.corpus_json.resolve()),
            "text_field": args.text_field,
        }

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face datasets is required for --hf-dataset; install it in the editor venv "
            "or pass a local JSON/JSONL corpus"
        ) from error
    dataset = load_dataset(
        args.hf_dataset,
        args.hf_config,
        split=args.hf_split,
    )
    if args.text_field not in dataset.column_names:
        raise KeyError(f"dataset columns {dataset.column_names} have no {args.text_field!r}")
    if args.sample_size > len(dataset):
        raise ValueError(f"requested {args.sample_size} documents from dataset size {len(dataset)}")
    generator = random.Random(args.seed)
    indices = generator.sample(range(len(dataset)), args.sample_size)
    sampled = [str(dataset[index][args.text_field]).strip() for index in indices]
    sampled = [text for text in sampled if text]
    if len(sampled) != args.sample_size:
        raise ValueError("the sampled HF corpus contains empty records; use a pre-cleaned local corpus")
    return sampled, {
        "kind": "huggingface-dataset",
        "publication_ready": (
            args.hf_dataset == "wikipedia" and args.hf_config == "20220301.en"
        ),
        "dataset": args.hf_dataset,
        "config": args.hf_config,
        "split": args.hf_split,
        "text_field": args.text_field,
    }


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3 KFAC collection")
    if args.sample_size <= 0 or args.batch_size <= 0 or args.max_length < 2:
        raise ValueError("sample size/batch size must be positive and max length must be >=2")
    layers = parse_layers(args.layers)
    identity = model_identity(args.model_dir)
    num_hidden_layers = int(identity["num_hidden_layers"])
    invalid = [layer for layer in layers if layer >= num_hidden_layers]
    if invalid:
        raise ValueError(f"layers outside Qwen3 depth {num_hidden_layers}: {invalid}")
    args.stats_dir.mkdir(parents=True, exist_ok=True)
    texts, corpus = load_corpus(args)
    corpus.update(
        {
            "sample_size": len(texts),
            "seed": args.seed,
            "sample_text_sha256": _json_fingerprint(texts),
        }
    )
    run_metadata = {
        "format": "crispedit-cache-run/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_identity": identity,
        "layers": layers,
        "module_template": "model.layers.{}.mlp.down_proj.weight",
        "corpus": corpus,
        "sample_size": args.sample_size,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "energy_threshold": args.energy_threshold,
        "accumulation_dtype": "float32",
        "projection_storage_dtype": args.projection_storage_dtype,
        "preset_label": args.preset_label,
        "qwen3_layer_selection_validated": False,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_qwen25_preset": {
            "layers": [15, 16, 17, 18, 19],
            "mom2_dataset": "wikipedia",
            "mom2_n_samples": 1000,
            "mom2_dtype": "float32",
            "energy_threshold": 0.99,
        },
    }
    atomic_json(args.stats_dir / "cache_run_config.json", run_metadata)

    missing_layers: list[int] = []
    for layer in layers:
        path = args.stats_dir / stats_filename(layer)
        if args.force or not path.exists():
            missing_layers.append(layer)
            continue
        payload = torch.load(path, map_location="cpu")
        validate_stats_payload(payload, identity=identity, layer=layer)
        cached_corpus = payload.get("corpus", {})
        if cached_corpus.get("sample_text_sha256") != corpus["sample_text_sha256"]:
            raise ValueError(
                f"existing layer {layer} cache uses a different corpus sample; pass --force intentionally"
            )

    tokenizer = None
    model = None
    device = torch.device(f"cuda:{args.gpu}")
    started = time.time()
    if missing_layers:
        torch.cuda.set_device(args.gpu)
        tokenizer = load_tokenizer(args.model_dir)
        model = load_model(args.model_dir, device)
        modules = resolve_modules(model, missing_layers)
        dataset = TextDataset(texts, tokenizer, args.max_length)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda examples: collate_texts(examples, int(pad_id)),
            num_workers=args.num_workers,
            pin_memory=True,
        )
        torch.cuda.reset_peak_memory_stats(device)
        stats = collect_kfac_stats(
            model,
            modules,
            loader,
            device,
            accumulation_dtype=torch.float32,
        )
        for layer, result in stats.items():
            payload = {
                "format": STATS_FORMAT,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model_identity": identity,
                "layer": layer,
                "module_name": module_name_for_layer(layer),
                "A": result.A,
                "B": result.B,
                "num_documents": result.num_documents,
                "num_tokens": result.num_tokens,
                "corpus": corpus,
                "max_length": args.max_length,
                "accumulation_dtype": "float32",
                "upstream_repository": UPSTREAM_REPOSITORY,
                "upstream_commit": UPSTREAM_COMMIT,
            }
            atomic_torch_save(args.stats_dir / stats_filename(layer), payload)
        del stats, modules, model
        torch.cuda.empty_cache()

    projection_records: list[dict[str, Any]] = []
    if not args.stats_only:
        torch.cuda.set_device(args.gpu)
        storage_dtype = getattr(torch, args.projection_storage_dtype)
        for layer in layers:
            output_path = args.stats_dir / projection_filename(layer, args.energy_threshold)
            if output_path.exists() and not args.force:
                existing = torch.load(output_path, map_location="cpu")
                validate_projection_payload(
                    existing,
                    identity=identity,
                    layer=layer,
                    energy_threshold=args.energy_threshold,
                )
                existing_corpus = existing.get("source_corpus", {})
                if existing_corpus.get("sample_text_sha256") != corpus["sample_text_sha256"]:
                    raise ValueError(
                        f"existing layer {layer} projection uses a different corpus sample; "
                        "pass --force intentionally"
                    )
                projection_records.append(
                    {
                        "layer": layer,
                        "path": str(output_path.resolve()),
                        "rank": existing.get("rank"),
                        "allowed_low_curvature_directions": existing.get(
                            "allowed_low_curvature_directions"
                        ),
                        "reused": True,
                    }
                )
                continue
            stats_payload = torch.load(
                args.stats_dir / stats_filename(layer),
                map_location="cpu",
            )
            validate_stats_payload(stats_payload, identity=identity, layer=layer)
            projection = low_curvature_projection(
                stats_payload["A"].to(device),
                stats_payload["B"].to(device),
                args.energy_threshold,
            )
            payload = {
                "format": PROJECTION_FORMAT,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model_identity": identity,
                "layer": layer,
                "module_name": module_name_for_layer(layer),
                "energy_threshold": args.energy_threshold,
                "source_stats_file": stats_filename(layer),
                "source_corpus": stats_payload["corpus"],
                "Ua": projection.pop("Ua").to(storage_dtype).cpu(),
                "Ub": projection.pop("Ub").to(storage_dtype).cpu(),
                "mask": projection.pop("mask").cpu(),
                "storage_dtype": args.projection_storage_dtype,
                **projection,
                "upstream_repository": UPSTREAM_REPOSITORY,
                "upstream_commit": UPSTREAM_COMMIT,
            }
            atomic_torch_save(output_path, payload)
            projection_records.append(
                {
                    "layer": layer,
                    "path": str(output_path.resolve()),
                    "rank": payload["rank"],
                    "allowed_low_curvature_directions": payload[
                        "allowed_low_curvature_directions"
                    ],
                    "reused": False,
                }
            )
            del stats_payload, projection, payload
            torch.cuda.empty_cache()

    atomic_json(
        args.stats_dir / "completed.json",
        {
            "status": "complete",
            "elapsed_seconds": time.time() - started,
            "layers": layers,
            "stats_files": [str((args.stats_dir / stats_filename(layer)).resolve()) for layer in layers],
            "projections": projection_records,
            "publication_ready_corpus": corpus.get("publication_ready"),
            "qwen3_layer_selection_validated": False,
        },
    )


if __name__ == "__main__":
    main()
