"""Frozen sentence encoding shared by the external-memory full writer.

The encoder and provenance helpers are extracted verbatim from run_rag.py.
Keeping them here prevents unrelated RAG runner edits or runner imports from
changing the writer's checkpoint dependency identity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from baselines.common import sha256_file


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fact_document(row: Mapping[str, Any]) -> str:
    """Canonical indexed/context document for one newly available fact."""

    return f"Question: {str(row['update']).strip()}\nAnswer: {str(row['ans']).strip()}"


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
