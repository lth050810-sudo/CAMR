#!/usr/bin/env python3
"""Shared training and evaluation utilities for WikiBigEdit baselines.

The module intentionally depends only on packages already present in the
project's ``.venv-qwen3`` environment.  External editing repositories are
kept isolated and can call this evaluator through JSON manifests.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import string
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


AXES = ("update", "rephrase", "persona", "locality", "multihop")


@dataclass
class TrainStats:
    loss: float
    optimizer_steps: int
    sequence_tokens: int
    answer_tokens: int
    wall_seconds: float
    peak_vram_gb: float


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qa_prompt(question: str) -> str:
    """The single fixed completion template used by every baseline."""

    return f"Question: {question.strip()}\nAnswer:"


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    lines = text.splitlines()
    text = lines[0] if lines else text
    text = "".join(" " if char in string.punctuation else char for char in text)
    return " ".join(text.split())


def answer_scores(prediction: str, gold: str) -> dict[str, bool]:
    pred = normalize_answer(prediction)
    target = normalize_answer(gold)
    if not target:
        return {"exact": False, "match": False}
    pred_tokens = pred.split()
    target_tokens = target.split()
    contains = any(
        pred_tokens[index : index + len(target_tokens)] == target_tokens
        for index in range(max(len(pred_tokens) - len(target_tokens) + 1, 0))
    )
    return {"exact": pred == target, "match": contains}


def _manifest_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise TypeError("manifest must be a JSON list or object")
    if payload.get("schema_version") == "wikibigedit-protocol/v1":
        records = payload.get("records")
        selected_ids = payload.get("selection", {}).get("sample_ids")
        if not isinstance(records, list) or not isinstance(selected_ids, list):
            raise ValueError("invalid wikibigedit-protocol/v1 manifest")
        record_map = {
            str(record.get("sample_id")): record
            for record in records
            if isinstance(record, dict) and record.get("sample_id") is not None
        }
        missing = [sample_id for sample_id in selected_ids if str(sample_id) not in record_map]
        if missing:
            raise ValueError(f"selection references missing records: {missing[:3]}")
        return [record_map[str(sample_id)] for sample_id in selected_ids]
    for key in ("samples", "selected_samples", "rows", "facts", "records"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return rows
    stages = payload.get("stages")
    if isinstance(stages, dict):
        rows = []
        for stage_key in sorted(stages, key=lambda value: int(str(value).replace("tau", ""))):
            stage_rows = stages[stage_key]
            if not isinstance(stage_rows, list):
                raise TypeError(f"manifest stages[{stage_key!r}] is not a list")
            rows.extend(stage_rows)
        return rows
    raise KeyError("manifest has no samples/selected_samples/rows/facts/stages field")


def load_manifest(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = _manifest_rows(payload)
    normalized: list[dict[str, Any]] = []
    required = ("sample_id", "stage", "update", "rephrase", "ans")
    for raw in rows:
        row = dict(raw)
        if "personas" not in row and "persona" in row:
            row["personas"] = row.get("persona")
        missing = [key for key in required if row.get(key) in (None, "")]
        if missing:
            raise ValueError(f"manifest row missing {missing}: {row.get('sample_id', '<unknown>')}")
        row["stage"] = int(row["stage"])
        normalized.append(row)
    normalized.sort(key=lambda row: (row["stage"], str(row["sample_id"])))
    identifiers = [str(row["sample_id"]) for row in normalized]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("manifest sample_id values are not unique")
    metadata = payload if isinstance(payload, dict) else {}
    return normalized, metadata


def group_by_stage(rows: Sequence[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    stages: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        stages.setdefault(int(row["stage"]), []).append(row)
    return dict(sorted(stages.items()))


def fixed_panel(
    rows: Sequence[dict[str, Any]],
    question_key: str,
    answer_key: str,
    size: int,
    seed: int,
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if isinstance(row.get(question_key), str)
        and row[question_key].strip()
        and isinstance(row.get(answer_key), str)
        and row[answer_key].strip()
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    if size > 0:
        candidates = candidates[:size]
    return candidates


def manifest_panel(
    manifest: dict[str, Any],
    panel: str,
    *,
    limit: int = 0,
) -> list[dict[str, Any]] | None:
    """Resolve a panel frozen by ``wikibigedit_protocol.py``.

    Returns ``None`` for older manifests so callers can use a deterministic
    fallback.  A positive limit takes the already-frozen prefix and therefore
    never resamples differently between methods.
    """

    if manifest.get("schema_version") != "wikibigedit-protocol/v1":
        return None
    panel_info = manifest.get("panels", {}).get(panel)
    records = manifest.get("records")
    if not isinstance(panel_info, dict) or not isinstance(records, list):
        raise ValueError(f"protocol manifest is missing panel {panel!r}")
    identifiers = panel_info.get("sample_ids")
    if not isinstance(identifiers, list):
        raise ValueError(f"protocol panel {panel!r} has no sample_ids")
    if limit > 0:
        identifiers = identifiers[:limit]
    record_map = {str(record["sample_id"]): record for record in records}
    missing = [sample_id for sample_id in identifiers if str(sample_id) not in record_map]
    if missing:
        raise ValueError(f"panel {panel!r} references missing records: {missing[:3]}")
    return [record_map[str(sample_id)] for sample_id in identifiers]


class AnswerOnlyDataset(Dataset):
    """Causal-LM examples whose prompt tokens are masked from the loss."""

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.examples: list[dict[str, torch.Tensor]] = []
        eos = tokenizer.eos_token or ""
        for row in rows:
            prompt_ids = tokenizer(
                qa_prompt(str(row["update"])),
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
            ).input_ids
            answer_ids = tokenizer(
                " " + str(row["ans"]).strip() + eos,
                add_special_tokens=False,
                truncation=True,
                max_length=max(1, max_length - len(prompt_ids)),
            ).input_ids
            input_ids = (prompt_ids + answer_ids)[:max_length]
            labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
            if not any(label != -100 for label in labels):
                continue
            self.examples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )
        if not self.examples:
            raise ValueError("no trainable examples after tokenization")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


def make_collator(tokenizer: Any):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def collate(examples: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        width = max(example["input_ids"].numel() for example in examples)
        input_ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
        labels = torch.full((len(examples), width), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), width), dtype=torch.long)
        for index, example in enumerate(examples):
            length = example["input_ids"].numel()
            input_ids[index, :length] = example["input_ids"]
            labels[index, :length] = example["labels"]
            attention_mask[index, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


def load_tokenizer(model_dir: Path) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_dir: Path,
    device: torch.device,
    *,
    use_flash_attention: bool = True,
) -> torch.nn.Module:
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "local_files_only": True,
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
    }
    if use_flash_attention:
        kwargs["attn_implementation"] = "flash_attention_2"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    except Exception:
        if not use_flash_attention:
            raise
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    model.to(device)
    return model


@torch.inference_mode()
def generate_answers(
    model: torch.nn.Module,
    tokenizer: Any,
    questions: Sequence[str],
    device: torch.device,
    *,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
) -> list[str]:
    answers: list[str] = []
    old_padding_side = tokenizer.padding_side
    old_cache = model.config.use_cache
    tokenizer.padding_side = "left"
    model.config.use_cache = True
    model.eval()
    try:
        for start in range(0, len(questions), batch_size):
            prompts = [qa_prompt(question) for question in questions[start : start + batch_size]]
            encoded = tokenizer(
                prompts,
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
            new_tokens = generated[:, encoded.input_ids.shape[1] :]
            answers.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = old_padding_side
        model.config.use_cache = old_cache
    return answers


def axis_fields(axis: str) -> tuple[str, str]:
    mapping = {
        "update": ("update", "ans"),
        "rephrase": ("rephrase", "ans"),
        "persona": ("personas", "ans"),
        "locality": ("loc", "loc_ans"),
        "multihop": ("mhop", "mhop_ans"),
    }
    if axis not in mapping:
        raise KeyError(f"unknown evaluation axis: {axis}")
    return mapping[axis]


def evaluate_axis(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    axis: str,
    device: torch.device,
    *,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    checkpoint: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    question_key, answer_key = axis_fields(axis)
    valid = [
        row
        for row in rows
        if isinstance(row.get(question_key), str)
        and row[question_key].strip()
        and isinstance(row.get(answer_key), str)
        and row[answer_key].strip()
    ]
    questions = [str(row[question_key]) for row in valid]
    predictions = generate_answers(
        model,
        tokenizer,
        questions,
        device,
        batch_size=batch_size,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )
    records: list[dict[str, Any]] = []
    exact_count = 0
    match_count = 0
    for row, prediction in zip(valid, predictions):
        scores = answer_scores(prediction, str(row[answer_key]))
        exact_count += int(scores["exact"])
        match_count += int(scores["match"])
        records.append(
            {
                "checkpoint": checkpoint,
                "sample_id": row["sample_id"],
                "fact_stage": int(row["stage"]),
                "axis": axis,
                "question": row[question_key],
                "gold": row[answer_key],
                "prediction": prediction,
                **scores,
            }
        )
    count = len(valid)
    summary = {
        "count": count,
        "exact_count": exact_count,
        "match_count": match_count,
        "exact_accuracy": exact_count / count if count else None,
        "match_accuracy": match_count / count if count else None,
    }
    return summary, records


def evaluate_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    device: torch.device,
    *,
    checkpoint: str,
    axes: Sequence[str],
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    locality_panel: Sequence[dict[str, Any]] | None = None,
    multihop_panel: Sequence[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics: dict[str, Any] = {"checkpoint": checkpoint, "num_facts": len(rows)}
    records: list[dict[str, Any]] = []
    for axis in axes:
        axis_rows = rows
        if axis == "locality" and locality_panel is not None:
            axis_rows = locality_panel
        if axis == "multihop" and multihop_panel is not None:
            axis_rows = multihop_panel
        summary, axis_records = evaluate_axis(
            model,
            tokenizer,
            axis_rows,
            axis,
            device,
            batch_size=batch_size,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            checkpoint=checkpoint,
        )
        metrics[axis] = summary
        records.extend(axis_records)
    return metrics, records


def train_answer_only(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    device: torch.device,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    max_length: int,
    optimizer: torch.optim.Optimizer | None = None,
    stage_lr_schedule: bool = False,
) -> tuple[TrainStats, torch.optim.Optimizer]:
    random.seed(seed)
    torch.manual_seed(seed)
    dataset = AnswerOnlyDataset(rows, tokenizer, max_length)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=make_collator(tokenizer),
        num_workers=0,
        pin_memory=True,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    if optimizer is None:
        try:
            optimizer = torch.optim.AdamW(
                parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
                fused=True,
            )
        except (TypeError, RuntimeError):
            optimizer = torch.optim.AdamW(
                parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
    model.train()
    model.config.use_cache = False
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    loss_count = 0
    optimizer_steps = 0
    sequence_tokens = 0
    answer_tokens = 0
    started = time.perf_counter()
    for _epoch in range(epochs):
        for batch_index, batch in enumerate(loader):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            (loss / gradient_accumulation_steps).backward()
            loss_total += float(loss.detach())
            loss_count += 1
            sequence_tokens += int(batch["attention_mask"].sum())
            answer_tokens += int((batch["labels"] != -100).sum())
            boundary = (batch_index + 1) % gradient_accumulation_steps == 0
            final_batch = batch_index + 1 == len(loader)
            if boundary or final_batch:
                if stage_lr_schedule:
                    total_steps = epochs * math.ceil(len(loader) / gradient_accumulation_steps)
                    factor = stage_lr_factor(optimizer_steps, total_steps)
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate * factor
                torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    return (
        TrainStats(
            loss=loss_total / max(loss_count, 1),
            optimizer_steps=optimizer_steps,
            sequence_tokens=sequence_tokens,
            answer_tokens=answer_tokens,
            wall_seconds=elapsed,
            peak_vram_gb=peak,
        ),
        optimizer,
    )


def stage_lr_factor(step: int, total: int) -> float:
    """TiC-style 10% stage rewarm, then cosine decay to 10% of peak LR."""
    warmup = max(1, math.ceil(total * 0.1))
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup - 1))
    return 0.1 + 0.9 * (1.0 + math.cos(math.pi * progress)) / 2.0


@torch.inference_mode()
def dense_prompt_embeddings(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    device: torch.device,
    *,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    """Compute frozen Qwen3 mean-pooled prompt states for semantic replay."""

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = getattr(base, "model", None)
    if backbone is None:
        raise AttributeError("causal LM has no .model backbone for dense embeddings")
    previous_cache = base.config.use_cache
    base.config.use_cache = False
    base.eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, len(rows), batch_size):
        texts = [qa_prompt(str(row["update"])) for row in rows[start : start + batch_size]]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        outputs = backbone(**encoded, use_cache=False, return_dict=True)
        hidden = outputs.last_hidden_state.float()
        mask = encoded.attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        chunks.append(F.normalize(pooled, dim=-1).cpu())
    base.config.use_cache = previous_cache
    return torch.cat(chunks, dim=0)


def select_semantic_replay(
    current_indices: Sequence[int],
    memory_indices: Sequence[int],
    embeddings: torch.Tensor,
    budget: int,
) -> list[int]:
    if budget <= 0 or not memory_indices:
        return []
    current = embeddings[torch.tensor(list(current_indices), dtype=torch.long)]
    memory = embeddings[torch.tensor(list(memory_indices), dtype=torch.long)]
    similarities = current @ memory.T
    best = similarities.max(dim=0).values
    count = min(budget, len(memory_indices))
    positions = torch.topk(best, k=count, largest=True, sorted=True).indices.tolist()
    return [int(memory_indices[position]) for position in positions]


class Reservoir:
    def __init__(self, capacity: int, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("reservoir capacity must be positive")
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.items: list[int] = []
        self.seen = 0

    def add_many(self, indices: Iterable[int]) -> None:
        for index in indices:
            self.seen += 1
            if len(self.items) < self.capacity:
                self.items.append(int(index))
                continue
            replacement = self.rng.randrange(self.seen)
            if replacement < self.capacity:
                self.items[replacement] = int(index)

    def sample(self, count: int) -> list[int]:
        if count <= 0:
            return []
        return self.rng.sample(self.items, min(count, len(self.items)))


def environment_metadata() -> dict[str, Any]:
    import peft
    import transformers

    return {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "hostname": os.uname().nodename,
    }
