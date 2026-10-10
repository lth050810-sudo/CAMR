#!/usr/bin/env python3
"""Run a Qwen3 compatibility port of MEMIT on the shared WikiBigEdit protocol.

The algorithm follows the EasyEdit/MEMIT implementation: optimize a target
activation at the last rewrite layer, extract subject-token keys at the input
to each MLP ``down_proj``, and distribute the residual through a covariance-
regularized rank update over the selected layers.

There is no official Qwen3-8B preset.  The default is a development-only Qwen3
candidate (rewrite layers 5--8, a small solver ridge, and final-block loss)
chosen to avoid the known layer-4 singular/unstable transfer.  Explicit
``--layers`` still selects exactly the requested layers.  A real comparison additionally needs
Qwen3-specific Wikipedia second moments for every rewrite layer.  The
``manifest_diag`` statistics source exists only for a tiny compatibility
smoke test and is deliberately rejected unless ``--allow-proxy-stats`` is
given; it must never be reported as a MEMIT benchmark result.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import math
import random
import re
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from baselines.common import (
    append_jsonl,
    atomic_json,
    environment_metadata,
    fixed_panel,
    group_by_stage,
    load_manifest,
    load_model,
    load_tokenizer,
    manifest_panel,
    qa_prompt,
    sha256_file,
)
from baselines.run_lora_family import evaluate_and_write
from baselines.run_ultraedit import parse_int_range, restrict_stages


UPSTREAM_REPOSITORY = "zjunlp/EasyEdit (MEMIT port)"
UPSTREAM_COMMIT = "14cea8245f06715684592ab55184939b99d70784"
MEMIT_STATS_SCHEMA = "engram-memit-mom2/v1"
UPSTREAM_QWEN25_PRESET = {
    "layers": [4, 5, 6, 7, 8],
    "clamp_norm_factor": 4.0,
    "fact_token": "subject_last",
    "v_num_grad_steps": 25,
    "v_lr": 5e-1,
    "v_loss_layer": 27,
    "v_weight_decay": 1e-3,
    "kl_factor": 0.0625,
    "mom2_update_weight": 15000.0,
    "mom2_dataset": "wikimedia/wikipedia:20231101.en",
    "mom2_n_samples": 100000,
    "mom2_dtype": "float32",
    "rewrite_module_tmp": "model.layers.{}.mlp.down_proj",
}
DEFAULT_QWEN3_LAYERS = [5, 6, 7, 8]
DEFAULT_SOLVER_RIDGE = 1e-5
DEFAULT_MAX_UPDATE_WEIGHT_RATIO = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("Qwen3-8B-Base"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--layers",
        type=parse_int_range,
        default=None,
        help=(
            "Qwen block indices; default 5-8 is an unvalidated development candidate. "
            "Explicit values are preserved exactly"
        ),
    )
    parser.add_argument("--v-loss-layer", type=int, default=35)
    parser.add_argument("--v-num-grad-steps", type=int, default=25)
    parser.add_argument("--v-lr", type=float, default=5e-1)
    parser.add_argument("--v-weight-decay", type=float, default=1e-3)
    parser.add_argument("--clamp-norm-factor", type=float, default=4.0)
    parser.add_argument("--kl-factor", type=float, default=0.0625)
    parser.add_argument("--mom2-update-weight", type=float, default=15000.0)
    parser.add_argument("--solver-ridge", type=float, default=DEFAULT_SOLVER_RIDGE)
    parser.add_argument(
        "--max-update-weight-ratio",
        type=float,
        default=DEFAULT_MAX_UPDATE_WEIGHT_RATIO,
        help=(
            "fail before committing a layer update when ||delta W||/||W|| exceeds "
            "this positive bound"
        ),
    )
    parser.add_argument(
        "--preset-label",
        type=str,
        default=None,
        help="optional audit label for a non-upstream sensitivity variant",
    )
    parser.add_argument(
        "--selection-provenance",
        type=str,
        default=None,
        help="optional note describing how a Qwen3-specific layer/config variant was selected",
    )
    parser.add_argument(
        "--selection-certificate",
        type=Path,
        default=None,
        help="hash-bound independent-development selection required for a formal full run",
    )
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="mark this run as a development candidate, never a benchmark result",
    )
    parser.add_argument("--edit-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--representation-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--locality-panel-size", type=int, default=0)
    parser.add_argument("--multihop-panel-size", type=int, default=0)
    parser.add_argument("--max-stages", type=int, default=0)
    parser.add_argument("--examples-per-stage", type=int, default=0)
    parser.add_argument("--no-stage-eval", action="store_true")
    parser.add_argument("--skip-base-eval", action="store_true")
    parser.add_argument(
        "--context-mode",
        choices=("upstream_generated", "static"),
        default="upstream_generated",
        help="static is intended only for a fast compatibility smoke",
    )
    parser.add_argument(
        "--stats-source",
        choices=("precomputed", "manifest_diag"),
        default="precomputed",
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=Path("/mnt/yy2607004_hs/engram_memit_stats"),
    )
    parser.add_argument(
        "--stats-file-template",
        type=str,
        default=None,
        help="optional Python format string with {layer} and {module}",
    )
    parser.add_argument("--stats-sample-size", type=int, default=100000)
    parser.add_argument("--allow-unverified-stats", action="store_true")
    parser.add_argument("--allow-proxy-stats", action="store_true")
    parser.add_argument("--proxy-stat-samples", type=int, default=64)
    parser.add_argument("--proxy-variance-floor", type=float, default=1e-5)
    parser.add_argument("--save-delta", type=Path, default=None)
    parser.add_argument("--allow-existing-output", action="store_true")
    args = parser.parse_args()
    args.default_qwen3_candidate = args.layers is None
    if args.layers is None:
        args.layers = list(DEFAULT_QWEN3_LAYERS)
        # The default is deliberately not benchmark-eligible until held-out
        # Qwen3 selection has completed.  An explicit --layers remains under
        # the caller's control and is not silently rewritten.
        args.development_only = True
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_module(root: torch.nn.Module, name: str) -> torch.nn.Module:
    module = root
    for part in name.split("."):
        module = getattr(module, part)
    return module


def get_primary_hidden(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        if not isinstance(output[0], torch.Tensor):
            raise TypeError(f"first module output is not a tensor: {type(output[0])}")
        return output[0]
    raise TypeError(f"unsupported module output type: {type(output)}")


def replace_primary_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise TypeError(f"unsupported module output type: {type(output)}")


def find_subject_span(prompt: str, subject: str) -> tuple[int, int, str]:
    """Find an entity mention, allowing WikiBigEdit punctuation mismatches."""

    exact = prompt.find(subject)
    if exact >= 0:
        return exact, exact + len(subject), "exact"
    subject_tokens = [
        match.group(0).casefold()
        for match in re.finditer(r"\w+", unicodedata.normalize("NFKC", subject))
    ]
    prompt_matches = list(re.finditer(r"\w+", prompt))
    prompt_tokens = [match.group(0).casefold() for match in prompt_matches]
    if subject_tokens:
        width = len(subject_tokens)
        for start in range(len(prompt_tokens) - width + 1):
            if prompt_tokens[start : start + width] == subject_tokens:
                return (
                    prompt_matches[start].start(),
                    prompt_matches[start + width - 1].end(),
                    "punctuation_normalized",
                )
    raise ValueError(f"subject {subject!r} is absent from prompt {prompt!r}")


def make_subject_template_details(
    question: str,
    subject: str,
    *,
    unified_qa: bool = True,
) -> tuple[str, str, str]:
    prompt = qa_prompt(question) if unified_qa else question.strip()
    if not subject:
        raise ValueError("subject is empty")
    start, end, match_mode = find_subject_span(prompt, subject)
    surface_subject = prompt[start:end]
    template = prompt[:start] + "{}" + prompt[end:]
    if template.count("{}") != 1:
        raise ValueError(f"MEMIT requires exactly one placeholder: {template!r}")
    return template, surface_subject, match_mode


def make_subject_template(question: str, subject: str, *, unified_qa: bool = True) -> str:
    """Create the single-placeholder template required by MEMIT."""

    return make_subject_template_details(
        question, subject, unified_qa=unified_qa
    )[0]


def make_request(row: dict[str, Any], case_id: int) -> dict[str, Any]:
    canonical_subject = str(row.get("subject") or "").strip()
    prompt, surface_subject, match_mode = make_subject_template_details(
        str(row["update"]), canonical_subject
    )
    return {
        "case_id": case_id,
        "sample_id": str(row["sample_id"]),
        "subject": surface_subject,
        "canonical_subject": canonical_subject,
        "subject_match": match_mode,
        "prompt": prompt,
        "target_new": " " + str(row["ans"]).strip(),
    }


def subject_last_index(tokenizer: Any, template: str, subject: str) -> int:
    """Resolve the last token overlapping the subject mention.

    EasyEdit estimates this from separately tokenized prefixes.  Modern Qwen
    BPE merges can make that length difference zero for ordinary one-token
    subjects, so fast-tokenizer offsets are used when available and the
    upstream length formula remains as a fallback.
    """

    if template.count("{}") != 1:
        raise ValueError("subject template must contain exactly one placeholder")
    prefix, _suffix = template.split("{}")
    rendered = template.format(subject)
    subject_start = len(prefix)
    subject_end = subject_start + len(subject)
    try:
        encoded = tokenizer(
            rendered,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping") if hasattr(encoded, "get") else None
        if offsets is not None:
            if isinstance(offsets, torch.Tensor):
                offsets = offsets.tolist()
            if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], list):
                offsets = offsets[0]
            overlapping = [
                index
                for index, (start, end) in enumerate(offsets)
                if end > subject_start and start < subject_end
            ]
            if overlapping:
                return overlapping[-1]
    except (TypeError, ValueError, NotImplementedError):
        pass
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
    through_subject = tokenizer.encode(prefix + subject, add_special_tokens=True)
    subject_token_count = len(through_subject) - len(prefix_ids)
    if subject_token_count <= 0:
        raise ValueError(f"subject produced no tokens in template: {subject!r}")
    return len(through_subject) - 1


@contextlib.contextmanager
def tokenizer_padding(tokenizer: Any, side: str):
    previous = tokenizer.padding_side
    tokenizer.padding_side = side
    try:
        yield
    finally:
        tokenizer.padding_side = previous


def _encoded_batch(
    tokenizer: Any,
    texts: Sequence[str],
    device: torch.device,
    max_length: int,
) -> dict[str, torch.Tensor]:
    with tokenizer_padding(tokenizer, "right"):
        encoded = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
    return {key: value.to(device) for key, value in encoded.items()}


def capture_representations(
    model: torch.nn.Module,
    tokenizer: Any,
    templates: Sequence[str],
    subjects: Sequence[str],
    module_name: str,
    track: str,
    device: torch.device,
    *,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    """Capture subject-token inputs or outputs of a named module."""

    if len(templates) != len(subjects):
        raise ValueError("templates and subjects must have equal length")
    if track not in {"in", "out"}:
        raise ValueError(f"invalid track value: {track}")
    module = get_module(model, module_name)
    captured: list[torch.Tensor] = []
    for start in range(0, len(templates), batch_size):
        batch_templates = templates[start : start + batch_size]
        batch_subjects = subjects[start : start + batch_size]
        indices = [
            subject_last_index(tokenizer, template, subject)
            for template, subject in zip(batch_templates, batch_subjects)
        ]
        texts = [
            template.format(subject)
            for template, subject in zip(batch_templates, batch_subjects)
        ]
        encoded = _encoded_batch(tokenizer, texts, device, max_length)
        lengths = encoded["attention_mask"].sum(dim=1).tolist()
        for index, length in zip(indices, lengths):
            if index < 0 or index >= int(length):
                raise ValueError("subject token was truncated from a MEMIT context")
        holder: dict[str, torch.Tensor] = {}

        def hook(_module, inputs, output):
            value = inputs[0] if track == "in" else get_primary_hidden(output)
            holder["value"] = value

        handle = module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                model(**encoded, use_cache=False)
        finally:
            handle.remove()
        value = holder.get("value")
        if value is None:
            raise RuntimeError(f"hook did not capture {module_name}")
        for row_index, token_index in enumerate(indices):
            captured.append(value[row_index, token_index].detach())
    return torch.stack(captured, dim=0)


def flatten_context_templates(context_groups: Sequence[Sequence[str]]) -> list[str]:
    return [template for group in context_groups for template in group]


def average_context_groups(
    representations: torch.Tensor,
    *,
    request_count: int,
    group_sizes: Sequence[int],
) -> torch.Tensor:
    """Average like upstream MEMIT: equal weight per context group."""

    contexts_per_request = sum(group_sizes)
    expected = request_count * contexts_per_request
    if representations.shape[0] != expected:
        raise ValueError(f"expected {expected} contexts, got {representations.shape[0]}")
    results: list[torch.Tensor] = []
    for request_index in range(request_count):
        offset = request_index * contexts_per_request
        group_means: list[torch.Tensor] = []
        for size in group_sizes:
            group_means.append(representations[offset : offset + size].mean(dim=0))
            offset += size
        results.append(torch.stack(group_means, dim=0).mean(dim=0))
    return torch.stack(results, dim=0)


def compute_keys(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[dict[str, Any]],
    layer: int,
    context_groups: Sequence[Sequence[str]],
    device: torch.device,
    *,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    flat_contexts = flatten_context_templates(context_groups)
    templates = [
        context.format(request["prompt"])
        for request in requests
        for context in flat_contexts
    ]
    subjects = [
        request["subject"]
        for request in requests
        for _context in flat_contexts
    ]
    raw = capture_representations(
        model,
        tokenizer,
        templates,
        subjects,
        f"model.layers.{layer}.mlp.down_proj",
        "in",
        device,
        batch_size=batch_size,
        max_length=max_length,
    )
    return average_context_groups(
        raw,
        request_count=len(requests),
        group_sizes=[len(group) for group in context_groups],
    )


def compute_current_zs(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[dict[str, Any]],
    z_layer: int,
    device: torch.device,
    *,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    return capture_representations(
        model,
        tokenizer,
        [request["prompt"] for request in requests],
        [request["subject"] for request in requests],
        f"model.layers.{z_layer}",
        "out",
        device,
        batch_size=batch_size,
        max_length=max_length,
    )


@dataclass
class ZOptimizationStats:
    sample_id: str
    iterations: int
    final_loss: float
    final_nll: float
    final_kl: float
    final_weight_decay: float
    target_init_norm: float
    delta_norm: float


def compute_z_star(
    model: torch.nn.Module,
    tokenizer: Any,
    request: dict[str, Any],
    context_groups: Sequence[Sequence[str]],
    z_layer: int,
    loss_layer: int,
    device: torch.device,
    args: argparse.Namespace,
    *,
    target_transform=None,
) -> tuple[torch.Tensor, ZOptimizationStats]:
    """Optimize one MEMIT target activation, following EasyEdit's objective."""

    target_ids = tokenizer.encode(
        request["target_new"], return_tensors="pt", add_special_tokens=False
    )[0].to(device)
    if target_ids.numel() == 0:
        raise ValueError(f"empty target tokenization for {request['sample_id']}")
    special_ids = {tokenizer.bos_token_id, tokenizer.unk_token_id}
    if int(target_ids[0]) in {value for value in special_ids if value is not None}:
        target_ids = target_ids[1:]
    if target_ids.numel() == 0:
        raise ValueError(f"target only contained a special token: {request['sample_id']}")

    target_prefix = tokenizer.decode(target_ids[:-1], skip_special_tokens=False)
    rewriting_templates = [
        context.format(request["prompt"]) + target_prefix
        for context in flatten_context_templates(context_groups)
    ]
    kl_templates = ["{} is a"]
    all_templates = rewriting_templates + kl_templates
    subjects = [request["subject"]] * len(all_templates)
    lookup_indices = [
        subject_last_index(tokenizer, template, request["subject"])
        for template in all_templates
    ]
    texts = [template.format(request["subject"]) for template in all_templates]
    encoded = _encoded_batch(tokenizer, texts, device, args.max_length)
    lengths = encoded["attention_mask"].sum(dim=1)
    if any(index >= int(length) for index, length in zip(lookup_indices, lengths.tolist())):
        raise ValueError("subject token was truncated during z optimization")

    rewriting_targets = torch.full(
        (len(rewriting_templates), encoded["input_ids"].shape[1]),
        -100,
        dtype=torch.long,
        device=device,
    )
    for row_index in range(len(rewriting_templates)):
        end = int(lengths[row_index])
        start = end - int(target_ids.numel())
        if start < 0:
            raise ValueError("target tokens do not fit in the MEMIT input")
        rewriting_targets[row_index, start:end] = target_ids

    rewrite_name = f"model.layers.{z_layer}"
    loss_name = f"model.layers.{loss_layer}"
    rewrite_module = get_module(model, rewrite_name)
    loss_module = get_module(model, loss_name)
    hidden_size = int(model.config.hidden_size)
    delta = torch.zeros(hidden_size, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=args.v_lr)
    target_init: torch.Tensor | None = None
    kl_initial: torch.Tensor | None = None
    final_values = (math.inf, math.inf, math.inf, math.inf)
    completed_iterations = 0

    for iteration in range(args.v_num_grad_steps):
        optimizer.zero_grad(set_to_none=True)
        holder: dict[str, torch.Tensor] = {}

        def rewrite_hook(_module, _inputs, output):
            nonlocal target_init
            hidden = get_primary_hidden(output)
            if target_init is None:
                target_init = hidden[0, lookup_indices[0]].detach().clone()
            edited = hidden.clone()
            update = delta.to(dtype=edited.dtype)
            for batch_index, token_index in enumerate(lookup_indices):
                edited[batch_index, token_index] = edited[batch_index, token_index] + update
            if rewrite_name == loss_name:
                holder["loss_hidden"] = edited
            return replace_primary_hidden(output, edited)

        def loss_hook(_module, _inputs, output):
            holder["loss_hidden"] = get_primary_hidden(output)

        handles = [rewrite_module.register_forward_hook(rewrite_hook)]
        if rewrite_name != loss_name:
            handles.append(loss_module.register_forward_hook(loss_hook))
        try:
            outputs = model(**encoded, use_cache=False)
        finally:
            for handle in reversed(handles):
                handle.remove()

        loss_hidden = holder.get("loss_hidden")
        if loss_hidden is None or target_init is None:
            raise RuntimeError("MEMIT z-optimization hooks did not fire")
        logits = outputs.logits
        kl_logits = logits[len(rewriting_templates), lookup_indices[-1], :]
        kl_log_probs = F.log_softmax(kl_logits.float(), dim=-1).unsqueeze(0)
        if kl_initial is None:
            kl_initial = kl_log_probs.detach().clone()

        projected = model.lm_head(model.model.norm(loss_hidden[: len(rewriting_templates)]))
        log_probs = F.log_softmax(projected.float(), dim=-1)
        gather_ids = torch.where(rewriting_targets != -100, rewriting_targets, 0)
        selected = torch.gather(log_probs, 2, gather_ids.unsqueeze(2)).squeeze(2)
        mask = rewriting_targets.ne(-100).to(selected.dtype)
        nll_each = -(selected * mask).sum(dim=1) / target_ids.numel()
        nll = nll_each.mean()
        kl = args.kl_factor * F.kl_div(
            kl_initial,
            kl_log_probs,
            log_target=True,
            reduction="batchmean",
        )
        weight_decay = args.v_weight_decay * delta.norm() / target_init.float().norm().square()
        loss = nll + kl + weight_decay
        final_values = (
            float(loss.detach()),
            float(nll.detach()),
            float(kl.detach()),
            float(weight_decay.detach()),
        )
        completed_iterations = iteration + 1
        if float(loss.detach()) < 5e-2 or iteration + 1 == args.v_num_grad_steps:
            break
        loss.backward()
        optimizer.step()
        max_norm = args.clamp_norm_factor * target_init.float().norm()
        with torch.no_grad():
            if delta.norm() > max_norm:
                delta.mul_(max_norm / delta.norm())

    target = target_init + delta.to(dtype=target_init.dtype)
    if target_transform is not None:
        target = target_transform(target_init.detach(), delta.detach(), rewriting_templates[0])
    return target.detach(), ZOptimizationStats(
        sample_id=request["sample_id"],
        iterations=completed_iterations,
        final_loss=final_values[0],
        final_nll=final_values[1],
        final_kl=final_values[2],
        final_weight_decay=final_values[3],
        target_init_norm=float(target_init.float().norm()),
        delta_norm=float(delta.detach().norm()),
    )


def model_stats_key(model_dir: Path) -> str:
    return model_dir.resolve().name.replace("/", "_")


def model_artifact_files(model_dir: Path) -> list[Path]:
    """Return the exact model files covered by the formal content fingerprint."""

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


def model_content_fingerprint(model_dir: Path) -> str:
    """Match the collector's content hash for the current checkpoint."""

    model_dir = model_dir.resolve()
    digest = hashlib.sha256()
    for path in model_artifact_files(model_dir):
        stat = path.stat()
        digest.update(str(path.relative_to(model_dir)).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def formal_stats_provenance_checks(
    metadata: dict[str, Any],
    *,
    moment_ndim: int,
    expected_model_dir: Path,
    expected_model_fingerprint: str | None,
    expected_config_sha256: str,
    expected_module_name: str,
    expected_sample_size: int,
) -> dict[str, bool]:
    """Return explicit checks used to qualify a formal MEMIT moment artifact."""

    recorded_model = metadata.get("model_dir") or metadata.get("model_name")
    model_path_matches = False
    if recorded_model is not None:
        model_path_matches = str(Path(recorded_model).resolve()) == str(
            expected_model_dir.resolve()
        )
    recorded_sample_size = _safe_int(metadata.get("sample_size_requested"))
    observed_documents = _safe_int(metadata.get("documents_observed"))
    corpus = metadata.get("corpus") if isinstance(metadata.get("corpus"), dict) else {}
    corpus_source = (
        corpus.get("source") if isinstance(corpus.get("source"), dict) else {}
    )
    corpus_documents = _safe_int(corpus.get("documents"))
    source_fingerprint = corpus_source.get("source_fingerprint")
    source_kind = corpus_source.get("source")
    source_is_auditable = bool(
        isinstance(source_fingerprint, str)
        and source_fingerprint
        and (
            (
                source_kind == "local"
                and corpus_source.get("source_fingerprint_mode") == "content"
            )
            or (
                source_kind == "huggingface_streaming"
                and corpus_source.get("dataset") == "wikimedia/wikipedia"
                and corpus_source.get("config") == "20231101.en"
                and corpus_source.get("split") == "train"
            )
        )
    )
    cache_sha = corpus.get("cache_sha256")
    corpus_cache_matches = bool(
        isinstance(cache_sha, str)
        and cache_sha
        and metadata.get("corpus_cache_sha256") == cache_sha
    )
    checks = {
        "matrix_is_full": moment_ndim == 2,
        "collector_schema_matches": metadata.get("schema_version") == MEMIT_STATS_SCHEMA,
        "kind_is_full_second_moment": metadata.get("kind") == "full_second_moment",
        "collector_marks_formal": metadata.get("formal_memit_eligible") is True,
        "model_path_matches": model_path_matches,
        "model_config_matches": metadata.get("model_config_sha256")
        == expected_config_sha256,
        "module_matches": metadata.get("module_name") == expected_module_name,
        "sample_size_matches": recorded_sample_size == expected_sample_size,
        "documents_cover_sample": observed_documents is not None
        and observed_documents >= expected_sample_size,
        "model_uses_content_fingerprint": metadata.get("model_fingerprint_mode")
        == "content",
        "model_content_fingerprint_matches": expected_model_fingerprint is not None
        and metadata.get("model_fingerprint") == expected_model_fingerprint,
        "corpus_schema_matches": corpus.get("schema_version")
        == "engram-memit-corpus-cache/v1",
        "corpus_documents_cover_sample": corpus_documents is not None
        and corpus_documents >= expected_sample_size,
        "corpus_cache_matches": corpus_cache_matches,
        "corpus_source_is_auditable": source_is_auditable,
    }
    return checks


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def easyedit_stats_path(
    stats_dir: Path,
    model_key: str,
    module_name: str,
    sample_size: int,
) -> Path:
    return (
        stats_dir
        / model_key
        / "wikipedia_stats"
        / f"{module_name}_float32_mom2_{sample_size}.npz"
    )


def load_moment_file(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load our torch format or EasyEdit's CombinedStat NPZ format."""

    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            keys = set(payload.files)
            if {"mom2.mom2", "mom2.count"}.issubset(keys):
                count = int(payload["mom2.count"])
                moment = torch.from_numpy(payload["mom2.mom2"]).float() / count
                return moment, {"format": "easyedit_npz", "count": count}
            if "moment" in keys:
                return torch.from_numpy(payload["moment"]).float(), {
                    "format": "moment_npz",
                    "count": int(payload["count"]) if "count" in keys else None,
                }
            if "diag_moment" in keys:
                return torch.from_numpy(payload["diag_moment"]).float(), {
                    "format": "diag_moment_npz",
                    "count": int(payload["count"]) if "count" in keys else None,
                }
        raise ValueError(f"unrecognized MEMIT stats npz: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, torch.Tensor):
        return payload.float(), {"format": "bare_tensor"}
    if not isinstance(payload, dict):
        raise TypeError(f"unrecognized MEMIT stats payload: {type(payload)}")
    for key in ("moment", "diag_moment", "mom2"):
        if key in payload and isinstance(payload[key], torch.Tensor):
            metadata = dict(payload.get("metadata") or {})
            metadata["format"] = f"torch_{key}"
            return payload[key].float(), metadata
    raise KeyError(f"MEMIT stats file has no moment tensor: {path}")


def validate_moment(
    moment: torch.Tensor,
    module: torch.nn.Module,
    *,
    source: str,
) -> None:
    if not isinstance(module, torch.nn.Linear):
        raise TypeError(f"MEMIT expects an nn.Linear rewrite module, got {type(module)}")
    dimension = int(module.in_features)
    if moment.ndim == 1 and tuple(moment.shape) != (dimension,):
        raise ValueError(f"{source}: diagonal shape {tuple(moment.shape)} != {(dimension,)}")
    if moment.ndim == 2 and tuple(moment.shape) != (dimension, dimension):
        raise ValueError(
            f"{source}: second-moment shape {tuple(moment.shape)} != {(dimension, dimension)}"
        )
    if moment.ndim not in {1, 2}:
        raise ValueError(f"{source}: moment must be a vector or square matrix")
    if not torch.isfinite(moment).all():
        raise ValueError(f"{source}: moment contains non-finite values")


def proxy_stat_texts(rows: Sequence[dict[str, Any]], sample_size: int) -> list[str]:
    texts: list[str] = []
    for row in rows[:sample_size]:
        for field in ("update", "rephrase", "personas", "loc", "mhop"):
            value = row.get(field)
            if isinstance(value, str) and value.strip():
                texts.append(qa_prompt(value))
    if not texts:
        raise ValueError("no text is available for proxy statistics")
    return texts


@torch.inference_mode()
def collect_diagonal_moment(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    module_name: str,
    device: torch.device,
    *,
    batch_size: int,
    max_length: int,
    variance_floor: float,
) -> tuple[torch.Tensor, int]:
    module = get_module(model, module_name)
    if not isinstance(module, torch.nn.Linear):
        raise TypeError(f"expected Linear at {module_name}, got {type(module)}")
    total = torch.zeros(module.in_features, dtype=torch.float64, device=device)
    count = 0
    for start in range(0, len(texts), batch_size):
        encoded = _encoded_batch(tokenizer, texts[start : start + batch_size], device, max_length)
        holder: dict[str, torch.Tensor] = {}

        def hook(_module, inputs, _output):
            holder["inputs"] = inputs[0]

        handle = module.register_forward_hook(hook)
        try:
            model(**encoded, use_cache=False)
        finally:
            handle.remove()
        values = holder["inputs"].double()
        mask = encoded["attention_mask"].bool()
        values = values[mask]
        total.add_((values * values).sum(dim=0))
        count += int(values.shape[0])
    if count == 0:
        raise ValueError("proxy covariance observed zero tokens")
    moment = (total / count).float().clamp_min(variance_floor)
    return moment.cpu(), count


class CovarianceProvider:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        model_dir: Path,
        rows: Sequence[dict[str, Any]],
        device: torch.device,
        args: argparse.Namespace,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_dir = model_dir
        self.rows = rows
        self.device = device
        self.args = args
        self.cache: dict[int, torch.Tensor] = {}
        self.audit: dict[str, dict[str, Any]] = {}
        self._content_fingerprint: str | None = None
        if args.stats_source == "manifest_diag" and not args.allow_proxy_stats:
            raise ValueError(
                "manifest_diag is a smoke-only proxy; pass --allow-proxy-stats explicitly"
            )

    def _resolve_precomputed(self, layer: int, module_name: str) -> Path:
        if self.args.stats_file_template:
            return Path(
                self.args.stats_file_template.format(layer=layer, module=module_name)
            )
        model_key = model_stats_key(self.model_dir)
        candidates = [
            easyedit_stats_path(
                self.args.stats_dir,
                model_key,
                module_name,
                self.args.stats_sample_size,
            ),
            self.args.stats_dir / model_key / f"layer_{layer}_mom2.pt",
            self.args.stats_dir / f"layer_{layer}_mom2.pt",
        ]
        existing = [path for path in candidates if path.exists()]
        if not existing:
            rendered = "\n".join(f"  - {path}" for path in candidates)
            raise FileNotFoundError(
                "Qwen3 MEMIT statistics are missing. Expected one of:\n" + rendered
            )
        return existing[0]

    def get(self, layer: int) -> torch.Tensor:
        if layer in self.cache:
            return self.cache[layer]
        module_name = f"model.layers.{layer}.mlp.down_proj"
        module = get_module(self.model, module_name)
        if self.args.stats_source == "manifest_diag":
            texts = proxy_stat_texts(self.rows, self.args.proxy_stat_samples)
            moment, count = collect_diagonal_moment(
                self.model,
                self.tokenizer,
                texts,
                module_name,
                self.device,
                batch_size=self.args.representation_batch_size,
                max_length=self.args.max_length,
                variance_floor=self.args.proxy_variance_floor,
            )
            metadata = {
                "source": "selected WikiBigEdit prompts",
                "kind": "diagonal_proxy",
                "documents": min(len(self.rows), self.args.proxy_stat_samples),
                "texts": len(texts),
                "tokens": count,
                "formal_baseline_eligible": False,
            }
            source = "manifest_diag"
        else:
            path = self._resolve_precomputed(layer, module_name)
            moment, metadata = load_moment_file(path)
            resolved_path = path.resolve()
            source = str(resolved_path)
            metadata = {
                **metadata,
                "path": source,
                "file_sha256": sha256_file(resolved_path),
            }
            recorded_model = metadata.get("model_dir") or metadata.get("model_name")
            if recorded_model is None and not self.args.allow_unverified_stats:
                expected_key = model_stats_key(self.model_dir)
                if expected_key not in path.parts:
                    raise ValueError(
                        f"stats file {path} has no model identity and is not namespaced by "
                        f"{expected_key!r}; pass --allow-unverified-stats only after manually "
                        "verifying it was collected from this Qwen3 checkpoint"
                    )
            if recorded_model is not None:
                expected = str(self.model_dir.resolve())
                actual = str(Path(recorded_model).resolve())
                if actual != expected:
                    raise ValueError(f"stats model mismatch: {actual} != {expected}")
            recorded_module = metadata.get("module_name")
            if recorded_module is not None and recorded_module != module_name:
                raise ValueError(
                    f"stats module mismatch: {recorded_module!r} != {module_name!r}"
                )
            recorded_config_hash = metadata.get("model_config_sha256")
            config_path = self.model_dir / "config.json"
            if recorded_config_hash is not None:
                if recorded_config_hash != sha256_file(config_path):
                    raise ValueError(f"stats config hash does not match {config_path}")
            recorded_fingerprint = metadata.get("model_fingerprint")
            fingerprint_mode = metadata.get("model_fingerprint_mode")
            if fingerprint_mode == "content" and recorded_fingerprint is not None:
                if self._content_fingerprint is None:
                    self._content_fingerprint = model_content_fingerprint(self.model_dir)
                if recorded_fingerprint != self._content_fingerprint:
                    raise ValueError(
                        "stats model content fingerprint does not match the current "
                        f"checkpoint at {self.model_dir}"
                    )
            recorded_sample_size = metadata.get("sample_size_requested")
            if (
                recorded_sample_size is not None
                and int(recorded_sample_size) != self.args.stats_sample_size
            ):
                raise ValueError(
                    f"stats sample size {recorded_sample_size} != requested "
                    f"{self.args.stats_sample_size}"
                )
            provenance_checks = formal_stats_provenance_checks(
                metadata,
                moment_ndim=moment.ndim,
                expected_model_dir=self.model_dir,
                expected_model_fingerprint=self._content_fingerprint,
                expected_config_sha256=sha256_file(config_path),
                expected_module_name=module_name,
                expected_sample_size=self.args.stats_sample_size,
            )
            metadata = {
                **metadata,
                "formal_provenance_checks": provenance_checks,
                "formal_baseline_eligible": all(provenance_checks.values()),
            }
        validate_moment(moment, module, source=source)
        self.audit[str(layer)] = {
            "source": source,
            "shape": list(moment.shape),
            "dtype": str(moment.dtype),
            **metadata,
        }
        self.cache[layer] = moment.to(self.device)
        return self.cache[layer]


class MomentSolver:
    """Exact Woodbury solve for ``(lambda C + K K^T)^-1 K``.

    For a full C this caches a Cholesky factor of ``lambda C``.  This is
    algebraically equivalent to EasyEdit's dense solve while avoiding a new
    12k-by-12k factorization for every sequential edit.
    """

    def __init__(
        self,
        moment: torch.Tensor,
        update_weight: float,
        ridge: float,
    ) -> None:
        if update_weight <= 0:
            raise ValueError("mom2_update_weight must be positive")
        if ridge < 0:
            raise ValueError("solver ridge cannot be negative")
        if moment.ndim not in (1, 2):
            raise ValueError(f"MEMIT moment must be rank 1 or 2, got rank {moment.ndim}")
        if moment.ndim == 2 and moment.shape[0] != moment.shape[1]:
            raise ValueError(f"full MEMIT moment must be square, got {tuple(moment.shape)}")
        if not torch.isfinite(moment).all():
            raise FloatingPointError("MEMIT moment contains NaN or Inf")
        self.dimension = int(moment.shape[0])
        self.diagonal: torch.Tensor | None = None
        self.cholesky: torch.Tensor | None = None
        if moment.ndim == 1:
            diagonal = update_weight * moment.double() + ridge
            if not torch.isfinite(diagonal).all():
                raise FloatingPointError("regularized diagonal MEMIT moment is non-finite")
            if torch.any(diagonal <= 0):
                minimum = float(diagonal.min())
                raise ValueError(
                    "diagonal MEMIT moment is not positive after regularization "
                    f"(minimum={minimum:.6g}, ridge={ridge:.6g})"
                )
            self.diagonal = diagonal
        else:
            base = update_weight * moment.double()
            if ridge:
                base.diagonal().add_(ridge)
            if not torch.isfinite(base).all():
                raise FloatingPointError("regularized full MEMIT moment is non-finite")
            try:
                self.cholesky = torch.linalg.cholesky(base)
            except RuntimeError as error:
                diagonal = base.diagonal()
                raise ValueError(
                    "MEMIT moment Cholesky failed; the selected layer statistics are "
                    "not positive definite at the configured regularization "
                    f"(ridge={ridge:.6g}, diagonal_min={float(diagonal.min()):.6g}, "
                    f"diagonal_max={float(diagonal.max()):.6g}). Increase --solver-ridge "
                    "or regenerate the checkpoint-specific moment."
                ) from error

    def base_solve(self, rhs: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(rhs).all():
            raise FloatingPointError("MEMIT solve right-hand side contains NaN or Inf")
        rhs = rhs.double()
        if self.diagonal is not None:
            return rhs / self.diagonal[:, None]
        assert self.cholesky is not None
        return torch.cholesky_solve(rhs, self.cholesky)

    def adjusted_keys(self, keys: torch.Tensor) -> torch.Tensor:
        if keys.ndim != 2 or keys.shape[0] != self.dimension:
            raise ValueError(f"keys must have shape [{self.dimension}, n]")
        inverse_keys = self.base_solve(keys)
        small = torch.eye(keys.shape[1], dtype=torch.float64, device=keys.device)
        small.add_(keys.double().T @ inverse_keys)
        try:
            adjusted = torch.linalg.solve(small, inverse_keys.T).T
        except RuntimeError as error:
            raise ValueError("MEMIT Woodbury correction solve failed") from error
        if not torch.isfinite(adjusted).all():
            raise FloatingPointError("MEMIT adjusted keys contain NaN or Inf")
        return adjusted


def direct_adjusted_keys(
    moment: torch.Tensor,
    keys: torch.Tensor,
    update_weight: float,
    ridge: float,
) -> torch.Tensor:
    """Reference dense solve used by CPU unit tests."""

    if moment.ndim == 1:
        base = torch.diag(update_weight * moment.double() + ridge)
    else:
        base = update_weight * moment.double().clone()
        if ridge:
            base.diagonal().add_(ridge)
    return torch.linalg.solve(base + keys.double() @ keys.double().T, keys.double())


@dataclass
class EditBatchStats:
    stage: int
    batch_index: int
    sample_ids: list[str]
    wall_seconds: float
    layer_update_norms: dict[str, float]
    layer_weight_norms: dict[str, float]
    layer_update_weight_ratios: dict[str, float]
    cumulative_weight_drift_ratios: dict[str, float]
    z_optimizations: list[dict[str, Any]]


@dataclass
class CumulativeWeightDriftAudit:
    """Audit selected weights against their immutable pre-edit values.

    This deliberately has no rejection threshold. ``measure`` is side-effect
    free so a failed MEMIT batch cannot advance the recorded maxima; callers
    record observations only after the whole cross-layer transaction succeeds.
    """

    initial_weights: dict[int, torch.Tensor]
    initial_weight_norms: dict[int, float]
    max_ratios: dict[str, float]

    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        layers: Sequence[int],
    ) -> "CumulativeWeightDriftAudit":
        initial_weights: dict[int, torch.Tensor] = {}
        initial_weight_norms: dict[int, float] = {}
        max_ratios: dict[str, float] = {}
        for layer in layers:
            weight = get_module(
                model, f"model.layers.{layer}.mlp.down_proj"
            ).weight.detach()
            initial = weight.clone()
            norm = float(torch.linalg.vector_norm(initial.float()))
            if not math.isfinite(norm) or norm <= 0:
                raise FloatingPointError(
                    f"invalid initial MEMIT weight norm for layer {layer}: {norm}"
                )
            initial_weights[layer] = initial
            initial_weight_norms[layer] = norm
            max_ratios[str(layer)] = 0.0
        return cls(initial_weights, initial_weight_norms, max_ratios)

    def measure(
        self,
        model: torch.nn.Module,
        layers: Sequence[int],
    ) -> dict[str, float]:
        ratios: dict[str, float] = {}
        for layer in layers:
            if layer not in self.initial_weights or layer not in self.initial_weight_norms:
                raise ValueError(f"missing initial MEMIT weight audit for layer {layer}")
            weight = get_module(
                model, f"model.layers.{layer}.mlp.down_proj"
            ).weight.detach()
            initial = self.initial_weights[layer]
            if weight.shape != initial.shape or weight.device != initial.device:
                raise ValueError(
                    f"MEMIT weight audit mismatch at layer {layer}: "
                    f"current={tuple(weight.shape)} on {weight.device}, "
                    f"initial={tuple(initial.shape)} on {initial.device}"
                )
            drift_norm = float(
                torch.linalg.vector_norm((weight - initial).float())
            )
            ratio = drift_norm / self.initial_weight_norms[layer]
            if not math.isfinite(ratio) or ratio < 0:
                raise FloatingPointError(
                    f"non-finite cumulative MEMIT weight drift at layer {layer}"
                )
            ratios[str(layer)] = ratio
        return ratios

    def record(self, ratios: dict[str, float]) -> None:
        for layer, ratio in ratios.items():
            if not math.isfinite(ratio) or ratio < 0:
                raise FloatingPointError(
                    f"invalid cumulative MEMIT weight drift for layer {layer}"
                )
        for layer, ratio in ratios.items():
            self.max_ratios[layer] = max(self.max_ratios.get(layer, 0.0), ratio)

    def completed_summary(
        self,
        model: torch.nn.Module,
        layers: Sequence[int],
    ) -> dict[str, dict[str, float]]:
        final_ratios = self.measure(model, layers)
        return {
            "final_cumulative_weight_drift_ratios": final_ratios,
            "max_cumulative_weight_drift_ratios": {
                str(layer): self.max_ratios.get(str(layer), 0.0)
                for layer in layers
            },
        }


def build_context_templates(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    mode: str,
) -> list[list[str]]:
    if mode == "static":
        return [["{}"]]
    prompts = ["The", "Therefore", "Because", "I", "You"]
    with tokenizer_padding(tokenizer, "left"):
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32,
            add_special_tokens=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generated = model.generate(
            **encoded,
            do_sample=True,
            top_k=5,
            max_new_tokens=10,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    cleaned = [
        unicodedata.normalize("NFKD", text)
        .replace("\n\n", " ")
        .replace("{", " ")
        .replace("}", " ")
        + ". {}"
        for text in texts
    ]
    return [["{}"], cleaned]


def apply_memit_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[dict[str, Any]],
    context_groups: Sequence[Sequence[str]],
    covariance: CovarianceProvider,
    solvers: dict[int, MomentSolver],
    device: torch.device,
    args: argparse.Namespace,
    *,
    stage: int,
    batch_index: int,
    z_function=None,
    cumulative_drift_audit: CumulativeWeightDriftAudit | None = None,
) -> EditBatchStats:
    started = time.perf_counter()
    z_layer = args.layers[-1]
    z_targets: list[torch.Tensor] = []
    z_stats: list[ZOptimizationStats] = []
    for request in requests:
        target, stats = (z_function or compute_z_star)(
            model,
            tokenizer,
            request,
            context_groups,
            z_layer,
            args.v_loss_layer,
            device,
            args,
        )
        z_targets.append(target)
        z_stats.append(stats)
    zs = torch.stack(z_targets, dim=1)
    if not torch.isfinite(zs).all():
        raise FloatingPointError("MEMIT optimized targets contain NaN or Inf")

    update_norms: dict[str, float] = {}
    weight_norms: dict[str, float] = {}
    update_weight_ratios: dict[str, float] = {}
    cumulative_weight_drift_ratios: dict[str, float] = {}
    max_ratio = float(
        getattr(args, "max_update_weight_ratio", DEFAULT_MAX_UPDATE_WEIGHT_RATIO)
    )
    if not math.isfinite(max_ratio) or max_ratio <= 0:
        raise ValueError("max_update_weight_ratio must be a finite positive number")

    # Later-layer residuals must observe earlier-layer updates, so calculate and
    # apply them in algorithm order. Exact snapshots make the whole batch
    # transactional: a later numerical failure restores every selected layer.
    originals = {
        layer: get_module(model, f"model.layers.{layer}.mlp.down_proj")
        .weight.detach().clone()
        for layer in args.layers
    }
    try:
        for layer_offset, layer in enumerate(args.layers):
            module_name = f"model.layers.{layer}.mlp.down_proj"
            module = get_module(model, module_name)
            if not torch.isfinite(module.weight).all():
                raise FloatingPointError(f"non-finite MEMIT weight before {module_name} update")
            keys = compute_keys(
                model,
                tokenizer,
                requests,
                layer,
                context_groups,
                device,
                batch_size=args.representation_batch_size,
                max_length=args.max_length,
            ).T
            current_zs = compute_current_zs(
                model,
                tokenizer,
                requests,
                z_layer,
                device,
                batch_size=args.representation_batch_size,
                max_length=args.max_length,
            ).T
            if not torch.isfinite(keys).all() or not torch.isfinite(current_zs).all():
                raise FloatingPointError(
                    f"non-finite MEMIT representation while updating layer {layer}"
                )
            targets = zs.to(current_zs) - current_zs
            residual = targets.double() / (len(args.layers) - layer_offset)
            if not torch.isfinite(residual).all():
                raise FloatingPointError(f"non-finite MEMIT residual at layer {layer}")
            if layer not in solvers:
                solvers[layer] = MomentSolver(
                    covariance.get(layer),
                    args.mom2_update_weight,
                    args.solver_ridge,
                )
            adjusted = solvers[layer].adjusted_keys(keys)
            update = residual @ adjusted.T
            if tuple(update.shape) != tuple(module.weight.shape):
                if tuple(update.T.shape) == tuple(module.weight.shape):
                    update = update.T
                else:
                    raise ValueError(
                        f"MEMIT update {tuple(update.shape)} does not match {module_name} "
                        f"weight {tuple(module.weight.shape)}"
                    )
            if not torch.isfinite(update).all():
                raise FloatingPointError(f"non-finite MEMIT update for layer {layer}")
            update_norm = float(torch.linalg.vector_norm(update.double()))
            weight_norm = float(torch.linalg.vector_norm(module.weight.float()))
            ratio = update_norm / weight_norm if weight_norm > 0 else math.inf
            update_norms[str(layer)] = update_norm
            weight_norms[str(layer)] = weight_norm
            update_weight_ratios[str(layer)] = ratio
            if not math.isfinite(ratio) or ratio > max_ratio:
                raise FloatingPointError(
                    f"MEMIT layer {layer} update rejected before commit: "
                    f"||delta W||/||W||={ratio:.6g} exceeds "
                    f"--max-update-weight-ratio={max_ratio:.6g}"
                )
            typed_update = update.to(dtype=module.weight.dtype)
            if not torch.isfinite(typed_update).all():
                raise FloatingPointError(
                    f"MEMIT layer {layer} update overflows {module.weight.dtype}"
                )
            with torch.no_grad():
                module.weight.add_(typed_update)
            if not torch.isfinite(module.weight).all():
                raise FloatingPointError(f"non-finite MEMIT weight after layer {layer} update")
        if cumulative_drift_audit is not None:
            # Keep this inside the transaction: an invalid post-update audit
            # must restore every layer changed by the batch.
            cumulative_weight_drift_ratios = cumulative_drift_audit.measure(
                model, args.layers
            )
    except Exception:
        with torch.no_grad():
            for layer, original in originals.items():
                get_module(
                    model, f"model.layers.{layer}.mlp.down_proj"
                ).weight.copy_(original)
        raise

    if cumulative_drift_audit is not None:
        cumulative_drift_audit.record(cumulative_weight_drift_ratios)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return EditBatchStats(
        stage=stage,
        batch_index=batch_index,
        sample_ids=[request["sample_id"] for request in requests],
        wall_seconds=time.perf_counter() - started,
        layer_update_norms=update_norms,
        layer_weight_norms=weight_norms,
        layer_update_weight_ratios=update_weight_ratios,
        cumulative_weight_drift_ratios=cumulative_weight_drift_ratios,
        z_optimizations=[asdict(stats) for stats in z_stats],
    )


def chunks(rows: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size <= 0:
        raise ValueError("edit batch size must be positive")
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def should_evaluate_stage(
    stage: int,
    stages: Sequence[int],
    *,
    no_stage_eval: bool,
) -> bool:
    """Always evaluate the final state, even when intermediate eval is disabled."""

    if not stages:
        return False
    return not no_stage_eval or stage == max(stages)


def validate_qwen3_model(model: torch.nn.Module, args: argparse.Namespace) -> None:
    if getattr(model.config, "model_type", None) != "qwen3":
        raise ValueError(f"runner is scoped to Qwen3, got {model.config.model_type!r}")
    layer_count = int(model.config.num_hidden_layers)
    if not args.layers:
        raise ValueError("at least one rewrite layer is required")
    if len(args.layers) != len(set(args.layers)) or list(args.layers) != sorted(args.layers):
        raise ValueError("rewrite layers must be unique and strictly increasing")
    if min(args.layers) < 0 or max(args.layers) >= layer_count:
        raise ValueError(f"rewrite layers must lie in [0, {layer_count - 1}]")
    if args.v_loss_layer < max(args.layers) or args.v_loss_layer >= layer_count:
        raise ValueError("v-loss-layer must be at or after every rewrite layer")
    for layer in args.layers:
        module = get_module(model, f"model.layers.{layer}.mlp.down_proj")
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"layer {layer} down_proj is not nn.Linear")


def main() -> None:
    run_started = time.time()
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.allow_existing_output:
        raise FileExistsError(args.output_dir)
    if not math.isfinite(args.max_update_weight_ratio) or args.max_update_weight_ratio <= 0:
        raise ValueError("max-update-weight-ratio must be a finite positive number")
    partial_protocol = bool(
        args.max_stages
        or args.examples_per_stage
        or args.locality_panel_size
        or args.multihop_panel_size
        or args.no_stage_eval
    )
    if partial_protocol and not args.development_only:
        raise ValueError(
            "partial stages/panels or disabled stage evaluation require --development-only"
        )
    formal_request = not args.development_only and not partial_protocol
    if formal_request and args.selection_certificate is None:
        raise ValueError(
            "A full MEMIT benchmark requires --selection-certificate from the "
            "independent development sweep"
        )
    development_selection = None
    if args.selection_certificate is not None:
        from baselines.editor_dev_selection import verify_selection

        development_selection = verify_selection(
            args.selection_certificate, args, "MEMIT", list(args.layers)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    all_rows, manifest_metadata = load_manifest(args.manifest)
    stages = restrict_stages(
        group_by_stage(all_rows),
        max_stages=args.max_stages,
        examples_per_stage=args.examples_per_stage,
    )
    selected_rows = [row for stage_rows in stages.values() for row in stage_rows]
    if not selected_rows:
        raise ValueError("no selected WikiBigEdit rows")
    request_alignment = Counter(
        make_request(row, index)["subject_match"]
        for index, row in enumerate(selected_rows)
    )
    locality_panel = manifest_panel(
        manifest_metadata, "locality", limit=args.locality_panel_size
    ) or fixed_panel(
        all_rows, "loc", "loc_ans", args.locality_panel_size, args.seed + 1001
    )
    multihop_panel = manifest_panel(
        manifest_metadata, "mhop", limit=args.multihop_panel_size
    ) or fixed_panel(
        all_rows, "mhop", "mhop_ans", args.multihop_panel_size, args.seed + 2001
    )

    tokenizer = load_tokenizer(args.model_dir)
    model = load_model(args.model_dir, device)
    validate_qwen3_model(model, args)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    context_groups = build_context_templates(
        model, tokenizer, device, args.context_mode
    )
    covariance = CovarianceProvider(
        model, tokenizer, args.model_dir, all_rows, device, args
    )
    # Materialize every statistic before the first edit.  In particular, the
    # smoke-only manifest proxy must describe the unedited base model rather
    # than a model already changed by earlier rewrite layers.
    solvers: dict[int, MomentSolver] = {
        layer: MomentSolver(
            covariance.get(layer),
            args.mom2_update_weight,
            args.solver_ridge,
        )
        for layer in args.layers
    }
    cumulative_drift_audit = CumulativeWeightDriftAudit.from_model(
        model, args.layers
    )

    original_weights: dict[int, torch.Tensor] | None = None
    if args.save_delta is not None:
        original_weights = {
            layer: get_module(model, f"model.layers.{layer}.mlp.down_proj")
            .weight.detach().cpu().clone()
            for layer in args.layers
        }

    if args.development_only:
        reproduction_status = "qwen3_development_candidate"
    elif args.stats_source == "manifest_diag":
        reproduction_status = "smoke_only_manifest_diagonal_proxy"
    elif development_selection is not None:
        reproduction_status = "qwen3_independent_dev_selected"
    elif args.selection_provenance:
        reproduction_status = "qwen3_adapted_sensitivity_variant_unqualified"
    else:
        reproduction_status = "qwen3_transfer_preset_untuned"
    full_qwen3_statistics = args.stats_source == "precomputed" and all(
        entry.get("formal_baseline_eligible") is True
        for entry in covariance.audit.values()
    )
    formal_baseline_eligible = bool(
        full_qwen3_statistics and formal_request and development_selection is not None
    )
    config = {
        **vars(args),
        "layers": list(args.layers),
        "model_dir": str(args.model_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "stats_dir": str(args.stats_dir.resolve()),
        "save_delta": str(args.save_delta.resolve()) if args.save_delta else None,
        "selection_certificate": (
            str(args.selection_certificate.resolve()) if args.selection_certificate else None
        ),
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_protocol_id": manifest_metadata.get("protocol_id"),
        "method": "MEMIT",
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_qwen25_preset": UPSTREAM_QWEN25_PRESET,
        "reproduction_status": reproduction_status,
        "development_only": bool(args.development_only),
        "development_selection": development_selection,
        "formal_baseline_eligible": formal_baseline_eligible,
        "formal_baseline_qualification": (
            "development-only Qwen3 layer/configuration candidate; do not report as a benchmark"
            if args.development_only
            else (
            (
                "Qwen3-adapted sensitivity variant with full checkpoint-specific "
                "moments; not an official upstream preset"
                if development_selection is not None
                else "unqualified Qwen3 transfer with full checkpoint-specific moments"
            )
            if full_qwen3_statistics
            else "smoke-only proxy statistics"
            )
        ),
        "qwen3_layer_selection_status": (
            (
                f"independent development selection: {args.selection_certificate.resolve()}"
                if development_selection is not None
                else None
            )
            or args.selection_provenance
            or (
                "unvalidated development candidate layers 5-8; layer 4 removed "
                "after the stability audit"
                if args.default_qwen3_candidate
                else "explicit caller-selected Qwen3 layers; no held-out validation recorded"
            )
        ),
        "qwen3_layer_selection_validated": development_selection is not None,
        "qwen3_required_stats": [
            f"model.layers.{layer}.mlp.down_proj input second moment"
            for layer in args.layers
        ],
        "statistics_provenance": {
            layer: {
                "path": entry.get("path"),
                "file_sha256": entry.get("file_sha256"),
            }
            for layer, entry in covariance.audit.items()
        },
        "statistics_audit": covariance.audit,
        "subject_alignment_counts": dict(request_alignment),
        "edit_prompt_template": "Question: <question-with-{subject}>\\nAnswer:",
        "closed_book_evaluation": True,
        "evaluation_prompt_template": "Question: <question>\\nAnswer:",
        "context_templates": context_groups,
        "num_facts": len(selected_rows),
        "stage_counts": {str(stage): len(stage_rows) for stage, stage_rows in stages.items()},
        "locality_panel_size_actual": len(locality_panel),
        "multihop_panel_size_actual": len(multihop_panel),
        **environment_metadata(),
    }
    atomic_json(args.output_dir / "run_config.json", config)

    metrics: list[dict[str, Any]] = []
    if not args.skip_base_eval:
        metrics.append(
            evaluate_and_write(
                model,
                tokenizer,
                selected_rows,
                locality_panel,
                multihop_panel,
                device,
                args,
                checkpoint="base",
                current_stage=None,
            )
        )
        atomic_json(args.output_dir / "metrics.json", metrics)

    case_id = 0
    for stage, stage_rows in stages.items():
        for batch_index, batch_rows in enumerate(chunks(stage_rows, args.edit_batch_size)):
            requests = [make_request(row, case_id + offset) for offset, row in enumerate(batch_rows)]
            case_id += len(requests)
            edit_stats = apply_memit_batch(
                model,
                tokenizer,
                requests,
                context_groups,
                covariance,
                solvers,
                device,
                args,
                stage=stage,
                batch_index=batch_index,
                cumulative_drift_audit=cumulative_drift_audit,
            )
            append_jsonl(args.output_dir / "edit_trace.jsonl", [asdict(edit_stats)])
            atomic_json(args.output_dir / "stats_audit.json", covariance.audit)
        if should_evaluate_stage(
            stage,
            list(stages),
            no_stage_eval=args.no_stage_eval,
        ):
            seen_rows = [
                row
                for seen_stage, seen_stage_rows in stages.items()
                if seen_stage <= stage
                for row in seen_stage_rows
            ]
            metrics.append(
                evaluate_and_write(
                    model,
                    tokenizer,
                    seen_rows,
                    locality_panel,
                    multihop_panel,
                    device,
                    args,
                    checkpoint=f"stage_{stage}",
                    current_stage=stage,
                )
            )
            atomic_json(args.output_dir / "metrics.json", metrics)

    if args.save_delta is not None:
        assert original_weights is not None
        args.save_delta.parent.mkdir(parents=True, exist_ok=True)
        delta = {
            f"model.layers.{layer}.mlp.down_proj.weight": (
                get_module(model, f"model.layers.{layer}.mlp.down_proj")
                .weight.detach().cpu()
                - original_weights[layer]
            )
            for layer in args.layers
        }
        torch.save(
            {
                "method": "MEMIT",
                "reproduction_status": reproduction_status,
                "layers": list(args.layers),
                "delta": delta,
            },
            args.save_delta,
        )

    cumulative_drift_summary = cumulative_drift_audit.completed_summary(
        model, args.layers
    )
    atomic_json(
        args.output_dir / "completed.json",
        {
            "status": "complete",
            "method": "MEMIT",
            "num_facts": len(selected_rows),
            "edits": case_id,
            "final_stage": max(stages) if stages else None,
            "elapsed_seconds": time.time() - run_started,
            "reproduction_status": reproduction_status,
            "formal_baseline_eligible": formal_baseline_eligible,
            "development_only": bool(args.development_only),
            "stats_audit": covariance.audit,
            **cumulative_drift_summary,
        },
    )


if __name__ == "__main__":
    main()
