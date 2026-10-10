"""Read-only safety boundaries for the full-parameter continual writer.

Integration: use ``safe_query`` instead of the runner's filtered_query,
``choose_safe_probe_and_replay_pools`` instead of its pool helper, and
``strict_mastery_match`` for the generation check before allowing IGNORE.
Keep the NLL/duplicate/retirement gates and benchmark scoring separately.

Ambiguity is conservative exclusion, never implicit retirement. Each query
scans only O(N) historical metadata; vector similarities are computed solely
for hash/exact-key candidates after the existing max_candidates limit.
"""

from __future__ import annotations

import copy
import unicodedata
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch

from baselines.engram_continual_core import FactMemory


def strict_normalize_text(value: Any) -> str:
    """NFC, uniform line endings and horizontal whitespace; no content loss.

Case, punctuation, compatibility characters, internal line breaks and all
lines are retained. Outer whitespace is ignored. Non-string/empty values
cannot certify mastery.
"""
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(" ".join(line.split()) for line in value.split("\n")).strip()


def strict_mastery_match(prediction: Any, target: Any) -> bool:
    expected = strict_normalize_text(target)
    return bool(expected) and strict_normalize_text(prediction) == expected


def _sample_id(row: Mapping[str, Any]) -> str:
    value = row.get("sample_id")
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value):
        raise ValueError("Safety checks require a nonempty sample_id for every history row")
    return str(value)


def _excluded_indices(memory: FactMemory, row: Mapping[str, Any], retired_ids: Sequence[str]) -> tuple[set[int], set[int]]:
    retired = {str(value) for value in retired_ids}
    # Both are read-only operations. Do not install fake supersession links or
    # temporarily mutate a shared memory to obtain a filtered query.
    replacement = memory._replacement_indices(row)
    excluded = set(memory._superseded_by) | replacement
    excluded.update(index for index, old in enumerate(memory.rows) if _sample_id(old) in retired)
    return excluded, replacement


def _conflict_indices(memory: FactMemory, row: Mapping[str, Any], active: Sequence[int]) -> set[int]:
    groups: dict[tuple, tuple[set, list[int]]] = {}

    def add(key: tuple, value: tuple | str, index: int | None) -> None:
        distinct, indices = groups.setdefault(key, (set(), []))
        distinct.add(value)
        if index is not None:
            indices.append(index)

    # Include the current row once as a constraint, but never add it to the
    # returned history IDs or insert it into the memory/index/vector store.
    for index in [*active, None]:
        current = row if index is None else memory.rows[index]
        slot, fact = memory._slot_key(current), memory._fact_key(current)
        answer = strict_normalize_text(current.get("ans"))
        if slot is not None and fact is not None:
            # Core keys use NFKC/case folding. The strict actual target must
            # supplement them: ①/1 and US/us cannot be certified duplicates.
            add(("metadata_slot", slot), (fact, answer), index)
        prompt = strict_normalize_text(current.get("update"))
        if prompt and answer:
            # Scope/subject metadata is not passed to qa_prompt. Identical
            # model questions with different targets remain incompatible even
            # when metadata describes different scopes or different entities.
            add(("model_question", prompt), answer, index)
    return {index for distinct, indices in groups.values() if len(distinct) > 1 for index in indices}


def conflicting_history_ids(memory: FactMemory, row: Mapping[str, Any], retired_ids=()) -> set[str]:
    """Find both endpoints of unresolved active-history/current conflicts.

The metadata rule is same slot with different fact/strict target. The actual
training-input rule is same normalized update with different strict answer,
independent of scope. This does not decide that either old fact is false.
"""
    excluded, _ = _excluded_indices(memory, row, retired_ids)
    active = [index for index in range(len(memory.rows)) if index not in excluded]
    return {_sample_id(memory.rows[index]) for index in _conflict_indices(memory, row, active)}


def _strict_duplicate(memory: FactMemory, old: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    if not strict_mastery_match(old.get("ans"), row.get("ans")):
        return False
    # Oracle structured identity may certify question paraphrases. Exact-QA
    # fallback cannot inherit the core's lossy NFKC/case-fold equivalence.
    if memory.fact_mode == "structured" and memory._structured_slot(row) is not None:
        return True
    return strict_mastery_match(old.get("update"), row.get("update"))


def safe_query(memory: FactMemory, row: Mapping[str, Any], embedding: Any, retired_ids=()) -> dict[str, Any]:
    """Core-compatible query with retirement before shortlist truncation.

Reads the core's bucket/exact-key indices and stored vectors without changing
them. Ambiguous metadata lists are uncapped and include conflicts elsewhere
in active history, so random replay and reward probes can exclude both ends.
"""
    vector = memory._vector(embedding)
    excluded, replacement = _excluded_indices(memory, row, retired_ids)
    active = [index for index in range(len(memory.rows)) if index not in excluded]
    conflicts = _conflict_indices(memory, row, active)
    fact_key, slot_key = memory._fact_key(row), memory._slot_key(row)
    exact = (set(memory._fact_index.get(fact_key, ())) if fact_key is not None else set()) - excluded
    exact = {index for index in exact if index not in conflicts and _strict_duplicate(memory, memory.rows[index], row)}
    same_slot = (set(memory._slot_index.get(slot_key, ())) if slot_key is not None else set()) - excluded
    ambiguous = conflicts | (same_slot - exact)
    votes: dict[int, int] = defaultdict(int)
    for table, key in zip(memory._buckets, memory._hashes(vector)):
        for index in table.get(key, ()):
            if index not in excluded:
                votes[index] += 2
        for bit in range(memory.bits):
            for index in table.get(key ^ (1 << bit), ()):
                if index not in excluded:
                    votes[index] += 1
    recalled = set(votes) | exact | same_slot
    shortlist = sorted(recalled, key=lambda index: (index in exact, index in same_slot, votes.get(index, 0), index),
                       reverse=True)[:memory.max_candidates]
    candidates = []
    if shortlist:
        scores = torch.stack([memory._vectors[index] for index in shortlist]) @ vector
        for index, score in zip(shortlist, scores.tolist()):
            candidates.append({"row": copy.deepcopy(memory.rows[index]),
                               "similarity": min(1.0, max(-1.0, float(score))),
                               "equivalent": index in exact, "same_slot": index in same_slot,
                               "memory_index": index, "lsh_votes": votes.get(index, 0),
                               "ambiguous": index in ambiguous})
        candidates.sort(key=lambda item: (item["similarity"], item["memory_index"]), reverse=True)
    return {
        "candidates": candidates,
        "similar_rows": [item["row"] for item in candidates if item["similarity"] >= memory.similarity_threshold
                         and not item["equivalent"] and not item["same_slot"] and not item["ambiguous"]],
        "duplicate_rows": [copy.deepcopy(memory.rows[index]) for index in sorted(exact, reverse=True)],
        "ambiguous_rows": [copy.deepcopy(memory.rows[index]) for index in sorted(ambiguous, reverse=True)],
        "replacement_rows": [copy.deepcopy(memory.rows[index]) for index in sorted(replacement, reverse=True)],
        "top_similarity": candidates[0]["similarity"] if candidates else 0.0,
        "candidate_count": len(candidates), "hash_recalled_count": len(recalled),
        "fact_mode": memory.fact_mode,
        "equivalence_method": "core_exact_keys_plus_strict_NFC_targets_and_global_prompt_compatibility",
        "conflicting_history_ids": sorted({_sample_id(memory.rows[index]) for index in conflicts}),
        "excluded_history_ids": sorted({_sample_id(memory.rows[index]) for index in excluded}),
    }


def choose_safe_probe_and_replay_pools(memory: FactMemory, query: Mapping[str, Any], row: Mapping[str, Any],
                                     probe_rng, probe_size: int, retired_ids=()):
    """Preserve the runner's pre-action sampling, with active safe history only."""
    if isinstance(probe_size, bool) or not isinstance(probe_size, int) or probe_size < 0:
        raise ValueError("probe_size must be a nonnegative integer")
    excluded, _ = _excluded_indices(memory, row, retired_ids)
    active = [index for index in range(len(memory.rows)) if index not in excluded]
    conflicts = _conflict_indices(memory, row, active)
    blocked = {_sample_id(old) for key in ("ambiguous_rows", "duplicate_rows") for old in query[key]}
    blocked.add(_sample_id(row))
    eligible = [copy.deepcopy(memory.rows[index]) for index in active
                if index not in conflicts and _sample_id(memory.rows[index]) not in blocked]
    probes = probe_rng.sample(eligible, min(probe_size, len(eligible) // 2))
    probe_ids = {_sample_id(old) for old in probes}
    available = [old for old in eligible if _sample_id(old) not in probe_ids]
    available_by_id = {_sample_id(old): old for old in available}
    related = [available_by_id[_sample_id(old)] for old in query["similar_rows"] if _sample_id(old) in available_by_id]
    return probes, available, related
