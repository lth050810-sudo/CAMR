"""CAMR replay selection from an already certified historical pool.

This module does not retrieve facts or relax the caller's safety boundary.
Only ``available`` rows can be selected: the caller must first remove current,
unseen, duplicate, retired, conflicting and current reward-probe records.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def _identifier(row: Mapping[str, Any]) -> str:
    value = row.get("sample_id")
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value):
        raise ValueError("Every replay row requires a nonempty sample_id")
    return str(value)


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def select_replay(
    action: str,
    available: Sequence[Mapping[str, Any]],
    related: Sequence[Mapping[str, Any]],
    k: int,
    rng: Any,
    history_stats: Mapping[str, Mapping[str, Any]],
    event_index: int,
    similar_k: int = 2,
    recent_k: int = 0,
    recent_window: int = 64,
    ignore_replay: bool = False,
) -> tuple[list[Mapping[str, Any]], list[str]]:
    """Choose at most ``k`` distinct safe historical rows without mutation.

    MERGE first takes up to ``similar_k`` available related rows in the supplied
    similarity order. Then both MERGE and ADD reserve up to ``recent_k`` slots
    for safe recent facts not already selected. Half of the remaining slots,
    rounded up, go to coverage; the rest are sampled uniformly without
    replacement. By default IGNORE returns empty lists without consuming RNG
    state. With ``ignore_replay=True``, IGNORE uses the same recent, coverage
    and random selection as ADD, for maintenance of safe history only. It
    never selects from ``related`` or adds the current fact; the caller must
    exclude current-equivalent facts from ``available`` before calling.
    Thus with k=4 and sufficient related history, MERGE is 2 similar + 1
    coverage + 1 random, while ADD is 2 coverage + 2 random. Missing related
    rows are filled using coverage and random history, never an unsafe pool.
    Quotas are maxima, and missing recent/related rows also fall back to
    coverage/random.

    A recent fact has ``inserted_event`` in the previous ``recent_window``
    arrival events, excluding the current event. Missing insertion metadata
    does not certify recency. Among these rows, prioritize the fewest previous
    replays, then the oldest last replay/insertion, then sample ID. This gives
    never-replayed recent facts priority without repeatedly taking the newest
    row. On a stream of unique arrivals the window is the latest 64 facts;
    repeated arrivals still consume events. No per-fact losses or evaluation
    data are consulted. ``recent_k=0`` preserves the legacy selections and RNG
    sequence exactly.

    ``history_stats`` uses string sample IDs. Each record may contain
    ``inserted_event``, ``last_replayed_event`` and ``replay_count``. Coverage
    prioritizes the oldest last replay (insertion for never-replayed rows),
    then the smallest replay count, then sample ID. A missing insertion event
    defaults to -1 for compatibility with a preexisting historical pool.
    Future or current-event timestamps are rejected: inputs describe strictly
    earlier events. NLL is deliberately not used as a proxy for forgetting.

    Returned rows are the original objects from ``available``; repeated IDs
    are collapsed to its first occurrence. ``sources`` aligns with rows and
    contains similar_history, recent_history, coverage_history or
    random_history. No input rows or statistics are changed. Save
    ``rng.getstate()`` and history_stats
    in checkpoints; stable sample-ID ordering makes replay reproducible even
    if the available pool is rebuilt in a different order after resume.
    """
    if action not in {"add", "merge", "ignore"}:
        raise ValueError("action must be add, merge or ignore")
    _nonnegative_integer(k, "k")
    _nonnegative_integer(similar_k, "similar_k")
    _nonnegative_integer(recent_k, "recent_k")
    _nonnegative_integer(recent_window, "recent_window")
    if recent_k and not recent_window:
        raise ValueError("recent_window must be positive when recent replay is enabled")
    _nonnegative_integer(event_index, "event_index")
    if (action == "ignore" and not ignore_replay) or k == 0:
        return [], []

    by_id: dict[str, Mapping[str, Any]] = {}
    for row in available:
        by_id.setdefault(_identifier(row), row)
    budget = min(k, len(by_id))
    if budget == 0:
        return [], []

    def coverage_key(identifier: str) -> tuple[int, int, str]:
        stats = history_stats.get(identifier, {})
        inserted = stats.get("inserted_event", -1)
        last_replayed = stats.get("last_replayed_event")
        count = _nonnegative_integer(stats.get("replay_count", 0), "replay_count")
        for name, value in (("inserted_event", inserted), ("last_replayed_event", last_replayed)):
            if value is None and name == "last_replayed_event":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or not -1 <= value < event_index:
                raise ValueError(f"{name} for {identifier} must precede event_index")
        # An old insertion must not override a recent successful replay.
        last_seen = inserted if last_replayed is None else last_replayed
        return last_seen - event_index, count, identifier

    # Validate all pool statistics before changing the RNG, including related
    # rows selected first. This keeps failures transactional for the sampler.
    coverage_order = sorted(by_id, key=coverage_key)
    selected: list[str] = []
    selected_ids: set[str] = set()
    sources: list[str] = []
    if action == "merge" and similar_k:
        for row in related:
            identifier = _identifier(row)
            if identifier in by_id and identifier not in selected_ids:
                selected.append(identifier)
                selected_ids.add(identifier)
                sources.append("similar_history")
                if len(selected) >= min(similar_k, budget):
                    break

    recent_count = min(recent_k, budget - len(selected))
    if recent_count:
        window_start = max(0, event_index - recent_window)

        def recent_key(identifier: str) -> tuple[int, int, str]:
            age, count, _ = coverage_key(identifier)
            return count, age, identifier

        recent_pool = [identifier for identifier in by_id
                       if identifier not in selected_ids
                       and history_stats.get(identifier, {}).get("inserted_event", -1) >= window_start]
        recent = sorted(recent_pool, key=recent_key)[:recent_count]
        selected.extend(recent)
        selected_ids.update(recent)
        sources.extend(["recent_history"] * len(recent))

    coverage_count = (budget - len(selected) + 1) // 2
    coverage = [identifier for identifier in coverage_order if identifier not in selected_ids][:coverage_count]
    selected.extend(coverage)
    selected_ids.update(coverage)
    sources.extend(["coverage_history"] * len(coverage))

    random_count = budget - len(selected)
    if random_count:
        random_pool = sorted(identifier for identifier in by_id if identifier not in selected_ids)
        sampled = rng.sample(random_pool, random_count)
        selected.extend(sampled)
        sources.extend(["random_history"] * len(sampled))
    return [by_id[identifier] for identifier in selected], sources


class StageStratifiedSampler:
    """Randomized round-robin over observed training stages, without replacement.

    Supports the longstream RowView without copying its whole row population.
    All randomness belongs to the caller's checkpointed probe RNG; the stage
    cache is derived solely from immutable memory rows and need not be saved.
    """

    def __init__(self, rng):
        self.rng = rng
        self._memory = None
        self._stages = []

    def sample(self, population, k):
        if not 0 <= k <= len(population):
            raise ValueError('Invalid probe sample size')
        if not k:
            return []
        if hasattr(population, 'indices') and hasattr(population, 'memory'):
            import numpy as np
            memory = population.memory
            if memory is not self._memory or len(memory.rows) < len(self._stages):
                self._memory, self._stages = memory, []
            self._stages.extend(int(row['stage']) for row in memory.rows[len(self._stages):])
            stages = np.asarray(self._stages)[population.indices]
            groups = {int(stage): np.flatnonzero(stages == stage) for stage in np.unique(stages)}
        else:
            groups = defaultdict(list)
            for i, row in enumerate(population):
                groups[int(row['stage'])].append(i)
        stage_order = self.rng.sample(sorted(groups), len(groups))
        allocations = dict.fromkeys(stage_order, 0)
        for _ in range(k):
            stage = stage_order.pop(0)
            allocations[stage] += 1
            if allocations[stage] < len(groups[stage]):
                stage_order.append(stage)
        selected = []
        for stage, count in allocations.items():
            if count:
                offsets = self.rng.sample(range(len(groups[stage])), count)
                selected.extend(population[int(groups[stage][offset])] for offset in offsets)
        return selected


def select_learned_replay(action, available, gate_candidates, budget, rng,
                          policy, decision, similar_k=4, uniform_fraction=0.5):
    """Reserve uniform history, then sample learned MERGE targets safely.

    ADD and IGNORE use uniform historical maintenance. MERGE learns only its
    extra target slots. Missing candidates are filled uniformly. The saved
    policy decision records the exact conditional target sampling population.
    """
    if action not in {'add', 'merge', 'ignore'}:
        raise ValueError('Unsupported action')
    if budget < 0 or similar_k < 0 or not math.isfinite(uniform_fraction) or not 0 <= uniform_fraction <= 1:
        raise ValueError('Invalid replay budget/fraction')
    budget = min(budget, len(available))
    target_slots = min(similar_k, budget - math.ceil(budget * uniform_fraction)) if action == 'merge' else 0
    uniform = rng.sample(available, budget - target_slots)
    selected_ids = {str(row['sample_id']) for row in uniform}
    # Candidates are provided by the runner from safe, non-probe `related`.
    available_ids = {str(row['sample_id']) for row in gate_candidates}
    if len(available_ids) != len(gate_candidates):
        raise ValueError('Duplicate target candidate IDs')
    eligible = [i for i, row in enumerate(gate_candidates) if str(row['sample_id']) not in selected_ids]
    chosen = policy.select_targets(decision, eligible, min(target_slots, len(eligible))) if target_slots else []
    learned = [gate_candidates[i] for i in chosen]
    selected_ids.update(str(row['sample_id']) for row in learned)
    # Rejection sampling avoids scanning/copying large histories on every
    # event. There are at most K blocked IDs; a dense tiny pool uses a list.
    missing = budget - len(uniform) - len(learned)
    fill = []
    if missing and len(available) <= 2 * budget:
        pool = [row for row in available if str(row['sample_id']) not in selected_ids]
        fill = rng.sample(pool, missing)
    else:
        while len(fill) < missing:
            row = available[rng.randrange(len(available))]
            identifier = str(row['sample_id'])
            if identifier not in selected_ids:
                fill.append(row)
                selected_ids.add(identifier)
    rows = uniform + learned + fill
    return rows, (['random_history'] * len(uniform) + ['learned_history'] * len(learned)
                  + ['random_history'] * len(fill))
