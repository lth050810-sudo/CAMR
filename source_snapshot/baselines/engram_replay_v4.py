"""Safe uniform/learned replay and training-stage-stratified reward probes.

Inputs are already safety-filtered by the runner. No evaluation fields or
unseen examples are consulted. Keep the v3 sampler available for ablations.
"""
from __future__ import annotations

import math
from collections import defaultdict


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
    # Rejection sampling avoids scanning/copying 100K rows on every event.
    # There are at most K blocked IDs; a dense tiny pool uses a finite list.
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
    return rows, ['random_history'] * len(uniform) + ['learned_history'] * len(learned) + ['random_history'] * len(fill)
