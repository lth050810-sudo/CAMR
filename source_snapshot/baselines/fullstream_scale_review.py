"""Exact scalable metadata/LSH safety adapter for the audited online writer.

No embedding search approximation, sample change, or safety relaxation. The
public lists are lazy Sequences: materializing them reproduces the old order.
Rows in FactMemory must remain append-only and immutable, as in the writer.
Cache is derived state, discarded/rebuilt after loading a memory checkpoint.
"""
from __future__ import annotations

import copy
import heapq
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import torch

from baselines.engram_full_safety import (
    _sample_id, _strict_duplicate, strict_mastery_match, strict_normalize_text,
)


class RowView(Sequence):
    def __init__(self, memory, indices):
        self.memory, self.indices = memory, np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return RowView(self.memory, self.indices[item])
        return copy.deepcopy(self.memory.rows[int(self.indices[item])])


class SortedIds(Sequence):
    def __init__(self, memory, indices):
        self.memory, self.indices, self._values = memory, indices, None

    def _get(self):
        if self._values is None:
            self._values = sorted({_sample_id(self.memory.rows[int(i)]) for i in self.indices})
        return self._values

    def __len__(self):
        return len(self._get())

    def __getitem__(self, item):
        return self._get()[item]


class SafetyIndex:
    def __init__(self, memory):
        self.memory = memory
        self.n = 0
        self.capacity = 16
        self.codes = np.empty((self.capacity, memory.tables), dtype=np.uint64)
        self.active = np.zeros(self.capacity, dtype=bool)
        self.conflict_count = np.zeros(self.capacity, dtype=np.int8)
        self.groups = {}
        self.memberships = []
        self.excluded = set()
        self.repeated_ids = set()

    def _reserve(self, needed):
        if needed <= self.capacity:
            return
        capacity = max(needed, self.capacity * 2)
        codes = np.empty((capacity, self.memory.tables), dtype=np.uint64)
        active = np.zeros(capacity, dtype=bool)
        count = np.zeros(capacity, dtype=np.int8)
        codes[:self.n], active[:self.n], count[:self.n] = self.codes[:self.n], self.active[:self.n], self.conflict_count[:self.n]
        self.codes, self.active, self.conflict_count, self.capacity = codes, active, count, capacity

    def keys(self, row):
        slot, fact = self.memory._slot_key(row), self.memory._fact_key(row)
        answer = strict_normalize_text(row.get('ans'))
        keys = []
        if slot is not None and fact is not None:
            keys.append((('metadata_slot', slot), (fact, answer)))
        prompt = strict_normalize_text(row.get('update'))
        if prompt and answer:
            keys.append((('model_question', prompt), answer))
        return keys

    def _activate(self, index):
        if self.active[index]:
            return
        self.active[index] = True
        for key, value in self.memberships[index]:
            group = self.groups.setdefault(key, {})
            was_conflict = len(group) > 1
            group.setdefault(value, set()).add(index)
            if was_conflict:
                self.conflict_count[index] += 1
            elif len(group) > 1:
                for members in group.values():
                    for old in members:
                        self.conflict_count[old] += 1

    def _deactivate(self, index):
        if not self.active[index]:
            return
        self.active[index] = False
        for key, value in self.memberships[index]:
            group = self.groups[key]
            was_conflict = len(group) > 1
            group[value].remove(index)
            if not group[value]:
                del group[value]
            if was_conflict:
                self.conflict_count[index] -= 1
                if len(group) <= 1:
                    for members in group.values():
                        for old in members:
                            self.conflict_count[old] -= 1
            if not group:
                del self.groups[key]

    def sync(self, retired_ids):
        memory = self.memory
        if len(memory.rows) < self.n:
            raise ValueError('SafetyIndex requires append-only memory; rebuild after restore')
        self._reserve(len(memory.rows))
        initial_bulk = self.n == 0 and len(memory.rows) > 1000
        if initial_bulk:
            for table, buckets in enumerate(memory._buckets):
                for key, indices in buckets.items():
                    self.codes[indices, table] = key
        while self.n < len(memory.rows):
            index = self.n
            row = memory.rows[index]
            sid = _sample_id(row)
            if len(memory._sample_index.get(sid, ())) > 1:
                self.repeated_ids.add(sid)
            self.memberships.append(self.keys(row))
            if not initial_bulk:
                self.codes[index] = memory._hashes(memory._vectors[index])
            self.n += 1
            self._activate(index)
        excluded = set(memory._superseded_by)
        for sid in retired_ids:
            excluded.update(memory._sample_index.get(str(sid), ()))
        for index in excluded - self.excluded:
            self._deactivate(index)
        for index in self.excluded - excluded:
            self._activate(index)
        self.excluded = excluded

    def snapshot(self, row, retired_ids):
        self.sync(retired_ids)
        replacement = self.memory._replacement_indices(row)
        temporary = replacement - self.excluded
        for index in temporary:
            self._deactivate(index)
        try:
            active = self.active[:self.n].copy()
            conflicts = (self.conflict_count[:self.n] > 0) & active
            for key, value in self.keys(row):
                group = self.groups.get(key, {})
                if any(other != value for other in group):
                    for members in group.values():
                        if members:
                            conflicts[np.fromiter(members, dtype=np.int64)] = True
            return active, conflicts, replacement
        finally:
            for index in temporary:
                self._activate(index)


def _index(memory):
    cached = getattr(memory, '_fullstream_scale_review_index', None)
    if cached is None:
        cached = SafetyIndex(memory)
        memory._fullstream_scale_review_index = cached
    return cached


def safe_query(memory, row, embedding, retired_ids=()):
    vector = memory._vector(embedding)
    cached = _index(memory)
    active, conflicts, replacement = cached.snapshot(row, retired_ids)
    n = cached.n
    fact_key, slot_key = memory._fact_key(row), memory._slot_key(row)
    exact = {i for i in memory._fact_index.get(fact_key, ()) if active[i] and not conflicts[i]
             and _strict_duplicate(memory, memory.rows[i], row)} if fact_key is not None else set()
    same_slot = {i for i in memory._slot_index.get(slot_key, ()) if active[i]} if slot_key is not None else set()
    exact_mask, slot_mask = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    if exact:
        exact_mask[np.fromiter(exact, dtype=np.int64)] = True
    if same_slot:
        slot_mask[np.fromiter(same_slot, dtype=np.int64)] = True
    ambiguous = conflicts | (slot_mask & ~exact_mask)
    # Each table votes 2 for Hamming distance zero, 1 for distance one.
    # Power-of-two XOR test covers exactly the original one-bit bucket loops.
    votes = np.zeros(n, dtype=np.uint16 if memory.tables < 32768 else np.uint32)
    for table, key in enumerate(memory._hashes(vector)):
        xor = cached.codes[:n, table] ^ np.uint64(key)
        exact_hash = xor == 0
        one_bit = (xor != 0) & ((xor & (xor - np.uint64(1))) == 0)
        votes += exact_hash.astype(votes.dtype) * 2 + one_bit.astype(votes.dtype)
    votes[~active] = 0
    recalled = np.flatnonzero(((votes > 0) | exact_mask | slot_mask) & active)
    order = np.lexsort((recalled, votes[recalled], slot_mask[recalled], exact_mask[recalled]))
    shortlist = recalled[order[-memory.max_candidates:][::-1]]
    candidates = []
    if len(shortlist):
        scores = torch.stack([memory._vectors[int(i)] for i in shortlist]) @ vector
        for raw_index, score in zip(shortlist, scores.tolist()):
            index = int(raw_index)
            candidates.append({'row': copy.deepcopy(memory.rows[index]),
                'similarity': min(1.0, max(-1.0, float(score))), 'equivalent': bool(exact_mask[index]),
                'same_slot': bool(slot_mask[index]), 'memory_index': index,
                'lsh_votes': int(votes[index]), 'ambiguous': bool(ambiguous[index])})
        candidates.sort(key=lambda item: (item['similarity'], item['memory_index']), reverse=True)
    result = {
        'candidates': candidates,
        'similar_rows': [item['row'] for item in candidates if item['similarity'] >= memory.similarity_threshold
                         and not item['equivalent'] and not item['same_slot'] and not item['ambiguous']],
        'duplicate_rows': RowView(memory, sorted(exact, reverse=True)),
        'ambiguous_rows': RowView(memory, np.flatnonzero(ambiguous)[::-1]),
        'replacement_rows': RowView(memory, sorted(replacement, reverse=True)),
        'top_similarity': candidates[0]['similarity'] if candidates else 0.0,
        'candidate_count': len(candidates), 'hash_recalled_count': len(recalled),
        'fact_mode': memory.fact_mode,
        'equivalence_method': 'core_exact_keys_plus_strict_NFC_targets_and_global_prompt_compatibility',
        'conflicting_history_ids': SortedIds(memory, np.flatnonzero(conflicts)),
        'excluded_history_ids': SortedIds(memory, np.flatnonzero(~active)),
        '_scale_state': (n, active, conflicts, ambiguous, exact, copy.deepcopy(row),
                         frozenset(str(sid) for sid in retired_ids), frozenset(memory._superseded_by)),
    }
    return result


def choose_safe_probe_and_replay_pools(memory, query, row, probe_rng, probe_size, retired_ids=()):
    if isinstance(probe_size, bool) or not isinstance(probe_size, int) or probe_size < 0:
        raise ValueError('probe_size must be a nonnegative integer')
    state = query.get('_scale_state')
    if (state is None or state[0] != len(memory.rows) or state[5] != row
            or state[6] != frozenset(str(sid) for sid in retired_ids)
            or state[7] != frozenset(memory._superseded_by)):
        active, conflicts, _ = _index(memory).snapshot(row, retired_ids)
        blocked_ids = {_sample_id(old) for key in ('ambiguous_rows', 'duplicate_rows') for old in query[key]}
    else:
        _, active, conflicts, ambiguous, exact = state[:5]
        # Most IDs are unique. Work only on repeated IDs to propagate blocking
        # across their members; the masks already exclude the original member.
        blocked_mask = ambiguous.copy()
        if exact:
            blocked_mask[np.fromiter(exact, dtype=np.int64)] = True
        active = active & ~blocked_mask
        blocked_ids = {sid for sid in _index(memory).repeated_ids
                       if blocked_mask[memory._sample_index[sid]].any()}
    blocked_ids.add(_sample_id(row))
    eligible_mask = active & ~conflicts
    for sid in blocked_ids:
        members = memory._sample_index.get(sid, ())
        if members:
            eligible_mask[members] = False
    eligible = RowView(memory, np.flatnonzero(eligible_mask))
    probes = probe_rng.sample(eligible, min(probe_size, len(eligible) // 2))
    for old in probes:
        eligible_mask[memory._sample_index[_sample_id(old)]] = False
    available = RowView(memory, np.flatnonzero(eligible_mask))
    related = []
    for old in query['similar_rows']:
        # available_by_id in the original retains the last eligible index.
        last = next((i for i in reversed(memory._sample_index.get(_sample_id(old), ())) if eligible_mask[i]), None)
        if last is not None:
            related.append(copy.deepcopy(memory.rows[last]))
    return probes, available, related


def explicit_retirements(event, memory, retired_ids, aliases=None):
    raw = event['row']
    if ('supersedes_sample_ids' in raw and 'supersedes' in raw
            and raw['supersedes_sample_ids'] != raw['supersedes']):
        raise ValueError('Conflicting explicit replacement fields')
    replacements = raw.get('supersedes_sample_ids', raw.get('supersedes', []))
    if not isinstance(replacements, list) or any(not isinstance(v, str) for v in replacements):
        raise ValueError('supersedes_sample_ids must be a list of observed sample IDs')
    aliases = aliases or {}
    # Slot/fact helpers ignore evaluation-only fields, so raw has the same keys.
    current = raw
    slot = memory._slot_key(current)
    if raw.get('single_valued') is True and raw.get('authoritative_update') is True and slot is not None:
        replacements = list(dict.fromkeys(replacements + [
            str(memory.rows[i]['sample_id']) for i in memory._slot_index.get(slot, ())
            if str(memory.rows[i]['sample_id']) not in retired_ids
            and (memory._fact_key(memory.rows[i]) != memory._fact_key(current)
                 or not strict_mastery_match(memory.rows[i].get('ans'), current.get('ans')))
        ]))
    canonical_ids = []
    for sid in replacements:
        canonical = aliases.get(sid, sid)
        known = memory._sample_index.get(canonical)
        if not known or sid == current['sample_id']:
            raise ValueError(f'Replacement references unknown/self fact: {sid}')
        if memory._slot_key(memory.rows[known[-1]]) != slot:
            raise ValueError('Explicit replacement must refer to the same fact slot')
        canonical_ids.append(canonical)
    retired_now = set(canonical_ids) | set(replacements)
    if canonical_ids:
        canonical_set = set(canonical_ids)
        retired_now.update(alias for alias, canonical in aliases.items() if canonical in canonical_set)
    retired_ids.update(retired_now)
    return sorted(retired_now)


def install(policy_module):
    policy_module.filtered_query = safe_query
    policy_module.choose_probe_and_replay_pools = choose_safe_probe_and_replay_pools
    policy_module.explicit_retirements = explicit_retirements
    if hasattr(policy_module, 'select_replay'):
        policy_module.select_replay = select_replay


def select_replay(action, available, related, k, rng, history_stats, event_index, similar_k=2,
                  recent_k=0, recent_window=64, ignore_replay=False):
    """Exact hybrid replay, without copying/sorting the whole row pool.

    All available statistics are still validated on each invocation, so even
    external modifications cannot bypass the original transactional checks.
    Coverage uses a size-k heap instead of sorting every available row.
    Random choice retains the original lexicographic sample-ID population.
    """
    def nonnegative(value, name):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
        return value

    def identifier_of(row):
        value = row.get('sample_id')
        if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value):
            raise ValueError('Every replay row requires a nonempty sample_id')
        return str(value)

    if action not in {'add', 'merge', 'ignore'}:
        raise ValueError('action must be add, merge or ignore')
    nonnegative(k, 'k')
    nonnegative(similar_k, 'similar_k')
    nonnegative(recent_k, 'recent_k')
    nonnegative(recent_window, 'recent_window')
    if recent_k and not recent_window:
        raise ValueError('recent_window must be positive')
    nonnegative(event_index, 'event_index')
    if (action == 'ignore' and not ignore_replay) or k == 0:
        return [], []
    # FactMemory owns immutable train-only rows. Retain references internally
    # and copy only the chosen rows; ordinary sequences preserve object identity.
    is_view = isinstance(available, RowView)
    rows = (available.memory.rows[int(i)] for i in available.indices) if is_view else iter(available)
    by_id = {}
    for row in rows:
        by_id.setdefault(identifier_of(row), row)
    budget = min(k, len(by_id))
    if budget == 0:
        return [], []

    def coverage_key(identifier):
        stats = history_stats.get(identifier, {})
        inserted, last_replayed = stats.get('inserted_event', -1), stats.get('last_replayed_event')
        count = nonnegative(stats.get('replay_count', 0), 'replay_count')
        for name, value in (('inserted_event', inserted), ('last_replayed_event', last_replayed)):
            if value is None and name == 'last_replayed_event':
                continue
            if isinstance(value, bool) or not isinstance(value, int) or not -1 <= value < event_index:
                raise ValueError(f'{name} for {identifier} must precede event_index')
        return (inserted if last_replayed is None else last_replayed) - event_index, count, identifier

    # Every key is evaluated before RNG mutation, including related rows.
    coverage_order = heapq.nsmallest(budget, by_id, key=coverage_key)
    selected, selected_ids, sources = [], set(), []
    if action == 'merge' and similar_k:
        for row in related:
            identifier = identifier_of(row)
            if identifier in by_id and identifier not in selected_ids:
                selected.append(identifier)
                selected_ids.add(identifier)
                sources.append('similar_history')
                if len(selected) >= min(similar_k, budget):
                    break
    recent_count = min(recent_k, budget - len(selected))
    if recent_count:
        recent_pool = [sid for sid in by_id if sid not in selected_ids
                       and history_stats.get(sid, {}).get('inserted_event', -1) >= max(0, event_index - recent_window)]
        recent = heapq.nsmallest(recent_count, recent_pool,
                                key=lambda sid: (coverage_key(sid)[1], coverage_key(sid)[0], sid))
        selected.extend(recent)
        selected_ids.update(recent)
        sources.extend(['recent_history'] * len(recent))
    coverage_count = (budget - len(selected) + 1) // 2
    coverage = [identifier for identifier in coverage_order if identifier not in selected_ids][:coverage_count]
    selected.extend(coverage)
    selected_ids.update(coverage)
    sources.extend(['coverage_history'] * len(coverage))
    random_count = budget - len(selected)
    if random_count:
        random_pool = sorted(identifier for identifier in by_id if identifier not in selected_ids)
        selected.extend(rng.sample(random_pool, random_count))
        sources.extend(['random_history'] * random_count)
    return ([copy.deepcopy(by_id[sid]) for sid in selected] if is_view else [by_id[sid] for sid in selected]), sources
