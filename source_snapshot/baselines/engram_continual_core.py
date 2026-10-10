"""Online hash recall and an explicit stochastic add/merge/ignore policy.

This is an Engram-*inspired* continual-learning experiment, not an implementation
of the paper's token N-gram memory.  Hashes only generate candidates: dense
cosine reranks the recalled vectors.  There is deliberately no full-memory
dense-search fallback.

``structured`` fact matching is an oracle diagnostic using dataset triples.
``text`` matching certifies only normalized exact question/answer duplicates;
it does not pretend to implement general semantic fact equivalence.
"""

from __future__ import annotations

import copy
import json
import math
import unicodedata
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


TRAIN_FIELDS = (
    "sample_id", "stage", "update", "ans", "subject", "subject_id",
    "relation", "relation_id", "object", "object_id", "tag",
)
OPTIONAL_TRAIN_FIELDS = (
    "supersedes", "scope", "single_valued", "authoritative_update",
)
ACTIONS = ("ignore", "merge", "add")


def normalized_text(value: Any) -> str:
    """Normalize typography, case, and whitespace, without semantic inference."""
    if value is None:
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def training_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the train-only whitelist; evaluation probes never enter memory."""
    return {key: copy.deepcopy(row[key]) for key in (*TRAIN_FIELDS, *OPTIONAL_TRAIN_FIELDS) if key in row}


class FactMemory:
    """Append-only, online multi-table random-hyperplane LSH fact memory.

    Query observes only previously inserted rows.  Exact fact and slot indices
    supplement LSH recall, but every returned similarity is computed from the
    recalled vectors.  Candidate selection uses exact-match priority followed
    by table votes and recency; it never scans all vectors to fill a shortlist.

    Structured duplicates also require the same normalized target answer:
    different aliases for one object ID are conservatively not certified.
    A same-subject/relation different-object record is ambiguous: the relation
    might be multi-valued.  We retain the record, report it, and conservatively
    exclude it from replay rather than claiming that either answer is obsolete.
    """

    def __init__(
        self,
        dim: int,
        seed: int = 42,
        tables: int = 8,
        bits: int = 8,
        max_candidates: int = 64,
        similarity_threshold: float = 0.65,
        fact_mode: str = "structured",
    ) -> None:
        if dim < 1 or tables < 1 or not 1 <= bits <= 63 or max_candidates < 1:
            raise ValueError("dim/tables/max_candidates must be positive; bits must be 1..63")
        if not -1 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be in [-1, 1]")
        if fact_mode not in {"structured", "text"}:
            raise ValueError("fact_mode must be 'structured' or 'text'")
        self.dim = dim
        self.seed = seed
        self.tables = tables
        self.bits = bits
        self.max_candidates = max_candidates
        self.similarity_threshold = similarity_threshold
        self.fact_mode = fact_mode
        self.equivalence_method = (
            "oracle_structured_triples_and_exact_answer_with_exact_qa_fallback"
            if fact_mode == "structured" else "normalized_exact_qa_only"
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.hyperplanes = torch.randn(tables, bits, dim, generator=generator)
        self.rows: list[dict[str, Any]] = []
        self._vectors: list[torch.Tensor] = []
        self._buckets: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(tables)]
        self._fact_index: dict[tuple[str, ...], list[int]] = defaultdict(list)
        self._slot_index: dict[tuple[str, ...], list[int]] = defaultdict(list)
        self._sample_index: dict[str, list[int]] = defaultdict(list)
        self._superseded_by: dict[int, int] = {}

    @property
    def active_rows(self) -> list[dict[str, Any]]:
        """Return train-only copies of records still eligible for future replay."""
        return [copy.deepcopy(row) for index, row in enumerate(self.rows) if index not in self._superseded_by]

    @staticmethod
    def _scope(row: Mapping[str, Any]) -> str:
        """Exact scope matching is conservative; no time/range inference occurs."""
        return json.dumps(row.get("scope"), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    def _replacement_indices(self, row: Mapping[str, Any]) -> set[int]:
        self._scope(row)  # Validate even when a row has no usable fact key.
        explicit = row.get("supersedes", [])
        if not isinstance(explicit, (list, tuple)) or any(not isinstance(value, (str, int)) for value in explicit):
            raise ValueError("supersedes must be a list of previous sample IDs")
        identifiers = {str(value) for value in explicit}
        if "sample_id" in row and str(row["sample_id"]) in identifiers:
            raise ValueError("an incoming sample cannot supersede its own sample ID")
        unknown = identifiers - self._sample_index.keys()
        if unknown:
            raise ValueError(f"supersedes references unknown or future sample IDs: {sorted(unknown)}")
        replacement = {index for identifier in identifiers for index in self._sample_index[identifier]
                       if index not in self._superseded_by}
        # True must be explicit booleans; strings such as "false" are not proof.
        if row.get("single_valued") is True and row.get("authoritative_update") is True:
            slot, scope = self._slot_key(row), self._scope(row)
            if slot is not None:
                replacement.update(
                    index for index in self._slot_index.get(slot, ())
                    if index not in self._superseded_by and self._scope(self.rows[index]) == scope
                    and self._fact_key(self.rows[index]) != self._fact_key(row)
                )
        return replacement

    def candidate_embeddings(self, result: Mapping[str, Any]) -> torch.Tensor:
        """Get frozen vectors in the same order as a query's logged candidates."""
        indices = [int(item["memory_index"]) for item in result["candidates"]]
        if any(index < 0 or index >= len(self._vectors) for index in indices):
            raise ValueError("candidate index does not belong to this memory")
        return torch.stack([self._vectors[index] for index in indices]) if indices else torch.empty(0, self.dim)

    def state_dict(self) -> dict[str, Any]:
        """Capture memory without held-out fields or mutable tensor aliases."""
        return {
            "version": 1,
            "config": {name: getattr(self, name) for name in (
                "dim", "seed", "tables", "bits", "max_candidates", "similarity_threshold", "fact_mode",
            )},
            "rows": [training_row(row) for row in self.rows],
            "vectors": torch.stack(self._vectors).clone() if self._vectors else torch.empty(0, self.dim),
            "hyperplanes": self.hyperplanes.clone(),
            "superseded_by": dict(self._superseded_by),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "FactMemory":
        if state.get("version") != 1:
            raise ValueError("unsupported fact memory checkpoint version")
        memory = cls(**dict(state["config"]))
        planes = torch.as_tensor(state["hyperplanes"]).detach().cpu().float().clone()
        vectors = torch.as_tensor(state["vectors"]).detach().cpu().float()
        if planes.shape != memory.hyperplanes.shape or not torch.isfinite(planes).all():
            raise ValueError("invalid checkpoint hyperplanes")
        if vectors.shape != (len(state["rows"]), memory.dim):
            raise ValueError("checkpoint rows and vectors do not align")
        memory.hyperplanes = planes
        for row, vector in zip(state["rows"], vectors):
            # Preserve exact stored floats (and hence LSH buckets) on resume.
            if not torch.isfinite(vector).all() or not torch.isclose(vector.norm(), torch.tensor(1.0), atol=1e-5):
                raise ValueError("checkpoint vectors must be finite and normalized")
            saved = training_row(row)
            memory._replacement_indices(saved)  # Validate references in stream order.
            index = len(memory.rows)
            memory.rows.append(saved)
            memory._vectors.append(vector.clone())
            for table, key in zip(memory._buckets, memory._hashes(vector)):
                table[key].append(index)
            fact_key, slot_key = memory._fact_key(saved), memory._slot_key(saved)
            if fact_key is not None:
                memory._fact_index[fact_key].append(index)
            if slot_key is not None:
                memory._slot_index[slot_key].append(index)
            if "sample_id" in saved:
                memory._sample_index[str(saved["sample_id"])].append(index)
        retired = {int(old): int(new) for old, new in state["superseded_by"].items()}
        if any(not 0 <= old < new < len(memory.rows) for old, new in retired.items()):
            raise ValueError("invalid supersession history in checkpoint")
        memory._superseded_by = retired
        return memory

    def _vector(self, embedding: Any) -> torch.Tensor:
        vector = torch.as_tensor(embedding).detach().to(device="cpu", dtype=torch.float32)
        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector[0]
        if vector.ndim != 1 or vector.numel() != self.dim:
            raise ValueError(f"expected one embedding of dimension {self.dim}")
        if not torch.isfinite(vector).all() or vector.norm().item() <= 0:
            raise ValueError("embedding must be finite and nonzero")
        return F.normalize(vector, dim=0).clone()

    def _hashes(self, vector: torch.Tensor) -> list[int]:
        signs = torch.einsum("tbd,d->tb", self.hyperplanes, vector) >= 0
        return [sum(int(bit) << position for position, bit in enumerate(table)) for table in signs]

    @staticmethod
    def _entity(row: Mapping[str, Any], name: str) -> str:
        identifier = normalized_text(row.get(f"{name}_id"))
        return f"id:{identifier}" if identifier else f"text:{normalized_text(row.get(name))}"

    def _structured_slot(self, row: Mapping[str, Any]) -> tuple[str, ...] | None:
        subject, relation = self._entity(row, "subject"), self._entity(row, "relation")
        if subject == "text:" or relation == "text:":
            return None
        return ("structured", subject, relation, self._scope(row))

    def _object(self, row: Mapping[str, Any]) -> str:
        identifier = normalized_text(row.get("object_id"))
        if identifier:
            return f"id:{identifier}"
        # ans is the target actually used for training; prefer it to a stale
        # optional object label when an identifier is unavailable.
        value = normalized_text(row.get("ans")) or normalized_text(row.get("object"))
        return f"text:{value}"

    def _slot_key(self, row: Mapping[str, Any]) -> tuple[str, ...] | None:
        if self.fact_mode == "structured":
            structured = self._structured_slot(row)
            if structured is not None:
                return structured
        question = normalized_text(row.get("update"))
        return ("exact_question", question, self._scope(row)) if question else None

    def _fact_key(self, row: Mapping[str, Any]) -> tuple[str, ...] | None:
        if self.fact_mode == "structured":
            slot = self._structured_slot(row)
            obj = self._object(row)
            if slot is not None and obj != "text:":
                # A reused/stale object ID must not hide a different training
                # answer.  This conservatively treats aliases with different
                # text as ambiguous, rather than certifying their equivalence.
                answer = normalized_text(row.get("ans")) or normalized_text(row.get("object"))
                return (*slot, obj, answer)
        question, answer = normalized_text(row.get("update")), normalized_text(row.get("ans"))
        return ("exact_qa", question, answer, self._scope(row)) if question and answer else None

    def insert(self, row: Mapping[str, Any], embedding: Any) -> int:
        """Insert one observed training fact, returning its memory index."""
        vector = self._vector(embedding)
        saved = training_row(row)
        # Validate metadata before committing any rows or indices.
        fact_key, slot_key = self._fact_key(saved), self._slot_key(saved)
        replacement = self._replacement_indices(saved)
        index = len(self.rows)
        self.rows.append(saved)
        self._vectors.append(vector)
        for table, key in zip(self._buckets, self._hashes(vector)):
            table[key].append(index)
        if fact_key is not None:
            self._fact_index[fact_key].append(index)
        if slot_key is not None:
            self._slot_index[slot_key].append(index)
        if "sample_id" in saved:
            self._sample_index[str(saved["sample_id"])].append(index)
        self._superseded_by.update({old: index for old in replacement})
        return index

    def query(self, row: Mapping[str, Any], embedding: Any) -> dict[str, Any]:
        """Recall then rerank historical records without inserting the query."""
        vector = self._vector(embedding)
        fact_key, slot_key = self._fact_key(row), self._slot_key(row)
        replacement = self._replacement_indices(row)
        excluded = set(self._superseded_by) | replacement
        exact = (set(self._fact_index.get(fact_key, ())) if fact_key is not None else set()) - excluded
        same_slot = (set(self._slot_index.get(slot_key, ())) if slot_key is not None else set()) - excluded
        votes: dict[int, int] = defaultdict(int)
        for table, key in zip(self._buckets, self._hashes(vector)):
            # Hamming distance 0 receives two votes; distance 1 receives one.
            # Only bucket members are inspected, never the full vector store.
            for index in table.get(key, ()):
                if index not in excluded:
                    votes[index] += 2
            for bit in range(self.bits):
                for index in table.get(key ^ (1 << bit), ()):
                    if index not in excluded:
                        votes[index] += 1
        recalled = set(votes) | exact | same_slot
        shortlist = sorted(
            recalled,
            key=lambda index: (index in exact, index in same_slot, votes.get(index, 0), index),
            reverse=True,
        )[:self.max_candidates]
        candidates: list[dict[str, Any]] = []
        if shortlist:
            # Cosine is applied only after hash/exact-key recall has completed.
            scores = torch.stack([self._vectors[index] for index in shortlist]) @ vector
            for index, score in zip(shortlist, scores.tolist()):
                candidates.append({
                    "row": copy.deepcopy(self.rows[index]),
                    "similarity": min(1.0, max(-1.0, float(score))),
                    "equivalent": index in exact,
                    "same_slot": index in same_slot,
                    "memory_index": index,
                    "lsh_votes": votes.get(index, 0),
                })
            candidates.sort(key=lambda item: (item["similarity"], item["memory_index"]), reverse=True)
        return {
            "candidates": candidates,
            "similar_rows": [
                item["row"] for item in candidates
                if item["similarity"] >= self.similarity_threshold
                and not item["equivalent"] and not item["same_slot"]
            ],
            # Exact-key safety decisions must not be truncated by the dense
            # reranking budget.  In particular a random-replay caller needs
            # every ambiguous old target, even when there are >64 of them.
            "duplicate_rows": [copy.deepcopy(self.rows[index]) for index in sorted(exact, reverse=True)],
            "ambiguous_rows": [
                copy.deepcopy(self.rows[index]) for index in sorted(same_slot - exact, reverse=True)
            ],
            "replacement_rows": [copy.deepcopy(self.rows[index]) for index in sorted(replacement, reverse=True)],
            "top_similarity": candidates[0]["similarity"] if candidates else 0.0,
            "candidate_count": len(candidates),
            "hash_recalled_count": len(recalled),
            "fact_mode": self.fact_mode,
            "equivalence_method": self.equivalence_method,
        }


class ActionPolicy(nn.Module):
    """Two Bernoulli gates define a sampled ignore/merge/add distribution.

    First sample ignore.  Conditional on not ignoring, sample merge; add is
    the remainder.  Eligibility masks disable unsafe ignore/merge decisions.
    REINFORCE uses caller-provided *training-only* rewards and an EMA baseline;
    this module does not receive, compute, or inspect held-out evaluation data.
    """

    def __init__(self, seed: int = 42, lr: float = 0.01) -> None:
        super().__init__()
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be positive and finite")
        self.weight = nn.Parameter(torch.zeros(2, 4))
        self.bias = nn.Parameter(torch.full((2,), math.log(4.0)))
        self.register_buffer("reward_baseline", torch.tensor(0.0))
        self.register_buffer("reward_count", torch.tensor(0, dtype=torch.long))
        self.baseline_decay = 0.9
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def get_extra_state(self) -> dict[str, Any]:
        return {"generator_state": self.generator.get_state(), "baseline_decay": self.baseline_decay}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.generator.set_state(state["generator_state"].cpu())
        self.baseline_decay = float(state["baseline_decay"])

    def choose(
        self,
        features: Sequence[float],
        allow_ignore: bool,
        allow_merge: bool,
        mode: str = "policy",
    ) -> dict[str, Any]:
        values = torch.as_tensor(features, dtype=self.weight.dtype, device=self.weight.device)
        if values.shape != (4,) or not torch.isfinite(values).all():
            raise ValueError("features must contain four finite values")
        if mode not in {"policy", "rule", "always_add", "random_replay"}:
            raise ValueError(f"unsupported policy mode: {mode}")
        if mode != "policy":
            if mode == "rule":
                action = "ignore" if allow_ignore else "merge" if allow_merge else "add"
            elif mode == "random_replay":
                action = "add"  # random history replay is semantic ADD
            else:
                action = "add"
            return {
                "action": action,
                "probabilities": {name: float(name == action) for name in ACTIONS},
                "log_prob": None,
                "mode": mode,
            }
        gates = torch.sigmoid(F.linear(values, self.weight, self.bias))
        ignore = gates[0] if allow_ignore else gates[0] * 0.0
        merge = gates[1] if allow_merge else gates[1] * 0.0
        probabilities = {"ignore": ignore, "merge": (1 - ignore) * merge, "add": (1 - ignore) * (1 - merge)}
        # Separate Bernoulli samples implement the two conditional decisions.
        if allow_ignore and bool(torch.bernoulli(ignore.detach().cpu(), generator=self.generator).item()):
            action = "ignore"
        elif allow_merge and bool(torch.bernoulli(merge.detach().cpu(), generator=self.generator).item()):
            action = "merge"
        else:
            action = "add"
        return {
            "action": action,
            "probabilities": {name: float(value.detach().cpu()) for name, value in probabilities.items()},
            # Forced add has no policy decision to credit.  Skipping its
            # optimizer step also avoids updates from stale Adam momentum.
            "log_prob": (
                probabilities[action].clamp_min(torch.finfo(values.dtype).tiny).log()
                if allow_ignore or allow_merge else None
            ),
            "mode": mode,
        }

    def learn(self, decision: Mapping[str, Any], reward: float) -> dict[str, Any]:
        """Take one REINFORCE step using the pre-update baseline.

        The caller must supply only training-loss improvement, rehearsal
        retention, and/or compute costs.  Rewards from test/rephrase/locality
        labels would invalidate the experiment, even though they are numeric.
        """
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        baseline = float(self.reward_baseline)
        advantage = reward - baseline
        log_prob = decision.get("log_prob")
        updated = isinstance(log_prob, torch.Tensor) and log_prob.requires_grad
        loss_value = 0.0
        if updated:
            loss = -log_prob * advantage
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
            self.optimizer.step()
            loss_value = float(loss.detach().cpu())
        with torch.no_grad():
            self.reward_baseline.mul_(self.baseline_decay).add_(reward * (1 - self.baseline_decay))
            self.reward_count.add_(1)
        return {
            "loss": loss_value,
            "reward": reward,
            "baseline": baseline,
            "advantage": advantage,
            "updated": updated,
        }


class ContextAwarePolicy(ActionPolicy):
    """Jointly learned relevance gate and masked softmax action policy.

    This first external-writer variant uses a frozen sentence encoder's current
    vector as context, rather than a Qwen hidden state. Trainable query/key
    projections and RMS normalization produce one sigmoid per recalled item.
    Their weighted value *sum* feeds action logits without detaching the gate.
    This is not a native token-level Engram layer and never modifies retrieval.

    The caller certifies fact equivalence/mastery and compatible replay using
    ``allow_ignore``/``allow_merge``. The mask preserves ADD as a safe learning
    option. It does not turn semantic categories into compulsory action labels.
    ``learn`` shares the legacy policy's online REINFORCE/EMA credit assignment;
    its single optimizer updates the gate and action network together.
    """

    def __init__(
        self,
        embedding_dim: int,
        feature_dim: int = 8,
        gate_dim: int = 64,
        hidden_dim: int = 64,
        seed: int = 42,
        lr: float = 0.001,
    ) -> None:
        # Reuse ActionPolicy's credit-assignment methods, not its old two-gate
        # architecture or parameters. Legacy ActionPolicy remains unchanged.
        nn.Module.__init__(self)
        dimensions = (embedding_dim, feature_dim, gate_dim, hidden_dim)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in dimensions):
            raise ValueError("embedding/feature/gate/hidden dimensions must be positive integers")
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be positive and finite")
        self.embedding_dim, self.feature_dim, self.gate_dim, self.hidden_dim = dimensions
        self.seed = seed
        self.norm_eps = 1e-6
        # Constructor determinism must not consume the language model's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.query_projection = nn.Linear(embedding_dim, gate_dim, bias=False)
            self.key_projection = nn.Linear(embedding_dim, gate_dim, bias=False)
            self.value_projection = nn.Linear(embedding_dim, gate_dim, bias=False)
            self.action_network = nn.Sequential(
                nn.Linear(2 * gate_dim + feature_dim + 3, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, len(ACTIONS)),
            )
        self.register_buffer("reward_baseline", torch.tensor(0.0))
        self.register_buffer("reward_count", torch.tensor(0, dtype=torch.long))
        self.baseline_decay = 0.9
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def _rms_norm(self, value: torch.Tensor) -> torch.Tensor:
        # Explicit implementation also supports installed PyTorch versions
        # without nn.RMSNorm. No candidate-wise softmax replaces the sigmoid.
        return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.norm_eps)

    def forward(
        self,
        query_embedding: Any,
        candidate_embeddings: Any,
        features: Sequence[float],
        allow_ignore: bool,
        allow_merge: bool,
    ) -> dict[str, torch.Tensor]:
        parameter = self.query_projection.weight

        def frozen_input(value: Any) -> torch.Tensor:
            # Frozen encoders may return inference tensors. A detached view
            # keeps that flag and cannot be saved for a trainable projection's
            # backward; cloning here creates an ordinary constant tensor.
            return torch.as_tensor(value, device=parameter.device, dtype=parameter.dtype).detach().clone()

        query = frozen_input(query_embedding)
        candidates = frozen_input(candidate_embeddings)
        values = frozen_input(features)
        if query.shape == (1, self.embedding_dim):
            query = query[0]
        if candidates.shape == (0,):
            candidates = candidates.reshape(0, self.embedding_dim)
        if query.shape != (self.embedding_dim,):
            raise ValueError(f"query must be one vector of dimension {self.embedding_dim}")
        if candidates.ndim != 2 or candidates.shape[1] != self.embedding_dim:
            raise ValueError(f"candidates must have shape [N, {self.embedding_dim}]")
        if values.shape != (self.feature_dim,):
            raise ValueError(f"features must contain {self.feature_dim} values")
        if not all(bool(torch.isfinite(tensor).all()) for tensor in (query, candidates, values)):
            raise ValueError("query, candidates, and features must be finite")
        query_key = self._rms_norm(self.query_projection(query))
        keys = self._rms_norm(self.key_projection(candidates))
        gates = torch.sigmoid((keys @ query_key) / math.sqrt(self.gate_dim))
        projected_values = self.value_projection(candidates)
        history = (gates.unsqueeze(-1) * projected_values).sum(dim=0)
        candidate_count = candidates.shape[0]
        if candidate_count:
            gate_summary = torch.stack((gates.mean(), gates.max(), gates.new_tensor(math.log1p(candidate_count))))
        else:
            gate_summary = query_key.new_zeros(3)
        state = torch.cat((query_key, history, gate_summary, values))
        logits = self.action_network(state)
        mask = torch.tensor(
            [bool(allow_ignore and candidate_count), bool(allow_merge and candidate_count), True],
            dtype=torch.bool, device=parameter.device,
        )
        log_probabilities = F.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
        return {
            "probabilities": log_probabilities.exp(),
            "log_probabilities": log_probabilities,
            "logits": logits,
            "action_mask": mask,
            "gate_values": gates,
            "history": history,
        }

    def choose(
        self,
        query_embedding: Any,
        candidate_embeddings: Any,
        features: Sequence[float],
        allow_ignore: bool,
        allow_merge: bool,
        mode: str = "policy",
    ) -> dict[str, Any]:
        if mode not in {"policy", "rule", "always_add", "random_replay"}:
            raise ValueError(f"unsupported policy mode: {mode}")
        output = self(query_embedding, candidate_embeddings, features, allow_ignore, allow_merge)
        log_prob = None
        if mode == "policy":
            probabilities = output["probabilities"]
            selected = int(torch.multinomial(probabilities.detach().cpu(), 1, generator=self.generator))
            if int(output["action_mask"].sum()) > 1:
                log_prob = output["log_probabilities"][selected]
            action = ACTIONS[selected]
            logged_probabilities = {name: float(value) for name, value in zip(ACTIONS, probabilities.detach().cpu())}
        else:
            if mode == "rule":
                mask = output["action_mask"]
                action = "ignore" if bool(mask[0]) else "merge" if bool(mask[1]) else "add"
            elif mode == "random_replay":
                # Random replay may exist even when LSH returns no candidates;
                # the caller certifies compatibility of its own replay pool.
                action = "add"  # random history replay is semantic ADD
            else:
                action = "add"
            logged_probabilities = {name: float(name == action) for name in ACTIONS}
        return {
            "action": action,
            "probabilities": logged_probabilities,
            "log_prob": log_prob,
            "mode": mode,
            "gate_values": output["gate_values"].detach().cpu().tolist(),
            "history_norm": float(output["history"].detach().norm().cpu()),
        }

    def checkpoint_state(self) -> dict[str, Any]:
        """Complete policy resume state, including Adam moments and sampling."""
        return copy.deepcopy({
            "version": 1,
            "config": {name: getattr(self, name) for name in (
                "embedding_dim", "feature_dim", "gate_dim", "hidden_dim",
            )},
            "model": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        })

    def load_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        expected = {name: getattr(self, name) for name in (
            "embedding_dim", "feature_dim", "gate_dim", "hidden_dim",
        )}
        if state.get("version") != 1 or state.get("config") != expected:
            raise ValueError("policy checkpoint version or architecture does not match")
        self.load_state_dict(state["model"])
        self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
