"""A CPU online writer with a learned relevance gate and actor--critic heads.

The frozen encoder supplies query/candidate vectors; this module never updates
that encoder or the language model.  Query/key projections and the actor learn
from the same policy-gradient loss.  The critic consumes detached routing
features, so a forced ADD cannot train the actor/gate through the value loss.
Optional target draws train the Q/K projections through their own sampled
conditional likelihood, including when the action itself was forced.

Collect a short rollout without changing the policy, then call
``finish_rollout`` once.  Pending transitions contain ordinary input values,
not live autograd graphs, and can be checkpointed in the middle of a rollout.
Safety/semantic-equivalence decisions are the caller's responsibility.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


ACTIONS = ("ignore", "merge", "add")
MODES = ("policy", "rule", "always_add", "random_replay", "bootstrap")


class OnlineWriterPolicy(nn.Module):
    """Jointly train a query/key gate, two sigmoid action gates, and a critic.

    ``allow_ignore`` must already certify an equivalent fact and a correct
    current answer.  Candidate vectors must represent safe replay candidates.
    Empty candidates always disable MERGE, even if the supplied mask allows it.
    ADD is the remaining probability after the two masked Bernoulli gates.

    This is an external replay writer, not an Engram layer inside Qwen.
    """

    def __init__(
        self,
        embedding_dim: int,
        feature_dim: int = 8,
        hidden_dim: int = 32,
        lr: float = 3e-4,
        gamma: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        seed: int = 42,
    ) -> None:
        super().__init__()
        for name, value in (("embedding_dim", embedding_dim), ("feature_dim", feature_dim),
                            ("hidden_dim", hidden_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self._validate_hyperparameters(lr, gamma, entropy_coef, value_coef)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.gamma = float(gamma)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self._policy_version = 0
        self._decision_counter = 0
        self._last_observed_id = -1
        self._pending: list[dict[str, Any]] = []
        # Independent copies detect repeat selection even through copied decisions.
        self._target_selections: dict[int, dict[str, Any]] = {}
        representation_dim = feature_dim + 2 * hidden_dim + 1
        # Deterministic initialization without disturbing the LM's CPU/CUDA RNG.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.generator.get_state())
            self.query_projection = nn.Linear(embedding_dim, hidden_dim, bias=False,
                                              device="cpu", dtype=torch.float32)
            self.key_projection = nn.Linear(embedding_dim, hidden_dim, bias=False,
                                            device="cpu", dtype=torch.float32)
            self.actor = nn.Sequential(
                nn.Linear(representation_dim, hidden_dim, device="cpu", dtype=torch.float32),
                nn.Tanh(),
                nn.Linear(hidden_dim, 2, device="cpu", dtype=torch.float32),
            )
            self.critic = nn.Sequential(
                nn.Linear(representation_dim, hidden_dim, device="cpu", dtype=torch.float32),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1, device="cpu", dtype=torch.float32),
            )
            nn.init.zeros_(self.actor[-1].bias)
            nn.init.zeros_(self.critic[-1].weight)
            nn.init.zeros_(self.critic[-1].bias)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=float(lr))

    @staticmethod
    def _validate_hyperparameters(lr, gamma, entropy_coef, value_coef) -> None:
        values = (lr, gamma, entropy_coef, value_coef)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("policy hyperparameters must be finite")
        if lr <= 0 or not 0 <= gamma <= 1 or entropy_coef < 0 or value_coef < 0:
            raise ValueError("lr must be positive, gamma in [0, 1], and loss coefficients nonnegative")

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @staticmethod
    def _cpu_tensor(value: Any) -> torch.Tensor:
        return torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32).clone()

    def _inputs(self, features, query_vector, candidate_vectors):
        features = self._cpu_tensor(features)
        query = self._cpu_tensor(query_vector)
        candidates = self._cpu_tensor(candidate_vectors)
        if features.shape != (self.feature_dim,) or not torch.isfinite(features).all():
            raise ValueError(f"features must contain {self.feature_dim} finite values")
        if query.shape != (self.embedding_dim,) or not torch.isfinite(query).all():
            raise ValueError(f"query_vector must contain {self.embedding_dim} finite values")
        if candidates.shape == (0,):
            candidates = candidates.reshape(0, self.embedding_dim)
        if (candidates.ndim != 2 or candidates.shape[1] != self.embedding_dim
                or not torch.isfinite(candidates).all()):
            raise ValueError(f"candidate_vectors must have shape [N, {self.embedding_dim}] and be finite")
        if query.norm().item() == 0 or (len(candidates) and (candidates.norm(dim=1) == 0).any()):
            raise ValueError("encoder vectors must be nonzero")
        return features, query, candidates

    @staticmethod
    def _rms_normalize(values: torch.Tensor) -> torch.Tensor:
        return values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + 1e-6)

    def _forward_state(self, state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        if self.query_projection.weight.device.type != "cpu":
            raise RuntimeError("OnlineWriterPolicy must remain on CPU")
        features, query, candidates = self._inputs(
            state["features"], state["query_vector"], state["candidate_vectors"])
        # Normalize frozen embeddings before the learned projections as well.
        query = self._rms_normalize(self.query_projection(F.normalize(query, dim=0)))
        if len(candidates):
            keys = self._rms_normalize(self.key_projection(F.normalize(candidates, dim=1)))
            gate_logits = (keys @ query) / math.sqrt(self.hidden_dim)
            gate_scores = torch.sigmoid(gate_logits)
            # Dividing by sum(gates) would erase the gate for a single candidate.
            pooled = (gate_scores[:, None] * keys).mean(dim=0)
            gate_mean = gate_scores.mean().reshape(1)
        else:
            gate_logits = query.new_empty((0,))
            gate_scores = query.new_empty((0,))
            pooled = query.new_zeros((self.hidden_dim,))
            gate_mean = query.new_zeros((1,))
        representation = torch.cat((features, query, pooled, gate_mean))
        action_gates = torch.sigmoid(self.actor(representation))
        ignore = action_gates[0] if state["allow_ignore"] else action_gates[0] * 0.0
        merge = action_gates[1] if state["allow_merge"] else action_gates[1] * 0.0
        probabilities = torch.stack((ignore, (1 - ignore) * merge, (1 - ignore) * (1 - merge)))
        # No value-loss gradient can reach the actor or shared relevance gate.
        value = self.critic(representation.detach()).squeeze(-1)
        entropy = -(probabilities * probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()).sum()
        if not all(torch.isfinite(value).all() for value in
                   (probabilities, value, entropy, gate_scores, gate_logits)):
            raise ValueError("nonfinite policy output")
        return {"probabilities": probabilities, "value": value,
                "entropy": entropy, "gate_scores": gate_scores, "gate_logits": gate_logits}

    def choose(
        self,
        features: Sequence[float],
        query_vector: Any,
        candidate_vectors: Any,
        allow_ignore: bool,
        allow_merge: bool,
        mode: str = "policy",
    ) -> dict[str, Any]:
        """Sample an action, or inspect the next state with ``mode='bootstrap'``.

        Bootstrap mode never consumes sampling RNG or a decision identifier;
        its ``action``/``log_prob`` are None and it cannot be observed.
        """
        if mode not in MODES:
            raise ValueError(f"unsupported policy mode: {mode}")
        values, query, candidates = self._inputs(features, query_vector, candidate_vectors)
        state = {
            "features": values.tolist(), "query_vector": query.tolist(),
            "candidate_vectors": candidates.tolist(), "allow_ignore": bool(allow_ignore),
            "allow_merge": bool(allow_merge) and len(candidates) > 0, "mode": mode,
            "policy_version": self._policy_version, "decision_id": self._decision_counter,
        }
        output = self._forward_state(state)
        probabilities = output["probabilities"]
        selectable = state["allow_ignore"] or state["allow_merge"]
        if mode == "bootstrap":
            action = None
            log_prob = None
            entropy = output["entropy"]
        elif mode == "policy":
            if selectable:
                # Equivalent to sampling the two conditional Bernoulli gates.
                index = int(torch.multinomial(probabilities.detach(), 1, generator=self.generator).item())
            else:
                index = ACTIONS.index("add")
            action = ACTIONS[index]
            log_prob = (probabilities[index].clamp_min(torch.finfo(torch.float32).tiny).log()
                        if selectable else None)
            entropy = output["entropy"]
        else:
            if mode == "rule":
                action = "ignore" if state["allow_ignore"] else "merge" if state["allow_merge"] else "add"
            elif mode == "random_replay":
                action = "add"  # random history replay is semantic ADD
            else:
                action = "add"
            probabilities = probabilities.new_tensor([float(name == action) for name in ACTIONS])
            log_prob = None
            entropy = probabilities.new_zeros(())
        state["sampled_action"] = action
        if mode != "bootstrap":
            self._decision_counter += 1
        return {
            "action": action, "mode": mode,
            "probabilities": {name: float(probabilities[index].detach()) for index, name in enumerate(ACTIONS)},
            "log_prob": log_prob, "value": output["value"], "entropy": entropy,
            "gate_scores": output["gate_scores"].detach().tolist(), "state": state,
        }

    @staticmethod
    def _validate_target_indices(indices, candidate_count: int, name: str) -> list[int]:
        if not isinstance(indices, (list, tuple)):
            raise ValueError(f"{name} must be a list or tuple of candidate indices")
        values = list(indices)
        if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < candidate_count
               for i in values) or len(values) != len(set(values)):
            raise ValueError(f"{name} must contain distinct valid candidate indices")
        return values

    def select_targets(self, decision: dict[str, Any], eligible_indices: Sequence[int],
                       count: int) -> list[int]:
        """Choose replay targets from caller-certified safe candidates, once.

        Indices refer to this decision's ``candidate_vectors``. Policy mode
        samples a Plackett--Luce ordered subset from the learned Q/K logits;
        ``finish_rollout`` adds its conditional log probability to the actor
        objective, giving the selected targets explicit reward credit. Other
        modes use stable deterministic ranking and are never observed/learned.

        Selecting the entire eligible set (or none) is deterministic and has
        no target gradient: order does not change the training batch. All
        input validation precedes RNG consumption. Call before ``observe``.
        Forced actions may still make a trainable, nontrivial target choice.
        """
        state = self._checked_state(decision)
        if state["mode"] == "bootstrap":
            raise ValueError("bootstrap decisions cannot select targets")
        decision_id = state["decision_id"]
        if (isinstance(decision_id, bool) or not isinstance(decision_id, int)
                or not self._last_observed_id < decision_id < self._decision_counter):
            raise ValueError("targets must be selected for an issued, unobserved decision")
        if decision_id in self._target_selections or "target_selection" in state:
            raise ValueError("targets may be selected only once per decision")
        eligible = self._validate_target_indices(
            eligible_indices, len(state["candidate_vectors"]), "eligible_indices")
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= len(eligible):
            raise ValueError("target count must be an integer between zero and eligible count")
        with torch.no_grad():
            logits = self._forward_state(state)["gate_logits"]
            if count == 0:
                selected = []
            elif count == len(eligible):
                selected = eligible.copy()
            elif state["mode"] != "policy":
                selected = sorted(eligible, key=lambda i: -float(logits[i]))[:count]
            else:
                remaining, selected = eligible.copy(), []
                # Draw on a temporary generator: even a sampling failure cannot
                # consume the policy RNG without committing the target record.
                sampler = torch.Generator(device="cpu")
                sampler.set_state(self.generator.get_state())
                for _ in range(count):
                    probabilities = torch.softmax(logits[remaining], dim=0)
                    position = int(torch.multinomial(probabilities, 1, generator=sampler).item())
                    selected.append(remaining.pop(position))
                self.generator.set_state(sampler.get_state())
        record = {"eligible_indices": eligible, "selected_indices": selected.copy(), "count": count}
        decision["state"]["target_selection"] = copy.deepcopy(record)
        self._target_selections[decision_id] = copy.deepcopy(record)
        return selected

    @staticmethod
    def _target_log_probability(state, output):
        """Reconstruct the sampled ordered-subset likelihood without RNG use."""
        record = state.get("target_selection")
        zero = output["value"].new_zeros(())
        if not record or not 0 < record["count"] < len(record["eligible_indices"]):
            return zero, zero, False
        remaining = record["eligible_indices"].copy()
        log_probability, entropy = zero, zero
        for target in record["selected_indices"]:
            log_probs = F.log_softmax(output["gate_logits"][remaining], dim=0)
            log_probability = log_probability + log_probs[remaining.index(target)]
            entropy = entropy - (log_probs.exp() * log_probs).sum()
            remaining.remove(target)
        return log_probability, entropy, True

    def _checked_state(self, decision: Mapping[str, Any]) -> dict[str, Any]:
        state = copy.deepcopy(decision["state"])
        if state["policy_version"] != self._policy_version:
            raise ValueError("stale decision: choose again after finishing a rollout")
        if state["mode"] not in MODES:
            raise ValueError("invalid decision mode")
        self._inputs(state["features"], state["query_vector"], state["candidate_vectors"])
        if state["allow_merge"] and not state["candidate_vectors"]:
            raise ValueError("MERGE requires candidate vectors")
        action = decision["action"]
        if state["mode"] == "bootstrap":
            if action is not None or state["sampled_action"] is not None:
                raise ValueError("bootstrap states do not have a sampled action")
            return state
        if action not in ACTIONS or action != state["sampled_action"]:
            raise ValueError("decision action differs from the sampled action")
        if action == "ignore" and not state["allow_ignore"]:
            raise ValueError("IGNORE is masked")
        if action == "merge" and not state["allow_merge"]:
            raise ValueError("MERGE is masked")
        record = state.get("target_selection")
        registered = self._target_selections.get(state["decision_id"])
        if record != registered:
            raise ValueError("target selection differs from the recorded draw")
        if record is not None:
            eligible = self._validate_target_indices(
                record["eligible_indices"], len(state["candidate_vectors"]), "eligible_indices")
            selected = self._validate_target_indices(
                record["selected_indices"], len(state["candidate_vectors"]), "selected_indices")
            if (isinstance(record["count"], bool) or not isinstance(record["count"], int)
                    or record["count"] != len(selected) or not set(selected).issubset(eligible)):
                raise ValueError("invalid recorded target selection")
        return state

    def observe(self, decision: Mapping[str, Any], reward: float) -> dict[str, Any]:
        """Record training-only feedback without performing an optimizer step."""
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        state = self._checked_state(decision)
        if state["mode"] == "bootstrap":
            raise ValueError("bootstrap states cannot be observed as transitions")
        if state["mode"] != "policy":
            return {"recorded": False, "pending": self.pending_count}
        decision_id = state["decision_id"]
        if decision_id <= self._last_observed_id or decision_id >= self._decision_counter:
            raise ValueError("decisions must be observed once, in sampling order")
        self._pending.append({"state": state, "action": decision["action"], "reward": reward})
        self._last_observed_id = decision_id
        return {"recorded": True, "pending": self.pending_count}

    def replace_pending_reward(self, decision_id: int, reward: float) -> dict[str, Any]:
        """Replace training-side feedback while its rollout is still pending.

        This supports delayed probe remeasurement before ``finish_rollout``.
        Completed/unknown decisions and nonfinite rewards are rejected without
        mutation. It does not change RNG, targets, order, or policy parameters.
        """
        if isinstance(decision_id, bool) or not isinstance(decision_id, int) or decision_id < 0:
            raise ValueError("decision_id must be a nonnegative integer")
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        matches = [transition for transition in self._pending
                   if transition["state"]["decision_id"] == decision_id]
        if len(matches) != 1:
            raise ValueError("reward replacement requires an observed decision in the pending rollout")
        transition = matches[0]
        previous = transition["reward"]
        transition["reward"] = reward
        return {"decision_id": decision_id, "previous_reward": previous, "reward": reward}

    def finish_rollout(
        self,
        bootstrap_decision: Mapping[str, Any] | None = None,
        terminal: bool = False,
    ) -> dict[str, Any]:
        """Update once with discounted n-step returns, then clear the rollout.

        A nonterminal rollout needs a decision at the next *observed* state to
        bootstrap its value (prefer ``choose(..., mode='bootstrap')``).
        That decision is for bootstrapping only: sample
        the actual next action again after this method changes the policy.
        ``terminal=True`` uses zero continuation value.  No future evaluation
        labels or unseen training examples are required or permitted here.
        """
        if terminal and bootstrap_decision is not None:
            raise ValueError("a terminal rollout cannot also have a bootstrap decision")
        empty = {"updated": False, "actor_updated": False, "critic_updated": False,
                 "gate_updated": False, "transition_count": 0, "actor_transition_count": 0,
                 "selector_updated": False, "target_transition_count": 0,
                 "loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0,
                 "returns": [], "advantages": [], "bootstrap_value": 0.0}
        if not self._pending:
            return empty
        if not terminal and bootstrap_decision is None:
            raise ValueError("nonterminal rollout requires a bootstrap_decision")
        bootstrap_value = 0.0
        if bootstrap_decision is not None:
            state = self._checked_state(bootstrap_decision)
            if state["mode"] not in {"policy", "bootstrap"} or state["decision_id"] <= self._last_observed_id:
                raise ValueError("bootstrap decision must be the next unobserved policy state")
            with torch.no_grad():
                bootstrap_value = float(self._forward_state(state)["value"])
        returns = []
        future = bootstrap_value
        for transition in reversed(self._pending):
            future = transition["reward"] + self.gamma * future
            returns.append(future)
        targets = torch.tensor(list(reversed(returns)), dtype=torch.float32)
        if not torch.isfinite(targets).all():
            raise ValueError("nonfinite discounted returns")
        outputs = [self._forward_state(transition["state"]) for transition in self._pending]
        values = torch.stack([output["value"] for output in outputs])
        advantages = targets - values.detach()
        actor_terms, entropies = [], []
        action_transition_count, target_transition_count = 0, 0
        for index, (transition, output) in enumerate(zip(self._pending, outputs)):
            state = transition["state"]
            log_prob, entropy, target_choice = self._target_log_probability(state, output)
            action_choice = state["allow_ignore"] or state["allow_merge"]
            if action_choice:
                probability = output["probabilities"][ACTIONS.index(transition["action"])]
                log_prob = log_prob + probability.clamp_min(torch.finfo(torch.float32).tiny).log()
                entropy = entropy + output["entropy"]
                action_transition_count += 1
            target_transition_count += int(target_choice)
            if action_choice or target_choice:
                actor_terms.append(-log_prob * advantages[index] - self.entropy_coef * entropy)
                entropies.append(entropy)
        actor_loss = torch.stack(actor_terms).mean() if actor_terms else values.new_zeros(())
        critic_loss = F.mse_loss(values, targets)
        loss = actor_loss
        if self.value_coef > 0:
            loss = loss + self.value_coef * critic_loss
        self.optimizer.zero_grad(set_to_none=True)
        updated = bool(loss.requires_grad)
        if updated:
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0, error_if_nonfinite=True)
            self.optimizer.step()
        result = {
            "updated": updated,
            "actor_updated": updated and any(parameter.grad is not None for parameter in self.actor.parameters()),
            "critic_updated": updated and any(parameter.grad is not None for parameter in self.critic.parameters()),
            "gate_updated": updated and any(parameter.grad is not None for module in
                                             (self.query_projection, self.key_projection) for parameter in module.parameters()),
            "selector_updated": updated and target_transition_count > 0,
            "transition_count": len(self._pending), "actor_transition_count": action_transition_count,
            "target_transition_count": target_transition_count,
            "loss": float(loss.detach()), "actor_loss": float(actor_loss.detach()),
            "critic_loss": float(critic_loss.detach()),
            "entropy": float(torch.stack(entropies).mean().detach()) if entropies else 0.0,
            "returns": targets.tolist(), "advantages": advantages.tolist(),
            "bootstrap_value": bootstrap_value,
        }
        self._pending.clear()
        self._target_selections.clear()
        self._policy_version += 1
        return result

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "format_version": 1, "sampler_state": self.generator.get_state().clone(),
            "policy_version": self._policy_version, "decision_counter": self._decision_counter,
            "last_observed_id": self._last_observed_id, "gamma": self.gamma,
            "entropy_coef": self.entropy_coef, "value_coef": self.value_coef,
            "target_selections": copy.deepcopy(self._target_selections),
        }

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        if state.get("format_version") != 1:
            raise ValueError("unsupported policy state format")
        self._validate_hyperparameters(1.0, state["gamma"], state["entropy_coef"], state["value_coef"])
        self.generator.set_state(state["sampler_state"].cpu())
        self._policy_version = int(state["policy_version"])
        self._decision_counter = int(state["decision_counter"])
        self._last_observed_id = int(state["last_observed_id"])
        self.gamma = float(state["gamma"])
        self.entropy_coef = float(state["entropy_coef"])
        self.value_coef = float(state["value_coef"])
        self._target_selections = copy.deepcopy(state.get("target_selections", {}))

    def training_state_dict(self) -> dict[str, Any]:
        """Return an independent checkpoint, including an unfinished rollout."""
        return copy.deepcopy({
            "format_version": 1,
            "dimensions": {"embedding_dim": self.embedding_dim, "feature_dim": self.feature_dim,
                           "hidden_dim": self.hidden_dim},
            "model": self.state_dict(), "optimizer": self.optimizer.state_dict(),
            "pending": self._pending,
        })

    def load_training_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("format_version") != 1:
            raise ValueError("unsupported training checkpoint format")
        expected = {"embedding_dim": self.embedding_dim, "feature_dim": self.feature_dim,
                    "hidden_dim": self.hidden_dim}
        if state.get("dimensions") != expected:
            raise ValueError("checkpoint policy dimensions do not match")
        self.load_state_dict(copy.deepcopy(state["model"]))
        self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        pending = copy.deepcopy(state["pending"])
        previous_id = -1
        for transition in pending:
            checked = self._checked_state(transition)
            if checked["mode"] != "policy" or not math.isfinite(float(transition["reward"])):
                raise ValueError("invalid pending policy transition")
            decision_id = checked["decision_id"]
            if not previous_id < decision_id <= self._last_observed_id < self._decision_counter:
                raise ValueError("invalid pending transition ordering")
            previous_id = decision_id
        self._pending = pending
