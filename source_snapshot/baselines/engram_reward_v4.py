"""Training-only, checkpointable scaling for current and retention rewards.

Each component gets its own bias-corrected exponentially weighted RMS. A
positive floor caps amplification of near-zero retention noise; warmup blends
from unit scaling, and component clipping bounds individual outliers. The
cost penalty is already in reward units and is never normalized or clipped.
No model state, evaluation data, tensor framework, or global RNG is used.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


class RewardCalibrator:
    """Scale two gains independently using observations up to this call.

    ``score`` updates the RMS with the supplied gain before scoring it, so
    a new outlier cannot be divided only by an obsolete, tiny past scale.
    During ``warmup`` observations, scales interpolate from at least 1.0 to
    the empirical RMS (bounded by their respective floors). Warmup should
    count finalized training-side rewards, not provisional plus revised
    versions of the same event. The runner owns delayed-probe scheduling.
    """

    def __init__(self, current_floor: float = 0.1, retention_floor: float = 0.01,
                 decay: float = 0.99, clip: float = 5.0, warmup: int = 32) -> None:
        config = self._validated_config(current_floor, retention_floor, decay, clip, warmup)
        self.current_floor = config["current_floor"]
        self.retention_floor = config["retention_floor"]
        self.decay = config["decay"]
        self.clip = config["clip"]
        self.warmup = config["warmup"]
        self.count = 0
        self.current_second_moment = 0.0
        self.retention_second_moment = 0.0

    @staticmethod
    def _validated_config(current_floor, retention_floor, decay, clip, warmup):
        values = {"current_floor": float(current_floor), "retention_floor": float(retention_floor),
                  "decay": float(decay), "clip": float(clip)}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("reward-calibration parameters must be finite")
        if values["current_floor"] <= 0 or values["retention_floor"] <= 0 or values["clip"] <= 0:
            raise ValueError("reward floors and clip must be positive")
        if not 0 <= values["decay"] < 1:
            raise ValueError("decay must be in [0, 1)")
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 1:
            raise ValueError("warmup must be a positive integer")
        return {**values, "warmup": warmup}

    def score(self, current_gain: float, retention_gain: float,
              retention_weight: float = 1.0, cost_penalty: float = 0.0) -> dict[str, Any]:
        current_gain, retention_gain = float(current_gain), float(retention_gain)
        retention_weight, cost_penalty = float(retention_weight), float(cost_penalty)
        if not all(math.isfinite(value) for value in
                   (current_gain, retention_gain, retention_weight, cost_penalty)):
            raise ValueError("reward inputs must be finite")
        if retention_weight < 0 or cost_penalty < 0:
            raise ValueError("retention_weight and cost_penalty must be nonnegative")
        current_square, retention_square = current_gain * current_gain, retention_gain * retention_gain
        if not math.isfinite(current_square) or not math.isfinite(retention_square):
            raise ValueError("reward inputs exceed the finite RMS range")
        count = self.count + 1
        current_moment = self.decay * self.current_second_moment + (1 - self.decay) * current_square
        retention_moment = self.decay * self.retention_second_moment + (1 - self.decay) * retention_square
        # expm1 avoids cancellation when decay is close to one.
        correction = -math.expm1(count * math.log(self.decay)) if self.decay else 1.0
        current_rms = math.sqrt(current_moment / correction)
        retention_rms = math.sqrt(retention_moment / correction)
        fraction = min(count / self.warmup, 1.0)

        def scale(rms, floor):
            return max(floor, (1 - fraction) * max(1.0, floor) + fraction * max(rms, floor))

        current_scale, retention_scale = scale(current_rms, self.current_floor), scale(retention_rms, self.retention_floor)
        normalized_current = max(-self.clip, min(self.clip, current_gain / current_scale))
        normalized_retention = max(-self.clip, min(self.clip, retention_gain / retention_scale))
        reward = normalized_current + retention_weight * normalized_retention - cost_penalty
        if not all(math.isfinite(value) for value in
                   (current_moment, retention_moment, current_scale, retention_scale, reward)):
            raise ValueError("nonfinite calibrated reward")
        # Commit only after all validation and arithmetic succeeds.
        self.count = count
        self.current_second_moment, self.retention_second_moment = current_moment, retention_moment
        return {"reward": reward, "current_gain": current_gain, "retention_gain": retention_gain,
                "current_scale": current_scale, "retention_scale": retention_scale,
                "current_rms": current_rms, "retention_rms": retention_rms,
                "normalized_current": normalized_current, "normalized_retention": normalized_retention,
                "retention_weight": retention_weight, "cost_penalty": cost_penalty,
                "calibration_count": count, "warmup_fraction": fraction,
                "current_clipped": abs(current_gain / current_scale) > self.clip,
                "retention_clipped": abs(retention_gain / retention_scale) > self.clip}

    def state_dict(self) -> dict[str, Any]:
        return {"format_version": 1,
                "config": {"current_floor": self.current_floor, "retention_floor": self.retention_floor,
                           "decay": self.decay, "clip": self.clip, "warmup": self.warmup},
                "count": self.count, "current_second_moment": self.current_second_moment,
                "retention_second_moment": self.retention_second_moment}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("format_version") != 1:
            raise ValueError("unsupported reward-calibration checkpoint format")
        config = self._validated_config(**state["config"])
        count = state["count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("invalid calibration observation count")
        current, retention = float(state["current_second_moment"]), float(state["retention_second_moment"])
        if not all(math.isfinite(value) and value >= 0 for value in (current, retention)):
            raise ValueError("invalid calibration second moment")
        if count == 0 and (current != 0 or retention != 0):
            raise ValueError("an unobserved calibrator must have zero second moments")
        for key, value in config.items():
            setattr(self, key, value)
        self.count, self.current_second_moment, self.retention_second_moment = count, current, retention
