from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from nexus.personal.baseline import Baseline, BaselineCalculator
from nexus.personal.dimensions import INVERTED_DIMENSIONS
from nexus.personal.state import DimensionState, PersonalStateEngine
from nexus.personal.trends import Trend, compute_trend

_PRIORITY_ORDER: tuple[Literal["LOW", "MEDIUM", "HIGH"], ...] = ("LOW", "MEDIUM", "HIGH")
_HIGH_SCORE_THRESHOLD = 0.15
_MEDIUM_SCORE_THRESHOLD = 0.08


@dataclass
class Weakness:
    dimension: str
    current: float
    baseline: float
    deviation: float
    priority: Literal["HIGH", "MEDIUM", "LOW"]
    confidence: float
    trend: Trend | None
    explanation: str


def sign_corrected_deviation(dimension: str, current: float, baseline: float) -> float:
    # Positive must ALWAYS mean "worse than baseline" regardless of whether
    # this dimension is higher-is-better or higher-is-worse, so callers can
    # filter/sort on deviation without knowing which kind a dimension is.
    raw_diff = current - baseline
    return raw_diff if dimension in INVERTED_DIMENSIONS else -raw_diff


# Public since ScenarioSimulator has to apply the identical sign convention
# when it recomputes deviations under a hypothetical delta — two different
# implementations of "worse" would silently disagree on inverted dimensions.
_sign_corrected_deviation = sign_corrected_deviation


def _base_priority(score: float) -> Literal["HIGH", "MEDIUM", "LOW"]:
    if score >= _HIGH_SCORE_THRESHOLD:
        return "HIGH"
    if score >= _MEDIUM_SCORE_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def _bump_priority(priority: Literal["HIGH", "MEDIUM", "LOW"]) -> Literal["HIGH", "MEDIUM", "LOW"]:
    index = _PRIORITY_ORDER.index(priority)
    return _PRIORITY_ORDER[min(index + 1, len(_PRIORITY_ORDER) - 1)]


class WeaknessEngine:
    """Compares current PersonalState against BaselineCalculator's history
    to surface dimensions that are meaningfully worse than usual —
    "meaningfully" gated by BOTH min_confidence and min_deviation so a
    single noisy data point never gets reported as an established
    weakness (principle 5)."""

    def __init__(
        self,
        state_engine: PersonalStateEngine,
        baseline_calculator: BaselineCalculator,
        *,
        min_confidence: float = 0.5,
        min_deviation: float = 0.08,
        trend_window_days: float = 90.0,
        trend_min_samples: int = 4,
    ) -> None:
        self._state_engine = state_engine
        self._baseline_calculator = baseline_calculator
        self._min_confidence = min_confidence
        self._min_deviation = min_deviation
        self._trend_window_days = trend_window_days
        self._trend_min_samples = trend_min_samples

    async def detect(self, user_id: str) -> list[Weakness]:
        state = await self._state_engine.get_state(user_id)
        baselines = await self._baseline_calculator.get_baselines(user_id)

        weaknesses: list[Weakness] = []
        for dimension, dim_state in state.dimensions.items():
            baseline = baselines.get(dimension)
            if baseline is None:
                continue
            if dim_state.confidence < self._min_confidence:
                continue

            deviation = _sign_corrected_deviation(dimension, dim_state.value, baseline.value)
            # Only positive (worse-than-baseline) deviations are weaknesses —
            # an improvement of the same magnitude is not a "weakness" just
            # because it's far from baseline, so this is deviation >=
            # min_deviation, not abs(deviation) >= min_deviation.
            if deviation < self._min_deviation:
                continue

            trend = await self._trend_for(user_id, dimension)

            score = deviation * dim_state.confidence
            priority = _base_priority(score)
            if trend is not None and trend.direction == "declining":
                priority = _bump_priority(priority)

            weaknesses.append(
                Weakness(
                    dimension=dimension,
                    current=dim_state.value,
                    baseline=baseline.value,
                    deviation=deviation,
                    priority=priority,
                    confidence=dim_state.confidence,
                    trend=trend,
                    explanation=self._explain(dimension, dim_state, baseline, deviation, trend),
                )
            )

        weaknesses.sort(
            key=lambda w: (
                _PRIORITY_ORDER.index(w.priority),
                abs(w.deviation) * w.confidence,
            ),
            reverse=True,
        )
        return weaknesses

    async def _trend_for(self, user_id: str, dimension: str) -> Trend | None:
        points = await self._state_engine.get_signal_history(
            user_id, dimension, window_days=self._trend_window_days
        )
        return compute_trend(dimension, points, min_samples=self._trend_min_samples)

    @staticmethod
    def _explain(
        dimension: str,
        dim_state: DimensionState,
        baseline: Baseline,
        deviation: float,
        trend: Trend | None,
    ) -> str:
        worse_word = "worse than" if deviation > 0 else "better than"
        trend_note = ""
        if trend is not None and trend.direction != "stable":
            trend_note = f"; trend is {trend.direction} ({trend.slope:+.3f}/day)"
        return (
            f"{dimension} is currently {dim_state.value:.2f} vs. a "
            f"{baseline.window_days:.0f}-day baseline of {baseline.value:.2f} "
            f"({abs(deviation):.2f} {worse_word} baseline), based on "
            f"{dim_state.sample_count} recent sample(s) at {dim_state.confidence:.0%} "
            f"confidence{trend_note}."
        )
