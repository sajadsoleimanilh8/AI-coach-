from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from nexus.personal.baseline import BaselineCalculator
from nexus.personal.dimensions import INVERTED_DIMENSIONS
from nexus.personal.state import PersonalStateEngine
from nexus.personal.trends import Trend, compute_trend
from nexus.personal.weakness import Weakness, WeaknessEngine, sign_corrected_deviation

ForecastMethod = Literal["trend_extrapolation", "insufficient_data"]

_DEFAULT_MAX_HORIZON_DAYS = 30
_DEFAULT_MIN_TREND_CONFIDENCE = 0.5
_DEFAULT_TREND_WINDOW_DAYS = 90.0
_DEFAULT_MIN_SAMPLES = 4

_CONFIDENCE_HALF_LIFE_DAYS = 14.0

_EXTRAPOLATION_CAVEAT = (
    "This is a straight-line extrapolation of your own recorded trend, not a "
    "prediction. It assumes nothing changes and nothing new happens, which is "
    "rarely true. Treat it as 'where the current trend points', not as what "
    "will occur."
)

_INSUFFICIENT_DATA_CAVEAT = (
    "Not enough recorded data to project this dimension. Rather than give you a "
    "number that would be indistinguishable from a guess, there is no projection "
    "here."
)


@dataclass
class DimensionForecast:
    dimension: str
    current_value: float
    projected_value: float | None
    horizon_days: int
    confidence: float
    method: ForecastMethod
    caveat: str
    trend: Trend | None = None


@dataclass
class ForecastResult:
    user_id: str
    horizon_days: int
    forecasts: list[DimensionForecast] = field(default_factory=list)
    caveat: str = _EXTRAPOLATION_CAVEAT


class StateForecaster:
    """Projects each dimension forward along the slope compute_trend()
    already fits. Pure arithmetic — no model call, in line with the rest of
    nexus/personal/.
    """

    def __init__(
        self,
        state_engine: PersonalStateEngine,
        *,
        max_horizon_days: int = _DEFAULT_MAX_HORIZON_DAYS,
        min_trend_confidence: float = _DEFAULT_MIN_TREND_CONFIDENCE,
        trend_window_days: float = _DEFAULT_TREND_WINDOW_DAYS,
        trend_min_samples: int = _DEFAULT_MIN_SAMPLES,
    ) -> None:
        self._state_engine = state_engine
        self._max_horizon_days = max_horizon_days
        self._min_trend_confidence = min_trend_confidence
        self._trend_window_days = trend_window_days
        self._trend_min_samples = trend_min_samples

    async def forecast(self, user_id: str, *, horizon_days: int = 14) -> ForecastResult:
        effective_horizon = max(1, min(horizon_days, self._max_horizon_days))

        state = await self._state_engine.get_state(user_id)
        forecasts: list[DimensionForecast] = []

        for dimension, dim_state in sorted(state.dimensions.items()):
            points = await self._state_engine.get_signal_history(
                user_id, dimension, window_days=self._trend_window_days
            )
            trend = compute_trend(dimension, points, min_samples=self._trend_min_samples)
            forecasts.append(
                self._project(dimension, dim_state.value, trend, effective_horizon)
            )

        return ForecastResult(
            user_id=user_id, horizon_days=effective_horizon, forecasts=forecasts
        )

    def _project(
        self, dimension: str, current: float, trend: Trend | None, horizon_days: int
    ) -> DimensionForecast:
        if trend is None or trend.confidence < self._min_trend_confidence:
            return DimensionForecast(
                dimension=dimension,
                current_value=current,
                projected_value=None,
                horizon_days=horizon_days,
                confidence=0.0,
                method="insufficient_data",
                caveat=_INSUFFICIENT_DATA_CAVEAT,
                trend=trend,
            )

        raw_projection = current + trend.slope * horizon_days
        projected = min(1.0, max(0.0, raw_projection))

        confidence = trend.confidence * (0.5 ** (horizon_days / _CONFIDENCE_HALF_LIFE_DAYS))

        return DimensionForecast(
            dimension=dimension,
            current_value=current,
            projected_value=projected,
            horizon_days=horizon_days,
            confidence=confidence,
            method="trend_extrapolation",
            caveat=_EXTRAPOLATION_CAVEAT,
            trend=trend,
        )


@dataclass
class ClearedWeakness:
    dimension: str
    deviation_before: float
    deviation_after: float
    cleared: bool


@dataclass
class SimulationResult:
    user_id: str
    deltas: dict[str, float]
    weaknesses_before: list[str]
    weaknesses_after: list[str]
    changes: list[ClearedWeakness] = field(default_factory=list)
    caveat: str = (
        "This is a mechanical recalculation: the delta you supplied is applied to the "
        "recorded value and the same weakness thresholds are re-evaluated. It makes NO "
        "claim that the change would cause any other effect, that the change is "
        "achievable, or that the dimensions are causally related. It only answers "
        "'which detected weaknesses would stop being flagged if this number were "
        "different'."
    )


class ScenarioSimulator:
    """Answers "what if sleep_consistency improved by 0.2?" by applying the
    delta and re-running the EXISTING WeaknessEngine thresholds.
    """

    def __init__(
        self,
        state_engine: PersonalStateEngine,
        baseline_calculator: BaselineCalculator,
        weakness_engine: WeaknessEngine,
        *,
        min_deviation: float = 0.08,
    ) -> None:
        self._state_engine = state_engine
        self._baseline_calculator = baseline_calculator
        self._weakness_engine = weakness_engine
        self._min_deviation = min_deviation

    async def simulate(self, user_id: str, deltas: dict[str, float]) -> SimulationResult:
        before: list[Weakness] = await self._weakness_engine.detect(user_id)
        baselines = await self._baseline_calculator.get_baselines(user_id)

        changes: list[ClearedWeakness] = []
        still_flagged: list[str] = []

        for weakness in before:
            delta = deltas.get(weakness.dimension, 0.0)
            baseline = baselines.get(weakness.dimension)
            if baseline is None:
                still_flagged.append(weakness.dimension)
                continue

            adjusted_value = min(1.0, max(0.0, weakness.current + self._directional(weakness.dimension, delta)))
            deviation_after = sign_corrected_deviation(
                weakness.dimension, adjusted_value, baseline.value
            )
            cleared = deviation_after < self._min_deviation

            changes.append(
                ClearedWeakness(
                    dimension=weakness.dimension,
                    deviation_before=weakness.deviation,
                    deviation_after=deviation_after,
                    cleared=cleared,
                )
            )
            if not cleared:
                still_flagged.append(weakness.dimension)

        return SimulationResult(
            user_id=user_id,
            deltas=dict(deltas),
            weaknesses_before=[w.dimension for w in before],
            weaknesses_after=still_flagged,
            changes=changes,
        )

    @staticmethod
    def _directional(dimension: str, delta: float) -> float:
        return -delta if dimension in INVERTED_DIMENSIONS else delta
