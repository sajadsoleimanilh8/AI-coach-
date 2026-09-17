from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

from nexus.personal.baseline import Baseline, BaselineCalculator
from nexus.personal.dimensions import DIMENSION_GROUPS, INVERTED_DIMENSIONS
from nexus.personal.state import DimensionState, PersonalState, PersonalStateEngine
from nexus.personal.trends import Trend, compute_trend
from nexus.personal.weakness import Weakness, WeaknessEngine

# A raw difference below this is noise, not a pattern — same spirit as
# WeaknessEngine's min_deviation, applied locally since these rules
# compare state-vs-baseline directly rather than going through
# WeaknessEngine (a pattern here can involve a dimension that hasn't
# crossed WeaknessEngine's own threshold yet, e.g. a rising-but-not-yet-
# "weak" stress trend).
_MARGIN = 0.05
_HIGH_FATIGUE_THRESHOLD = 0.6
_TREND_WINDOW_DAYS = 90.0
_TREND_MIN_SAMPLES = 4
_TREND_DIMENSIONS = (
    "lifestyle.sleep_consistency",
    "mental.stress",
)

_SIGNIFICANT_SCORE = 0.25
_NOTABLE_SCORE = 0.12


@dataclass
class HealthPattern:
    name: str
    dimensions_involved: list[str]
    severity: Literal["watch", "notable", "significant"]
    confidence: float
    sample_size: int
    explanation: str


@dataclass
class HealthAnalysis:
    user_id: str
    patterns: list[HealthPattern]
    scorecard: dict[str, float]
    data_sufficiency: Literal["none", "sparse", "adequate"]
    computed_at: float


@dataclass
class _RuleContext:
    state: PersonalState
    baselines: dict[str, Baseline]
    weaknesses_by_dim: dict[str, Weakness]
    trends_by_dim: dict[str, Trend | None]
    min_sample_size: int


def _severity_for(magnitude: float, confidence: float) -> Literal["watch", "notable", "significant"]:
    score = magnitude * confidence
    if score >= _SIGNIFICANT_SCORE:
        return "significant"
    if score >= _NOTABLE_SCORE:
        return "notable"
    return "watch"


def _dim_and_baseline(
    ctx: _RuleContext, dimension: str
) -> tuple[DimensionState, Baseline] | None:
    dim_state = ctx.state.dimensions.get(dimension)
    baseline = ctx.baselines.get(dimension)
    if dim_state is None or baseline is None:
        return None
    return dim_state, baseline


def _rule_high_load_low_recovery(ctx: _RuleContext) -> HealthPattern | None:
    activity = _dim_and_baseline(ctx, "physical.activity")
    recovery = _dim_and_baseline(ctx, "physical.recovery")
    if activity is None or recovery is None:
        return None
    activity_state, activity_baseline = activity
    recovery_state, recovery_baseline = recovery

    activity_above = activity_state.value > activity_baseline.value + _MARGIN
    recovery_below = recovery_state.value < recovery_baseline.value - _MARGIN
    if not (activity_above and recovery_below):
        return None

    confidence = min(activity_state.confidence, recovery_state.confidence)
    magnitude = (activity_state.value - activity_baseline.value) + (
        recovery_baseline.value - recovery_state.value
    )
    return HealthPattern(
        name="high_load_low_recovery",
        dimensions_involved=["physical.activity", "physical.recovery"],
        severity=_severity_for(magnitude, confidence),
        confidence=confidence,
        sample_size=min(activity_state.sample_count, recovery_state.sample_count),
        explanation=(
            f"Activity is elevated ({activity_state.value:.2f} vs. baseline "
            f"{activity_baseline.value:.2f}) while recovery is depressed "
            f"({recovery_state.value:.2f} vs. baseline {recovery_baseline.value:.2f}) — "
            f"a load/recovery imbalance."
        ),
    )


def _rule_sleep_inconsistency(ctx: _RuleContext) -> HealthPattern | None:
    dimension = "lifestyle.sleep_consistency"
    resolved = _dim_and_baseline(ctx, dimension)
    if resolved is None:
        return None
    dim_state, baseline = resolved
    trend = ctx.trends_by_dim.get(dimension)

    below = dim_state.value < baseline.value - _MARGIN
    declining = trend is not None and trend.direction == "declining"
    if not (below and declining):
        return None

    return HealthPattern(
        name="sleep_inconsistency",
        dimensions_involved=[dimension],
        severity=_severity_for(baseline.value - dim_state.value, dim_state.confidence),
        confidence=dim_state.confidence,
        sample_size=dim_state.sample_count,
        explanation=(
            f"Sleep consistency is below baseline ({dim_state.value:.2f} vs. "
            f"{baseline.value:.2f}) and trending {trend.direction} "
            f"({trend.slope:+.3f}/day)."
        ),
    )


def _rule_stress_accumulation(ctx: _RuleContext) -> HealthPattern | None:
    dimension = "mental.stress"
    trend = ctx.trends_by_dim.get(dimension)
    if trend is None:
        return None
    # mental.stress is INVERTED (nexus/personal/dimensions.py) — trends.py
    # already sign-corrects for that, so direction="declining" here means
    # stress is RISING (getting worse), not literally declining in value.
    if trend.direction != "declining":
        return None

    # A week's worth of drift, expressed on the same 0..1 scale a raw
    # deviation would be — makes this comparable to the other rules'
    # magnitude without needing a second severity scale.
    weekly_drift = abs(trend.slope) * 7
    return HealthPattern(
        name="stress_accumulation",
        dimensions_involved=[dimension],
        severity=_severity_for(weekly_drift, trend.confidence),
        confidence=trend.confidence,
        sample_size=trend.sample_count,
        explanation=(
            f"Stress has been rising over the recent window "
            f"({trend.slope:+.3f}/day), based on {trend.sample_count} sample(s)."
        ),
    )


def _rule_fatigue_without_load(ctx: _RuleContext) -> HealthPattern | None:
    fatigue_state = ctx.state.dimensions.get("mental.fatigue")
    activity = _dim_and_baseline(ctx, "physical.activity")
    if fatigue_state is None or activity is None:
        return None
    activity_state, activity_baseline = activity

    fatigue_high = fatigue_state.value >= _HIGH_FATIGUE_THRESHOLD
    activity_at_or_below = activity_state.value <= activity_baseline.value + _MARGIN
    if not (fatigue_high and activity_at_or_below):
        return None

    confidence = min(fatigue_state.confidence, activity_state.confidence)
    return HealthPattern(
        name="fatigue_without_load",
        dimensions_involved=["mental.fatigue", "physical.activity"],
        severity=_severity_for(fatigue_state.value, confidence),
        confidence=confidence,
        sample_size=min(fatigue_state.sample_count, activity_state.sample_count),
        explanation=(
            f"Fatigue is elevated ({fatigue_state.value:.2f}) despite activity being "
            f"at or below baseline ({activity_state.value:.2f} vs. "
            f"{activity_baseline.value:.2f}) — worth attention precisely because it "
            f"is NOT explained by training load."
        ),
    )


def _rule_recovery_debt(ctx: _RuleContext) -> HealthPattern | None:
    dimension = "physical.recovery"
    resolved = _dim_and_baseline(ctx, dimension)
    if resolved is None:
        return None
    dim_state, baseline = resolved

    # "Sustained" — a single low reading is noise, not debt.
    if dim_state.sample_count < ctx.min_sample_size:
        return None
    if not (dim_state.value < baseline.value - _MARGIN):
        return None

    return HealthPattern(
        name="recovery_debt",
        dimensions_involved=[dimension],
        severity=_severity_for(baseline.value - dim_state.value, dim_state.confidence),
        confidence=dim_state.confidence,
        sample_size=dim_state.sample_count,
        explanation=(
            f"Recovery has been below baseline ({dim_state.value:.2f} vs. "
            f"{baseline.value:.2f}) across {dim_state.sample_count} recent "
            f"sample(s), not just a single low reading."
        ),
    )


def _rule_cognitive_dip(ctx: _RuleContext) -> HealthPattern | None:
    sleep = _dim_and_baseline(ctx, "lifestyle.sleep_quality")
    if sleep is None:
        return None
    sleep_state, sleep_baseline = sleep
    if not (sleep_state.value < sleep_baseline.value - _MARGIN):
        return None

    hits: list[tuple[str, DimensionState, Baseline]] = []
    for dimension in ("mental.focus", "cognitive.working_memory"):
        resolved = _dim_and_baseline(ctx, dimension)
        if resolved is None:
            continue
        dim_state, baseline = resolved
        if dim_state.value < baseline.value - _MARGIN:
            hits.append((dimension, dim_state, baseline))
    if not hits:
        return None

    confidence = min([sleep_state.confidence] + [h[1].confidence for h in hits])
    sample_size = min([sleep_state.sample_count] + [h[1].sample_count for h in hits])
    magnitude = (sleep_baseline.value - sleep_state.value) + sum(
        h[2].value - h[1].value for h in hits
    )
    details = "; ".join(f"{d}={s.value:.2f} (baseline {b.value:.2f})" for d, s, b in hits)
    return HealthPattern(
        name="cognitive_dip",
        dimensions_involved=["lifestyle.sleep_quality"] + [h[0] for h in hits],
        severity=_severity_for(magnitude, confidence),
        confidence=confidence,
        sample_size=sample_size,
        explanation=(
            f"Sleep quality is below baseline ({sleep_state.value:.2f} vs. "
            f"{sleep_baseline.value:.2f}) alongside below-baseline cognitive "
            f"readings: {details}."
        ),
    )


# Module-level so new rules can be added without touching HealthAnalyzer.
RULES: list[Callable[[_RuleContext], HealthPattern | None]] = [
    _rule_high_load_low_recovery,
    _rule_sleep_inconsistency,
    _rule_stress_accumulation,
    _rule_fatigue_without_load,
    _rule_recovery_debt,
    _rule_cognitive_dip,
]


class HealthAnalyzer:
    """Reads PersonalStateEngine / BaselineCalculator / WeaknessEngine
    output and applies deterministic RULES to surface multi-dimension
    patterns. No LLM anywhere in this class (principle 1) — HealthAgent is
    the only place a model touches this data, and only to narrate it."""

    def __init__(
        self,
        state_engine: PersonalStateEngine,
        baseline_calculator: BaselineCalculator,
        weakness_engine: WeaknessEngine,
        *,
        min_sample_size: int = 3,
    ) -> None:
        self._state_engine = state_engine
        self._baseline_calculator = baseline_calculator
        self._weakness_engine = weakness_engine
        self._min_sample_size = min_sample_size

    async def analyze(self, user_id: str) -> HealthAnalysis:
        state = await self._state_engine.get_state(user_id)
        baselines = await self._baseline_calculator.get_baselines(user_id)
        weaknesses = await self._weakness_engine.detect(user_id)
        weaknesses_by_dim = {w.dimension: w for w in weaknesses}

        trends_by_dim: dict[str, Trend | None] = {}
        for dimension in _TREND_DIMENSIONS:
            if dimension not in state.dimensions:
                continue
            points = await self._state_engine.get_signal_history(
                user_id, dimension, window_days=_TREND_WINDOW_DAYS
            )
            trends_by_dim[dimension] = compute_trend(
                dimension, points, min_samples=_TREND_MIN_SAMPLES
            )

        ctx = _RuleContext(
            state=state,
            baselines=baselines,
            weaknesses_by_dim=weaknesses_by_dim,
            trends_by_dim=trends_by_dim,
            min_sample_size=self._min_sample_size,
        )
        patterns = [pattern for rule in RULES if (pattern := rule(ctx)) is not None]

        data_sufficiency = self._data_sufficiency(state)
        if data_sufficiency != "adequate":
            # Thin data must never support a "significant" claim (principle
            # 3/4) — cap in place rather than drop the pattern, since "watch
            # this, but the data is thin" is still useful signal.
            patterns = [replace(p, severity="watch") for p in patterns]

        return HealthAnalysis(
            user_id=user_id,
            patterns=patterns,
            scorecard=self._scorecard(state),
            data_sufficiency=data_sufficiency,
            computed_at=time.time(),
        )

    def _data_sufficiency(self, state: PersonalState) -> Literal["none", "sparse", "adequate"]:
        if not state.dimensions:
            return "none"
        best_sample_count = max(d.sample_count for d in state.dimensions.values())
        if best_sample_count < self._min_sample_size:
            return "sparse"
        return "adequate"

    @staticmethod
    def _scorecard(state: PersonalState) -> dict[str, float]:
        scorecard: dict[str, float] = {}
        for group_name in DIMENSION_GROUPS:
            group_dims = state.group(group_name)
            if not group_dims:
                continue
            # Confidence-weighted, and sign-corrected for inverted
            # dimensions so "higher scorecard value" always means "better"
            # across every group, not just the non-inverted ones.
            total_weight = sum(d.confidence for d in group_dims.values())
            if total_weight == 0:
                continue
            weighted_sum = sum(
                (1.0 - d.value if dim in INVERTED_DIMENSIONS else d.value) * d.confidence
                for dim, d in group_dims.items()
            )
            scorecard[group_name] = weighted_sum / total_weight
        return scorecard
