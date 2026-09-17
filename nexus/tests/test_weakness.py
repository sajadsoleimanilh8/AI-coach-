from __future__ import annotations

import pytest

from nexus.personal.baseline import Baseline
from nexus.personal.state import DimensionState, PersonalState
from nexus.personal.weakness import WeaknessEngine


class _FakeStateEngine:
    def __init__(self, dimensions: dict[str, DimensionState], history=None) -> None:
        self._state = PersonalState(user_id="u1", dimensions=dimensions, computed_at=0.0)
        self._history = history or {}

    async def get_state(self, user_id: str) -> PersonalState:
        return self._state

    async def get_signal_history(self, user_id: str, dimension: str, *, window_days: float):
        return self._history.get(dimension, [])


class _FakeBaselineCalculator:
    def __init__(self, baselines: dict[str, Baseline]) -> None:
        self._baselines = baselines

    async def get_baselines(self, user_id: str) -> dict[str, Baseline]:
        return self._baselines


def _dim(dimension: str, value: float, confidence: float = 0.9, sample_count: int = 5) -> DimensionState:
    return DimensionState(
        dimension=dimension, value=value, confidence=confidence, sample_count=sample_count,
        latest_at=100.0,
    )


def _baseline(dimension: str, value: float) -> Baseline:
    return Baseline(dimension=dimension, value=value, sample_count=10, window_days=90.0)


@pytest.mark.asyncio
async def test_normal_dimension_dropping_below_baseline_is_a_positive_deviation() -> None:
    # physical.energy dropped from a 0.7 baseline to 0.4 — worse.
    state_engine = _FakeStateEngine({"physical.energy": _dim("physical.energy", 0.4)})
    baselines = _FakeBaselineCalculator({"physical.energy": _baseline("physical.energy", 0.7)})
    engine = WeaknessEngine(state_engine, baselines, min_confidence=0.5, min_deviation=0.05)

    weaknesses = await engine.detect("u1")

    assert len(weaknesses) == 1
    w = weaknesses[0]
    assert w.dimension == "physical.energy"
    assert w.deviation == pytest.approx(0.3, abs=1e-6)
    assert w.current == 0.4
    assert w.baseline == 0.7


@pytest.mark.asyncio
async def test_inverted_dimension_rising_above_baseline_is_a_positive_deviation() -> None:
    # mental.stress rose from a 0.4 baseline to 0.7 — worse (higher stress is bad).
    state_engine = _FakeStateEngine({"mental.stress": _dim("mental.stress", 0.7)})
    baselines = _FakeBaselineCalculator({"mental.stress": _baseline("mental.stress", 0.4)})
    engine = WeaknessEngine(state_engine, baselines, min_confidence=0.5, min_deviation=0.05)

    weaknesses = await engine.detect("u1")

    assert len(weaknesses) == 1
    assert weaknesses[0].deviation == pytest.approx(0.3, abs=1e-6)


@pytest.mark.asyncio
async def test_normal_dimension_improving_above_baseline_is_not_a_weakness() -> None:
    # physical.energy went UP relative to baseline — an improvement, not a weakness.
    state_engine = _FakeStateEngine({"physical.energy": _dim("physical.energy", 0.9)})
    baselines = _FakeBaselineCalculator({"physical.energy": _baseline("physical.energy", 0.5)})
    engine = WeaknessEngine(state_engine, baselines, min_confidence=0.5, min_deviation=0.05)

    weaknesses = await engine.detect("u1")

    assert weaknesses == []


@pytest.mark.asyncio
async def test_min_confidence_filters_out_low_confidence_state() -> None:
    state_engine = _FakeStateEngine(
        {"physical.energy": _dim("physical.energy", 0.3, confidence=0.2)}
    )
    baselines = _FakeBaselineCalculator({"physical.energy": _baseline("physical.energy", 0.7)})
    engine = WeaknessEngine(state_engine, baselines, min_confidence=0.5, min_deviation=0.05)

    assert await engine.detect("u1") == []


@pytest.mark.asyncio
async def test_min_deviation_filters_out_small_deviations() -> None:
    state_engine = _FakeStateEngine({"physical.energy": _dim("physical.energy", 0.68)})
    baselines = _FakeBaselineCalculator({"physical.energy": _baseline("physical.energy", 0.7)})
    engine = WeaknessEngine(state_engine, baselines, min_confidence=0.5, min_deviation=0.08)

    assert await engine.detect("u1") == []


@pytest.mark.asyncio
async def test_dimension_without_a_baseline_is_skipped() -> None:
    state_engine = _FakeStateEngine({"physical.energy": _dim("physical.energy", 0.1)})
    baselines = _FakeBaselineCalculator({})  # no baseline computed yet
    engine = WeaknessEngine(state_engine, baselines, min_confidence=0.5, min_deviation=0.05)

    assert await engine.detect("u1") == []


@pytest.mark.asyncio
async def test_priority_ordering_highest_deviation_score_first() -> None:
    state_engine = _FakeStateEngine(
        {
            "physical.energy": _dim("physical.energy", 0.3, confidence=0.9),  # dev 0.4 -> HIGH
            "mental.mood": _dim("mental.mood", 0.55, confidence=0.9),  # dev 0.1 -> MEDIUM
        }
    )
    baselines = _FakeBaselineCalculator(
        {
            "physical.energy": _baseline("physical.energy", 0.7),
            "mental.mood": _baseline("mental.mood", 0.65),
        }
    )
    engine = WeaknessEngine(state_engine, baselines, min_confidence=0.5, min_deviation=0.05)

    weaknesses = await engine.detect("u1")

    assert [w.dimension for w in weaknesses] == ["physical.energy", "mental.mood"]
    assert weaknesses[0].priority == "HIGH"
    assert weaknesses[1].priority == "MEDIUM"


@pytest.mark.asyncio
async def test_declining_trend_bumps_priority_up_one_level() -> None:
    now = 1_000_000.0
    day = 86400.0
    # A small, MEDIUM-tier deviation on its own...
    state_engine = _FakeStateEngine(
        {"mental.mood": _dim("mental.mood", 0.55, confidence=0.9)},
        history={
            "mental.mood": [
                (now - 3 * day, 0.70),
                (now - 2 * day, 0.65),
                (now - 1 * day, 0.60),
                (now, 0.55),
            ]
        },
    )
    baselines = _FakeBaselineCalculator({"mental.mood": _baseline("mental.mood", 0.65)})
    engine = WeaknessEngine(
        state_engine, baselines, min_confidence=0.5, min_deviation=0.05, trend_min_samples=4
    )

    weaknesses = await engine.detect("u1")

    assert len(weaknesses) == 1
    w = weaknesses[0]
    assert w.trend is not None
    assert w.trend.direction == "declining"
    # base score = 0.1 * 0.9 = 0.09 -> MEDIUM on its own, bumped to HIGH.
    assert w.priority == "HIGH"


@pytest.mark.asyncio
async def test_explanation_mentions_dimension_and_confidence() -> None:
    state_engine = _FakeStateEngine({"physical.energy": _dim("physical.energy", 0.3)})
    baselines = _FakeBaselineCalculator({"physical.energy": _baseline("physical.energy", 0.7)})
    engine = WeaknessEngine(state_engine, baselines, min_confidence=0.5, min_deviation=0.05)

    weaknesses = await engine.detect("u1")

    assert "physical.energy" in weaknesses[0].explanation
    assert "0.7" in weaknesses[0].explanation or "0.70" in weaknesses[0].explanation
