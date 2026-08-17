from __future__ import annotations

import pytest

from nexus.personal.baseline import Baseline
from nexus.personal.forecast import ScenarioSimulator
from nexus.personal.state import DimensionState, PersonalState
from nexus.personal.weakness import WeaknessEngine


class _FakeStateEngine:
    def __init__(self, dimensions: dict[str, DimensionState]) -> None:
        self._state = PersonalState(user_id="u1", dimensions=dimensions, computed_at=0.0)

    async def get_state(self, user_id: str) -> PersonalState:
        return self._state

    async def get_signal_history(self, user_id: str, dimension: str, *, window_days: float):
        return []


class _FakeBaselineCalculator:
    def __init__(self, baselines: dict[str, Baseline]) -> None:
        self._baselines = baselines

    async def get_baselines(self, user_id: str) -> dict[str, Baseline]:
        return self._baselines


def _dim(dimension: str, value: float) -> DimensionState:
    return DimensionState(
        dimension=dimension, value=value, confidence=0.9, sample_count=5, latest_at=100.0
    )


def _baseline(dimension: str, value: float) -> Baseline:
    return Baseline(dimension=dimension, value=value, sample_count=10, window_days=90.0)


def _simulator(dimensions: dict[str, float], baselines: dict[str, float], *, min_deviation: float = 0.08):
    state_engine = _FakeStateEngine({d: _dim(d, v) for d, v in dimensions.items()})
    baseline_calc = _FakeBaselineCalculator({d: _baseline(d, v) for d, v in baselines.items()})
    weakness_engine = WeaknessEngine(
        state_engine, baseline_calc, min_confidence=0.5, min_deviation=min_deviation
    )
    return ScenarioSimulator(
        state_engine, baseline_calc, weakness_engine, min_deviation=min_deviation
    )


@pytest.mark.asyncio
async def test_a_sufficient_delta_clears_the_weakness() -> None:
    simulator = _simulator({"physical.energy": 0.4}, {"physical.energy": 0.7})

    result = await simulator.simulate("u1", {"physical.energy": 0.3})

    assert result.weaknesses_before == ["physical.energy"]
    assert result.weaknesses_after == []
    assert result.changes[0].cleared is True
    assert result.changes[0].deviation_after == pytest.approx(0.0, abs=1e-6)


@pytest.mark.asyncio
async def test_an_insufficient_delta_leaves_the_weakness_flagged() -> None:
    simulator = _simulator({"physical.energy": 0.4}, {"physical.energy": 0.7})

    result = await simulator.simulate("u1", {"physical.energy": 0.05})

    assert result.weaknesses_after == ["physical.energy"]
    assert result.changes[0].cleared is False
    assert result.changes[0].deviation_after < result.changes[0].deviation_before


@pytest.mark.asyncio
async def test_a_delta_on_an_unrelated_dimension_changes_nothing() -> None:
    simulator = _simulator({"physical.energy": 0.4}, {"physical.energy": 0.7})

    result = await simulator.simulate("u1", {"lifestyle.sleep_quality": 0.5})

    assert result.weaknesses_after == ["physical.energy"]
    assert result.changes[0].deviation_after == pytest.approx(result.changes[0].deviation_before)


@pytest.mark.asyncio
async def test_an_inverted_dimension_improves_when_its_raw_value_falls() -> None:
    """"Improve stress by 0.3" must LOWER the raw stress reading — the same
    sign convention WeaknessEngine uses, or the two would disagree."""
    simulator = _simulator({"mental.stress": 0.7}, {"mental.stress": 0.4})

    result = await simulator.simulate("u1", {"mental.stress": 0.3})

    assert result.weaknesses_before == ["mental.stress"]
    assert result.weaknesses_after == []
    assert result.changes[0].cleared is True


@pytest.mark.asyncio
async def test_a_wrong_signed_delta_on_an_inverted_dimension_makes_it_worse() -> None:
    simulator = _simulator({"mental.stress": 0.7}, {"mental.stress": 0.4})

    result = await simulator.simulate("u1", {"mental.stress": -0.2})

    assert result.changes[0].cleared is False
    assert result.changes[0].deviation_after > result.changes[0].deviation_before


@pytest.mark.asyncio
async def test_multiple_weaknesses_are_reported_independently() -> None:
    simulator = _simulator(
        {"physical.energy": 0.4, "mental.mood": 0.4},
        {"physical.energy": 0.7, "mental.mood": 0.7},
    )

    result = await simulator.simulate("u1", {"physical.energy": 0.3})

    assert set(result.weaknesses_before) == {"physical.energy", "mental.mood"}
    assert result.weaknesses_after == ["mental.mood"]
    cleared = {c.dimension: c.cleared for c in result.changes}
    assert cleared == {"physical.energy": True, "mental.mood": False}


@pytest.mark.asyncio
async def test_no_weaknesses_means_nothing_to_simulate() -> None:
    simulator = _simulator({"physical.energy": 0.9}, {"physical.energy": 0.7})

    result = await simulator.simulate("u1", {"physical.energy": 0.1})

    assert result.weaknesses_before == []
    assert result.changes == []


@pytest.mark.asyncio
async def test_the_delta_is_clamped_to_the_scale() -> None:
    simulator = _simulator({"physical.energy": 0.4}, {"physical.energy": 0.7})

    result = await simulator.simulate("u1", {"physical.energy": 5.0})

    assert result.changes[0].deviation_after == pytest.approx(-0.3, abs=1e-6)


@pytest.mark.asyncio
async def test_the_result_makes_no_causal_claim_and_says_so() -> None:
    simulator = _simulator({"physical.energy": 0.4}, {"physical.energy": 0.7})

    result = await simulator.simulate("u1", {"physical.energy": 0.3})

    assert "mechanical recalculation" in result.caveat
    assert "NO" in result.caveat and "claim" in result.caveat
    assert "causally related" in result.caveat


@pytest.mark.asyncio
async def test_the_deltas_are_echoed_back(tmp_path) -> None:
    simulator = _simulator({"physical.energy": 0.4}, {"physical.energy": 0.7})

    result = await simulator.simulate("u1", {"physical.energy": 0.3})

    assert result.deltas == {"physical.energy": 0.3}
    assert result.user_id == "u1"
