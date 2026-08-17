from __future__ import annotations

import pytest

from nexus.generation.planner import BriefBuilder
from nexus.personal.state import DimensionState, PersonalState
from nexus.personal.weakness import Weakness


class _FakeStateEngine:
    def __init__(self, dimensions: dict[str, DimensionState]) -> None:
        self._state = PersonalState(user_id="u1", dimensions=dimensions, computed_at=0.0)

    async def get_state(self, user_id: str) -> PersonalState:
        return self._state


class _FakeWeaknessEngine:
    def __init__(self, weaknesses: list[Weakness] | None = None) -> None:
        self._weaknesses = weaknesses or []

    async def detect(self, user_id: str) -> list[Weakness]:
        return self._weaknesses


class _FakeHealthAnalyzer:
    class _Analysis:
        patterns: list = []

    async def analyze(self, user_id: str):
        return self._Analysis()


class _FakeProfileStore:
    def __init__(self, profile: dict | None = None) -> None:
        self._profile = profile or {}

    async def get_profile(self, user_id: str) -> dict:
        return self._profile


def _weakness(dimension: str, deviation: float) -> Weakness:
    return Weakness(
        dimension=dimension,
        current=0.5,
        baseline=0.5 + deviation,
        deviation=deviation,
        priority="MEDIUM",
        confidence=0.8,
        trend=None,
        explanation=f"{dimension} is off baseline.",
    )


def _builder(dimensions, weaknesses) -> BriefBuilder:
    return BriefBuilder(
        _FakeStateEngine(dimensions),
        _FakeWeaknessEngine(weaknesses),
        _FakeHealthAnalyzer(),
        _FakeProfileStore(),
    )


@pytest.mark.asyncio
async def test_zero_signals_uses_neutral_default_and_says_so() -> None:
    builder = _builder({}, [])
    brief = await builder.build(user_id="u1", request_type="workout", constraints={})

    assert not brief.state.dimensions
    assert brief.target_intensity == pytest.approx(0.7)
    assert len(brief.rationale) == 1
    assert "generic" in brief.rationale[0].lower()
    assert "no personal signals" in brief.rationale[0].lower()


@pytest.mark.asyncio
async def test_intensity_arithmetic_matches_hand_computed_value() -> None:
    dimensions = {"physical.energy": DimensionState("physical.energy", 0.6, 0.9, 5, 1.0)}
    weaknesses = [
        _weakness("physical.recovery", 0.2),
        _weakness("mental.fatigue", 0.3),
        _weakness("mental.stress", 0.1),
        _weakness("lifestyle.sleep_quality", 0.15),
    ]
    builder = _builder(dimensions, weaknesses)

    brief = await builder.build(user_id="u1", request_type="workout", constraints={})

    assert brief.target_intensity == pytest.approx(0.475, abs=1e-9)
    assert len(brief.rationale) == 5
    assert "0.70" in brief.rationale[0]


@pytest.mark.asyncio
async def test_rationale_cites_each_contributing_dimension() -> None:
    dimensions = {"physical.energy": DimensionState("physical.energy", 0.6, 0.9, 5, 1.0)}
    weaknesses = [_weakness("physical.recovery", 0.2), _weakness("mental.fatigue", 0.3)]
    builder = _builder(dimensions, weaknesses)

    brief = await builder.build(user_id="u1", request_type="recovery", constraints={})

    joined = " ".join(brief.rationale)
    assert "physical.recovery" in joined
    assert "mental.fatigue" in joined
    assert "low recovery" in joined
    assert "high fatigue" in joined


@pytest.mark.asyncio
async def test_intensity_clamps_to_minimum_and_records_the_clamp() -> None:
    dimensions = {"physical.energy": DimensionState("physical.energy", 0.6, 0.9, 5, 1.0)}
    weaknesses = [
        _weakness("physical.recovery", 1.0),
        _weakness("mental.fatigue", 1.0),
        _weakness("mental.stress", 1.0),
        _weakness("lifestyle.sleep_quality", 1.0),
    ]
    builder = _builder(dimensions, weaknesses)

    brief = await builder.build(user_id="u1", request_type="workout", constraints={})

    assert brief.target_intensity == pytest.approx(0.2, abs=1e-9)
    assert any("clamped" in line.lower() for line in brief.rationale)


@pytest.mark.asyncio
async def test_intensity_never_exceeds_default_when_no_negative_factors_present() -> None:
    dimensions = {"physical.energy": DimensionState("physical.energy", 0.9, 0.9, 5, 1.0)}
    builder = _builder(dimensions, [])

    brief = await builder.build(user_id="u1", request_type="workout", constraints={})

    assert brief.target_intensity == pytest.approx(0.7, abs=1e-9)
    assert len(brief.rationale) == 1


@pytest.mark.asyncio
async def test_constraints_and_request_type_are_carried_through_unchanged() -> None:
    builder = _builder({}, [])
    constraints = {"time_minutes": 30, "equipment": "none"}

    brief = await builder.build(user_id="u1", request_type="nutrition", constraints=constraints)

    assert brief.request_type == "nutrition"
    assert brief.constraints == constraints
    assert brief.user_id == "u1"
