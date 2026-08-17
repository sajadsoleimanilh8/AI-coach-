from __future__ import annotations

import pytest

from nexus.health.analyzer import HealthAnalyzer
from nexus.personal.baseline import Baseline
from nexus.personal.state import DimensionState, PersonalState

_DAY = 86400.0


class _FakeStateEngine:
    def __init__(self, dimensions: dict[str, DimensionState], history=None) -> None:
        self._state = PersonalState(user_id="u1", dimensions=dimensions, computed_at=1_000_000.0)
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


class _FakeWeaknessEngine:
    def __init__(self, weaknesses=None) -> None:
        self._weaknesses = weaknesses or []

    async def detect(self, user_id: str):
        return self._weaknesses


def _dim(dimension: str, value: float, *, confidence: float = 0.9, sample_count: int = 5) -> DimensionState:
    return DimensionState(
        dimension=dimension, value=value, confidence=confidence, sample_count=sample_count,
        latest_at=1_000_000.0,
    )


def _baseline(dimension: str, value: float) -> Baseline:
    return Baseline(dimension=dimension, value=value, sample_count=10, window_days=90.0)


def _declining_points(now: float, start: float, step: float) -> list[tuple[float, float]]:
    return [(now - (3 - i) * _DAY, start + i * step) for i in range(4)]


def _analyzer(dimensions, baselines, history=None, weaknesses=None, min_sample_size=3) -> HealthAnalyzer:
    return HealthAnalyzer(
        _FakeStateEngine(dimensions, history),
        _FakeBaselineCalculator(baselines),
        _FakeWeaknessEngine(weaknesses),
        min_sample_size=min_sample_size,
    )


@pytest.mark.asyncio
async def test_high_load_low_recovery_fires() -> None:
    analyzer = _analyzer(
        {"physical.activity": _dim("physical.activity", 0.8), "physical.recovery": _dim("physical.recovery", 0.3)},
        {"physical.activity": _baseline("physical.activity", 0.5), "physical.recovery": _baseline("physical.recovery", 0.6)},
    )
    analysis = await analyzer.analyze("u1")
    assert any(p.name == "high_load_low_recovery" for p in analysis.patterns)


@pytest.mark.asyncio
async def test_high_load_low_recovery_silent_when_activity_not_elevated() -> None:
    analyzer = _analyzer(
        {"physical.activity": _dim("physical.activity", 0.4), "physical.recovery": _dim("physical.recovery", 0.3)},
        {"physical.activity": _baseline("physical.activity", 0.5), "physical.recovery": _baseline("physical.recovery", 0.6)},
    )
    analysis = await analyzer.analyze("u1")
    assert not any(p.name == "high_load_low_recovery" for p in analysis.patterns)


@pytest.mark.asyncio
async def test_sleep_inconsistency_fires_on_low_value_and_declining_trend() -> None:
    now = 1_000_000.0
    analyzer = _analyzer(
        {"lifestyle.sleep_consistency": _dim("lifestyle.sleep_consistency", 0.3)},
        {"lifestyle.sleep_consistency": _baseline("lifestyle.sleep_consistency", 0.6)},
        history={"lifestyle.sleep_consistency": _declining_points(now, 0.6, -0.1)},
    )
    analysis = await analyzer.analyze("u1")
    assert any(p.name == "sleep_inconsistency" for p in analysis.patterns)


@pytest.mark.asyncio
async def test_sleep_inconsistency_silent_without_declining_trend() -> None:
    now = 1_000_000.0
    flat_points = [(now - (3 - i) * _DAY, 0.3) for i in range(4)]
    analyzer = _analyzer(
        {"lifestyle.sleep_consistency": _dim("lifestyle.sleep_consistency", 0.3)},
        {"lifestyle.sleep_consistency": _baseline("lifestyle.sleep_consistency", 0.6)},
        history={"lifestyle.sleep_consistency": flat_points},
    )
    analysis = await analyzer.analyze("u1")
    assert not any(p.name == "sleep_inconsistency" for p in analysis.patterns)


@pytest.mark.asyncio
async def test_stress_accumulation_fires_on_rising_stress() -> None:
    now = 1_000_000.0
    analyzer = _analyzer(
        {"mental.stress": _dim("mental.stress", 0.7)},
        {},
        history={"mental.stress": _declining_points(now, 0.3, 0.15)},
    )
    analysis = await analyzer.analyze("u1")
    assert any(p.name == "stress_accumulation" for p in analysis.patterns)


@pytest.mark.asyncio
async def test_stress_accumulation_silent_when_falling() -> None:
    now = 1_000_000.0
    analyzer = _analyzer(
        {"mental.stress": _dim("mental.stress", 0.3)},
        {},
        history={"mental.stress": _declining_points(now, 0.7, -0.15)},
    )
    analysis = await analyzer.analyze("u1")
    assert not any(p.name == "stress_accumulation" for p in analysis.patterns)


@pytest.mark.asyncio
async def test_fatigue_without_load_fires() -> None:
    analyzer = _analyzer(
        {"mental.fatigue": _dim("mental.fatigue", 0.7), "physical.activity": _dim("physical.activity", 0.4)},
        {"physical.activity": _baseline("physical.activity", 0.5)},
    )
    analysis = await analyzer.analyze("u1")
    assert any(p.name == "fatigue_without_load" for p in analysis.patterns)


@pytest.mark.asyncio
async def test_fatigue_without_load_silent_when_fatigue_not_high() -> None:
    analyzer = _analyzer(
        {"mental.fatigue": _dim("mental.fatigue", 0.4), "physical.activity": _dim("physical.activity", 0.4)},
        {"physical.activity": _baseline("physical.activity", 0.5)},
    )
    analysis = await analyzer.analyze("u1")
    assert not any(p.name == "fatigue_without_load" for p in analysis.patterns)


@pytest.mark.asyncio
async def test_recovery_debt_fires_on_sustained_low_recovery() -> None:
    analyzer = _analyzer(
        {"physical.recovery": _dim("physical.recovery", 0.3, sample_count=6)},
        {"physical.recovery": _baseline("physical.recovery", 0.6)},
        min_sample_size=3,
    )
    analysis = await analyzer.analyze("u1")
    assert any(p.name == "recovery_debt" for p in analysis.patterns)


@pytest.mark.asyncio
async def test_recovery_debt_silent_on_single_low_reading() -> None:
    analyzer = _analyzer(
        {"physical.recovery": _dim("physical.recovery", 0.3, sample_count=1)},
        {"physical.recovery": _baseline("physical.recovery", 0.6)},
        min_sample_size=3,
    )
    analysis = await analyzer.analyze("u1")
    assert not any(p.name == "recovery_debt" for p in analysis.patterns)


@pytest.mark.asyncio
async def test_cognitive_dip_fires_with_poor_sleep_and_low_focus() -> None:
    analyzer = _analyzer(
        {
            "lifestyle.sleep_quality": _dim("lifestyle.sleep_quality", 0.3),
            "mental.focus": _dim("mental.focus", 0.3),
        },
        {
            "lifestyle.sleep_quality": _baseline("lifestyle.sleep_quality", 0.6),
            "mental.focus": _baseline("mental.focus", 0.6),
        },
    )
    analysis = await analyzer.analyze("u1")
    assert any(p.name == "cognitive_dip" for p in analysis.patterns)


@pytest.mark.asyncio
async def test_cognitive_dip_silent_without_poor_sleep() -> None:
    analyzer = _analyzer(
        {
            "lifestyle.sleep_quality": _dim("lifestyle.sleep_quality", 0.6),
            "mental.focus": _dim("mental.focus", 0.3),
        },
        {
            "lifestyle.sleep_quality": _baseline("lifestyle.sleep_quality", 0.6),
            "mental.focus": _baseline("mental.focus", 0.6),
        },
    )
    analysis = await analyzer.analyze("u1")
    assert not any(p.name == "cognitive_dip" for p in analysis.patterns)


@pytest.mark.asyncio
async def test_data_sufficiency_none_with_no_signals() -> None:
    analyzer = _analyzer({}, {})
    analysis = await analyzer.analyze("u1")
    assert analysis.data_sufficiency == "none"
    assert analysis.patterns == []


@pytest.mark.asyncio
async def test_data_sufficiency_sparse_caps_severity_to_watch() -> None:
    analyzer = _analyzer(
        {
            "physical.activity": _dim("physical.activity", 0.95, confidence=0.95, sample_count=1),
            "physical.recovery": _dim("physical.recovery", 0.05, confidence=0.95, sample_count=1),
        },
        {"physical.activity": _baseline("physical.activity", 0.3), "physical.recovery": _baseline("physical.recovery", 0.8)},
        min_sample_size=3,
    )
    analysis = await analyzer.analyze("u1")
    assert analysis.data_sufficiency == "sparse"
    pattern = next(p for p in analysis.patterns if p.name == "high_load_low_recovery")
    assert pattern.severity == "watch"


@pytest.mark.asyncio
async def test_data_sufficiency_adequate_allows_higher_severity() -> None:
    analyzer = _analyzer(
        {
            "physical.activity": _dim("physical.activity", 0.95, confidence=0.95, sample_count=6),
            "physical.recovery": _dim("physical.recovery", 0.05, confidence=0.95, sample_count=6),
        },
        {"physical.activity": _baseline("physical.activity", 0.3), "physical.recovery": _baseline("physical.recovery", 0.8)},
        min_sample_size=3,
    )
    analysis = await analyzer.analyze("u1")
    assert analysis.data_sufficiency == "adequate"
    pattern = next(p for p in analysis.patterns if p.name == "high_load_low_recovery")
    assert pattern.severity == "significant"


@pytest.mark.asyncio
async def test_scorecard_sign_corrects_inverted_dimensions() -> None:
    analyzer = _analyzer(
        {"mental.stress": _dim("mental.stress", 0.9, confidence=1.0)},
        {},
    )
    analysis = await analyzer.analyze("u1")
    assert analysis.scorecard["mental"] == pytest.approx(0.1, abs=1e-6)


@pytest.mark.asyncio
async def test_scorecard_omits_groups_with_no_data() -> None:
    analyzer = _analyzer({"physical.energy": _dim("physical.energy", 0.6)}, {})
    analysis = await analyzer.analyze("u1")
    assert "physical" in analysis.scorecard
    assert "cognitive" not in analysis.scorecard
