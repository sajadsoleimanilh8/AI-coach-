from __future__ import annotations

import pytest

from nexus.memory.storage import create_async_db_engine
from nexus.personal.state import PersonalStateEngine
from nexus.sports.adapter import MatchAnalysis, SportsMetric
from nexus.sports.ingest import ingest_player_metrics_as_signals

_MAPPING = {
    "decision_making_score": "sports.decision_making",
    "off_ball_movement_score": "sports.positioning",
    "press_resistance_score": "sports.agility",
}


def _metric(name: str, value, confidence: str = "normal", player_id: int = 7) -> SportsMetric:
    return SportsMetric(
        metric_name=name, value=value, method="deterministic", confidence=confidence,
        sample_size=10, sub_scores={}, player_id=player_id,
    )


async def _make_engine(tmp_path) -> PersonalStateEngine:
    async_engine = create_async_db_engine(str(tmp_path / "nexus.db"))
    engine = PersonalStateEngine(async_engine)
    await engine.init()
    return engine


@pytest.mark.asyncio
async def test_available_metrics_are_ingested_with_correct_normalization(tmp_path) -> None:
    engine = await _make_engine(tmp_path)
    analysis = MatchAnalysis(
        match_id="m1",
        team_metrics=[],
        player_metrics=[_metric("decision_making_score", 80.0)],
        unavailable=[],
        coverage=1.0,
    )

    recorded = await ingest_player_metrics_as_signals(
        engine, user_id="u1", analysis=analysis, mapping=_MAPPING
    )

    assert recorded == 1
    state = await engine.get_state("u1")
    assert state.dimensions["sports.decision_making"].value == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_recorded_signals_use_inferred_source_confidence(tmp_path) -> None:
    engine = await _make_engine(tmp_path)
    analysis = MatchAnalysis(
        match_id="m1", team_metrics=[], player_metrics=[_metric("decision_making_score", 80.0)],
        unavailable=[], coverage=1.0,
    )

    await ingest_player_metrics_as_signals(engine, user_id="u1", analysis=analysis, mapping=_MAPPING)

    state = await engine.get_state("u1")
    dim = state.dimensions["sports.decision_making"]
    assert dim.confidence == pytest.approx(0.11, abs=1e-6)


@pytest.mark.asyncio
async def test_unavailable_metrics_are_skipped_entirely(tmp_path) -> None:
    engine = await _make_engine(tmp_path)
    analysis = MatchAnalysis(
        match_id="m1",
        team_metrics=[],
        player_metrics=[
            _metric("decision_making_score", 80.0),
            _metric("press_resistance_score", None, confidence="low_upstream_confidence"),
        ],
        unavailable=[_metric("press_resistance_score", None, confidence="low_upstream_confidence")],
        coverage=0.5,
    )

    recorded = await ingest_player_metrics_as_signals(
        engine, user_id="u1", analysis=analysis, mapping=_MAPPING
    )

    assert recorded == 1
    state = await engine.get_state("u1")
    assert "sports.agility" not in state.dimensions


@pytest.mark.asyncio
async def test_metrics_absent_from_mapping_are_skipped(tmp_path) -> None:
    engine = await _make_engine(tmp_path)
    analysis = MatchAnalysis(
        match_id="m1", team_metrics=[], player_metrics=[_metric("finishing_efficiency_score", 90.0)],
        unavailable=[], coverage=1.0,
    )

    recorded = await ingest_player_metrics_as_signals(
        engine, user_id="u1", analysis=analysis, mapping=_MAPPING
    )

    assert recorded == 0
    state = await engine.get_state("u1")
    assert state.dimensions == {}


@pytest.mark.asyncio
async def test_multiple_available_metrics_all_recorded(tmp_path) -> None:
    engine = await _make_engine(tmp_path)
    analysis = MatchAnalysis(
        match_id="m1",
        team_metrics=[],
        player_metrics=[
            _metric("decision_making_score", 80.0),
            _metric("off_ball_movement_score", 40.0),
            _metric("press_resistance_score", 100.0),
        ],
        unavailable=[],
        coverage=1.0,
    )

    recorded = await ingest_player_metrics_as_signals(
        engine, user_id="u1", analysis=analysis, mapping=_MAPPING
    )

    assert recorded == 3
    state = await engine.get_state("u1")
    assert state.dimensions["sports.decision_making"].value == pytest.approx(0.8)
    assert state.dimensions["sports.positioning"].value == pytest.approx(0.4)
    assert state.dimensions["sports.agility"].value == pytest.approx(1.0)
