from __future__ import annotations

import time

import pytest

from nexus.core.exceptions import InvalidSignalError
from nexus.memory.storage import PersonalSignalRecord, create_async_db_engine, make_session_factory
from nexus.personal.state import PersonalStateEngine

_DAY = 86400.0


async def _make_engine(tmp_path) -> PersonalStateEngine:
    async_engine = create_async_db_engine(str(tmp_path / "nexus.db"))
    engine = PersonalStateEngine(async_engine, half_life_days=7.0, recent_window_days=14.0)
    await engine.init()
    return engine


async def _insert_raw(async_engine, *, user_id, dimension, value, source, age_days) -> None:
    session_factory = make_session_factory(async_engine)
    async with session_factory() as db:
        db.add(
            PersonalSignalRecord(
                user_id=user_id,
                dimension=dimension,
                value=value,
                source=source,
                note="",
                recorded_at=time.time() - age_days * _DAY,
            )
        )
        await db.commit()


@pytest.mark.asyncio
async def test_record_signal_rejects_unknown_dimension(tmp_path) -> None:
    engine = await _make_engine(tmp_path)
    with pytest.raises(InvalidSignalError):
        await engine.record_signal(
            user_id="u1", dimension="not.a.real.dimension", value=0.5, source="explicit"
        )


@pytest.mark.asyncio
async def test_record_signal_rejects_value_outside_unit_range(tmp_path) -> None:
    engine = await _make_engine(tmp_path)
    with pytest.raises(InvalidSignalError):
        await engine.record_signal(
            user_id="u1", dimension="physical.energy", value=1.5, source="explicit"
        )
    with pytest.raises(InvalidSignalError):
        await engine.record_signal(
            user_id="u1", dimension="physical.energy", value=-0.1, source="explicit"
        )


@pytest.mark.asyncio
async def test_record_signal_rejects_unknown_source(tmp_path) -> None:
    engine = await _make_engine(tmp_path)
    with pytest.raises(InvalidSignalError):
        await engine.record_signal(
            user_id="u1", dimension="physical.energy", value=0.5, source="made_up_source"
        )


@pytest.mark.asyncio
async def test_dimensions_with_no_signals_are_absent_not_defaulted(tmp_path) -> None:
    engine = await _make_engine(tmp_path)
    await engine.record_signal(
        user_id="u1", dimension="physical.energy", value=0.6, source="explicit"
    )

    state = await engine.get_state("u1")

    assert "physical.energy" in state.dimensions
    assert "physical.recovery" not in state.dimensions
    assert "mental.stress" not in state.dimensions


@pytest.mark.asyncio
async def test_get_state_excludes_signals_older_than_recent_window(tmp_path) -> None:
    engine = await _make_engine(tmp_path)
    async_engine = engine._engine
    await _insert_raw(
        async_engine,
        user_id="u1",
        dimension="physical.energy",
        value=0.9,
        source="explicit",
        age_days=30,
    )

    state = await engine.get_state("u1")

    assert "physical.energy" not in state.dimensions


@pytest.mark.asyncio
async def test_recency_and_source_weighted_mean_matches_hand_computed_value(tmp_path) -> None:
    engine = await _make_engine(tmp_path)
    async_engine = engine._engine

    await _insert_raw(
        async_engine, user_id="u1", dimension="physical.energy", value=0.8,
        source="explicit", age_days=0.0,
    )
    await _insert_raw(
        async_engine, user_id="u1", dimension="physical.energy", value=0.4,
        source="behavioral", age_days=7.0,
    )

    state = await engine.get_state("u1")
    dim = state.dimensions["physical.energy"]

    expected_value = (0.95 * 0.8 + 0.375 * 0.4) / (0.95 + 0.375)
    expected_confidence = ((0.95 + 0.75) / 2) * min(1.0, 2 / 5)

    assert dim.value == pytest.approx(expected_value, rel=1e-6)
    assert dim.value == pytest.approx(0.6867924528301887, rel=1e-6)
    assert dim.confidence == pytest.approx(expected_confidence, rel=1e-6)
    assert dim.confidence == pytest.approx(0.34, rel=1e-6)
    assert dim.sample_count == 2


@pytest.mark.asyncio
async def test_confidence_scales_with_sample_count_up_to_five(tmp_path) -> None:
    engine = await _make_engine(tmp_path)
    async_engine = engine._engine

    for _ in range(5):
        await _insert_raw(
            async_engine, user_id="u1", dimension="mental.focus", value=0.5,
            source="explicit", age_days=0.0,
        )

    state = await engine.get_state("u1")
    dim = state.dimensions["mental.focus"]

    assert dim.confidence == pytest.approx(0.95, rel=1e-6)
    assert dim.sample_count == 5


@pytest.mark.asyncio
async def test_delete_all_removes_every_signal_for_the_user(tmp_path) -> None:
    engine = await _make_engine(tmp_path)
    await engine.record_signal(user_id="u1", dimension="physical.energy", value=0.5, source="explicit")
    await engine.record_signal(user_id="u2", dimension="physical.energy", value=0.5, source="explicit")

    await engine.delete_all("u1")

    state_u1 = await engine.get_state("u1")
    state_u2 = await engine.get_state("u2")
    assert state_u1.dimensions == {}
    assert "physical.energy" in state_u2.dimensions
