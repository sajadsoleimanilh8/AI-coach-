from __future__ import annotations

import time

import pytest

from nexus.memory.storage import (
    PersonalSignalRecord,
    create_async_db_engine,
    init_db,
    make_session_factory,
)
from nexus.personal.baseline import BaselineCalculator

_DAY = 86400.0


async def _make_engine(tmp_path):
    async_engine = create_async_db_engine(str(tmp_path / "nexus.db"))
    await init_db(async_engine)  # BaselineCalculator has no init() of its own — read-only
    return async_engine


async def _insert_raw(async_engine, *, user_id, dimension, value, age_days) -> None:
    session_factory = make_session_factory(async_engine)
    async with session_factory() as db:
        db.add(
            PersonalSignalRecord(
                user_id=user_id,
                dimension=dimension,
                value=value,
                source="explicit",
                note="",
                recorded_at=time.time() - age_days * _DAY,
            )
        )
        await db.commit()


def _calculator(async_engine, *, window_days=90.0, recent_window_days=14.0, min_samples=3):
    return BaselineCalculator(
        async_engine, window_days=window_days, recent_window_days=recent_window_days,
        min_samples=min_samples,
    )


@pytest.mark.asyncio
async def test_baseline_excludes_the_recent_window(tmp_path) -> None:
    async_engine = await _make_engine(tmp_path)
    calculator = _calculator(async_engine, recent_window_days=14.0, min_samples=1)

    # Inside the last 14 days — must NOT count toward the baseline, or
    # "current" would be compared against itself.
    await _insert_raw(async_engine, user_id="u1", dimension="physical.energy", value=0.9, age_days=1.0)
    # Outside the recent window, inside the 90-day baseline window.
    await _insert_raw(async_engine, user_id="u1", dimension="physical.energy", value=0.3, age_days=20.0)

    baselines = await calculator.get_baselines("u1")

    assert baselines["physical.energy"].value == pytest.approx(0.3)
    assert baselines["physical.energy"].sample_count == 1


@pytest.mark.asyncio
async def test_baseline_excludes_signals_older_than_the_window(tmp_path) -> None:
    async_engine = await _make_engine(tmp_path)
    calculator = _calculator(async_engine, window_days=90.0, recent_window_days=14.0, min_samples=1)

    await _insert_raw(async_engine, user_id="u1", dimension="physical.energy", value=0.3, age_days=20.0)
    # Older than the 90-day window entirely.
    await _insert_raw(async_engine, user_id="u1", dimension="physical.energy", value=0.9, age_days=200.0)

    baselines = await calculator.get_baselines("u1")

    assert baselines["physical.energy"].value == pytest.approx(0.3)
    assert baselines["physical.energy"].sample_count == 1


@pytest.mark.asyncio
async def test_baseline_is_the_unweighted_mean_of_the_window(tmp_path) -> None:
    async_engine = await _make_engine(tmp_path)
    calculator = _calculator(async_engine, min_samples=1)

    for value, age in [(0.2, 20.0), (0.4, 30.0), (0.6, 40.0)]:
        await _insert_raw(
            async_engine, user_id="u1", dimension="mental.mood", value=value, age_days=age
        )

    baselines = await calculator.get_baselines("u1")

    assert baselines["mental.mood"].value == pytest.approx(0.4)
    assert baselines["mental.mood"].sample_count == 3


@pytest.mark.asyncio
async def test_dimension_below_min_samples_is_absent(tmp_path) -> None:
    async_engine = await _make_engine(tmp_path)
    calculator = _calculator(async_engine, min_samples=5)

    for age in (20.0, 30.0):
        await _insert_raw(
            async_engine, user_id="u1", dimension="physical.energy", value=0.5, age_days=age
        )

    baselines = await calculator.get_baselines("u1")

    assert "physical.energy" not in baselines


@pytest.mark.asyncio
async def test_no_signals_returns_empty_baselines(tmp_path) -> None:
    async_engine = await _make_engine(tmp_path)
    calculator = _calculator(async_engine)

    assert await calculator.get_baselines("nobody") == {}
