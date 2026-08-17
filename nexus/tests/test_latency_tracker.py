from __future__ import annotations

import pytest
from sqlalchemy import select

from nexus.core.latency_tracker import LatencyTracker
from nexus.memory.storage import LatencyRecord, create_async_db_engine, make_session_factory


async def _make_tracker(tmp_path, **kwargs) -> LatencyTracker:
    engine = create_async_db_engine(str(tmp_path / "nexus.db"))
    tracker = LatencyTracker(engine, **kwargs)
    await tracker.init()
    return tracker


@pytest.mark.asyncio
async def test_p50_is_none_below_min_samples(tmp_path) -> None:
    tracker = await _make_tracker(tmp_path, min_samples=5)

    for latency in (0.1, 0.2, 0.3):
        await tracker.record(
            session_id="s1", provider_name="local", model_id="mistral:7b", latency_seconds=latency
        )

    assert tracker.p50("local", "mistral:7b") is None
    assert tracker.sample_count("local", "mistral:7b") == 3


@pytest.mark.asyncio
async def test_p50_is_sane_at_min_samples(tmp_path) -> None:
    tracker = await _make_tracker(tmp_path, min_samples=5)

    for latency in (1.0, 2.0, 3.0, 4.0, 5.0):
        await tracker.record(
            session_id="s1", provider_name="openai", model_id="gpt-4o-mini", latency_seconds=latency
        )

    assert tracker.p50("openai", "gpt-4o-mini") == 3.0


@pytest.mark.asyncio
async def test_p50_averages_the_two_middle_values_for_even_sample_count(tmp_path) -> None:
    tracker = await _make_tracker(tmp_path, min_samples=4)

    for latency in (1.0, 2.0, 3.0, 4.0):
        await tracker.record(
            session_id="s1", provider_name="openai", model_id="gpt-4o-mini", latency_seconds=latency
        )

    assert tracker.p50("openai", "gpt-4o-mini") == pytest.approx(2.5)


@pytest.mark.asyncio
async def test_window_size_evicts_oldest_sample(tmp_path) -> None:
    tracker = await _make_tracker(tmp_path, window_size=3, min_samples=3)

    for latency in (1.0, 2.0, 3.0, 4.0, 5.0):
        await tracker.record(
            session_id="s1", provider_name="openai", model_id="gpt-4o-mini", latency_seconds=latency
        )

    assert tracker.sample_count("openai", "gpt-4o-mini") == 3
    assert tracker.p50("openai", "gpt-4o-mini") == 4.0


@pytest.mark.asyncio
async def test_record_persists_a_row_per_call(tmp_path) -> None:
    engine = create_async_db_engine(str(tmp_path / "nexus.db"))
    tracker = LatencyTracker(engine)
    await tracker.init()

    await tracker.record(
        session_id="s1", provider_name="local", model_id="mistral:7b", latency_seconds=0.42
    )
    await tracker.record(
        session_id="s1", provider_name="local", model_id="mistral:7b", latency_seconds=0.55
    )

    session_factory = make_session_factory(engine)
    async with session_factory() as db:
        result = await db.execute(select(LatencyRecord).where(LatencyRecord.session_id == "s1"))
        rows = result.scalars().all()

    assert len(rows) == 2
    assert {row.latency_seconds for row in rows} == {0.42, 0.55}
    assert all(row.provider_name == "local" and row.model_id == "mistral:7b" for row in rows)


@pytest.mark.asyncio
async def test_p50_is_scoped_per_provider_and_model(tmp_path) -> None:
    tracker = await _make_tracker(tmp_path, min_samples=1)

    await tracker.record(
        session_id="s1", provider_name="local", model_id="mistral:7b", latency_seconds=1.0
    )
    await tracker.record(
        session_id="s1", provider_name="openai", model_id="gpt-4o-mini", latency_seconds=9.0
    )

    assert tracker.p50("local", "mistral:7b") == 1.0
    assert tracker.p50("openai", "gpt-4o-mini") == 9.0
