from __future__ import annotations

import pytest

from nexus.core.cost_tracker import CostTracker
from nexus.core.types import ModelInfo, Usage
from nexus.memory.storage import create_async_db_engine


async def _make_tracker(tmp_path) -> CostTracker:
    engine = create_async_db_engine(str(tmp_path / "nexus.db"))
    tracker = CostTracker(engine)
    await tracker.init()
    return tracker


@pytest.mark.asyncio
async def test_record_computes_cost_from_usage_and_rates(tmp_path) -> None:
    tracker = await _make_tracker(tmp_path)
    model_info = ModelInfo(
        id="gpt-4o-mini",
        provider="openai",
        cost_per_1k_input_tokens=0.00015,
        cost_per_1k_output_tokens=0.0006,
    )
    usage = Usage(prompt_tokens=1000, completion_tokens=2000)

    cost = await tracker.record(
        session_id="s1",
        provider_name="openai",
        model_id="gpt-4o-mini",
        usage=usage,
        model_info=model_info,
    )

    assert cost == pytest.approx(0.00015 + 0.0012)


@pytest.mark.asyncio
async def test_record_is_free_for_local_models_without_cost_fields(tmp_path) -> None:
    tracker = await _make_tracker(tmp_path)
    model_info = ModelInfo(id="mistral:7b", provider="local")
    usage = Usage(prompt_tokens=500, completion_tokens=500)

    cost = await tracker.record(
        session_id="s1",
        provider_name="local",
        model_id="mistral:7b",
        usage=usage,
        model_info=model_info,
    )

    assert cost == 0.0


@pytest.mark.asyncio
async def test_record_is_free_when_model_info_is_unknown(tmp_path) -> None:
    tracker = await _make_tracker(tmp_path)
    usage = Usage(prompt_tokens=500, completion_tokens=500)

    cost = await tracker.record(
        session_id="s1",
        provider_name="fake",
        model_id="fake-model",
        usage=usage,
        model_info=None,
    )

    assert cost == 0.0


@pytest.mark.asyncio
async def test_session_total_accumulates_across_records(tmp_path) -> None:
    tracker = await _make_tracker(tmp_path)
    model_info = ModelInfo(
        id="gpt-4o-mini",
        provider="openai",
        cost_per_1k_input_tokens=0.001,
        cost_per_1k_output_tokens=0.001,
    )

    await tracker.record(
        session_id="s1",
        provider_name="openai",
        model_id="gpt-4o-mini",
        usage=Usage(prompt_tokens=1000, completion_tokens=0),
        model_info=model_info,
    )
    await tracker.record(
        session_id="s1",
        provider_name="openai",
        model_id="gpt-4o-mini",
        usage=Usage(prompt_tokens=1000, completion_tokens=0),
        model_info=model_info,
    )
    await tracker.record(
        session_id="other-session",
        provider_name="openai",
        model_id="gpt-4o-mini",
        usage=Usage(prompt_tokens=1000, completion_tokens=0),
        model_info=model_info,
    )

    assert await tracker.session_total("s1") == pytest.approx(0.002)


@pytest.mark.asyncio
async def test_session_total_for_unknown_session_is_zero(tmp_path) -> None:
    tracker = await _make_tracker(tmp_path)
    assert await tracker.session_total("does-not-exist") == 0.0
