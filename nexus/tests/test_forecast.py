from __future__ import annotations

import time

import pytest

from nexus.memory.storage import create_async_db_engine
from nexus.personal.forecast import StateForecaster
from nexus.personal.state import PersonalStateEngine

_SECONDS_PER_DAY = 86400.0


async def _engine_with(tmp_path, dimension: str, values: list[float], *, spacing: float = 1.0):
    engine = PersonalStateEngine(create_async_db_engine(str(tmp_path / "nexus.db")))
    await engine.init()
    now = time.time()
    for index, value in enumerate(values):
        offset_days = (len(values) - 1 - index) * spacing
        await engine.record_signal_at(
            user_id="u1", dimension=dimension, value=value, source="explicit",
            recorded_at=now - offset_days * _SECONDS_PER_DAY,
        )
    return engine


async def _forecast(engine, *, horizon_days: int = 7, **kwargs):
    forecaster = StateForecaster(engine, **kwargs)
    result = await forecaster.forecast("u1", horizon_days=horizon_days)
    return result


@pytest.mark.asyncio
async def test_known_slope_projects_correctly(tmp_path) -> None:
    # +0.02/day over four clean points; current ~0.532, +0.14 over 7 days.
    engine = await _engine_with(tmp_path, "physical.energy", [0.5, 0.52, 0.54, 0.56])

    result = await _forecast(engine, horizon_days=7)
    forecast = result.forecasts[0]

    assert forecast.method == "trend_extrapolation"
    assert forecast.projected_value == pytest.approx(0.672, abs=0.02)
    assert forecast.trend.slope == pytest.approx(0.02, abs=0.002)


@pytest.mark.asyncio
async def test_a_declining_series_projects_downward(tmp_path) -> None:
    engine = await _engine_with(tmp_path, "physical.energy", [0.8, 0.78, 0.76, 0.74])

    forecast = (await _forecast(engine, horizon_days=7)).forecasts[0]

    assert forecast.projected_value < forecast.current_value


@pytest.mark.asyncio
async def test_a_flat_series_projects_to_itself(tmp_path) -> None:
    engine = await _engine_with(tmp_path, "physical.energy", [0.6, 0.6, 0.6, 0.6])

    forecast = (await _forecast(engine, horizon_days=10)).forecasts[0]

    assert forecast.method == "trend_extrapolation"
    assert forecast.projected_value == pytest.approx(0.6, abs=0.001)


@pytest.mark.asyncio
async def test_too_few_samples_returns_insufficient_data_and_no_number(tmp_path) -> None:
    """The discipline that matters most here: no data means say nothing, not
    produce a confident-looking default."""
    engine = await _engine_with(tmp_path, "physical.energy", [0.5, 0.55, 0.6])

    forecast = (await _forecast(engine)).forecasts[0]

    assert forecast.method == "insufficient_data"
    assert forecast.projected_value is None
    assert forecast.confidence == 0.0


@pytest.mark.asyncio
async def test_a_single_point_returns_insufficient_data(tmp_path) -> None:
    engine = await _engine_with(tmp_path, "physical.energy", [0.5])

    assert (await _forecast(engine)).forecasts[0].method == "insufficient_data"


@pytest.mark.asyncio
async def test_a_noisy_series_below_confidence_threshold_returns_insufficient_data(tmp_path) -> None:
    # Enough samples, but the line explains almost none of the scatter.
    engine = await _engine_with(tmp_path, "physical.energy", [0.2, 0.9, 0.25, 0.85])

    forecast = (await _forecast(engine, min_trend_confidence=0.5)).forecasts[0]

    assert forecast.method == "insufficient_data"
    assert forecast.projected_value is None


@pytest.mark.asyncio
async def test_insufficient_data_carries_a_plain_caveat(tmp_path) -> None:
    engine = await _engine_with(tmp_path, "physical.energy", [0.5, 0.55])

    forecast = (await _forecast(engine)).forecasts[0]

    assert "Not enough recorded data" in forecast.caveat
    assert "guess" in forecast.caveat


@pytest.mark.asyncio
async def test_caveat_states_it_is_extrapolation_not_prediction(tmp_path) -> None:
    engine = await _engine_with(tmp_path, "physical.energy", [0.5, 0.52, 0.54, 0.56])

    result = await _forecast(engine)

    assert "not a prediction" in result.caveat
    assert "extrapolation" in result.forecasts[0].caveat


@pytest.mark.asyncio
async def test_confidence_decays_with_horizon(tmp_path) -> None:
    engine = await _engine_with(tmp_path, "physical.energy", [0.5, 0.52, 0.54, 0.56])

    near = (await _forecast(engine, horizon_days=3)).forecasts[0]
    far = (await _forecast(engine, horizon_days=28)).forecasts[0]

    assert near.confidence > far.confidence
    assert far.confidence > 0.0


@pytest.mark.asyncio
async def test_horizon_is_hard_capped_at_thirty_days(tmp_path) -> None:
    engine = await _engine_with(tmp_path, "physical.energy", [0.5, 0.52, 0.54, 0.56])

    result = await _forecast(engine, horizon_days=365)

    # Capped rather than rejected — the response reports the horizon it used.
    assert result.horizon_days == 30
    assert result.forecasts[0].horizon_days == 30


@pytest.mark.asyncio
async def test_the_cap_is_configurable(tmp_path) -> None:
    engine = await _engine_with(tmp_path, "physical.energy", [0.5, 0.52, 0.54, 0.56])

    result = await _forecast(engine, horizon_days=100, max_horizon_days=14)

    assert result.horizon_days == 14


@pytest.mark.asyncio
async def test_projection_is_clamped_to_the_scale(tmp_path) -> None:
    # A steep slope over a long horizon would otherwise report 3.8 on a
    # [0, 1] scale.
    engine = await _engine_with(tmp_path, "physical.energy", [0.7, 0.8, 0.9, 1.0])

    forecast = (await _forecast(engine, horizon_days=30)).forecasts[0]

    assert forecast.projected_value == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_projection_is_clamped_at_the_bottom_too(tmp_path) -> None:
    engine = await _engine_with(tmp_path, "physical.energy", [0.4, 0.3, 0.2, 0.1])

    forecast = (await _forecast(engine, horizon_days=30)).forecasts[0]

    assert forecast.projected_value == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_every_recorded_dimension_gets_a_forecast(tmp_path) -> None:
    engine = PersonalStateEngine(create_async_db_engine(str(tmp_path / "nexus.db")))
    await engine.init()
    now = time.time()
    for dimension in ("physical.energy", "mental.mood"):
        for index, value in enumerate([0.5, 0.52, 0.54, 0.56]):
            await engine.record_signal_at(
                user_id="u1", dimension=dimension, value=value, source="explicit",
                recorded_at=now - (3 - index) * _SECONDS_PER_DAY,
            )

    result = await _forecast(engine)

    assert {f.dimension for f in result.forecasts} == {"physical.energy", "mental.mood"}


@pytest.mark.asyncio
async def test_a_user_with_no_signals_gets_an_empty_forecast(tmp_path) -> None:
    engine = PersonalStateEngine(create_async_db_engine(str(tmp_path / "nexus.db")))
    await engine.init()

    result = await _forecast(engine)

    assert result.forecasts == []


@pytest.mark.asyncio
async def test_an_inverted_dimension_projects_on_its_raw_value(tmp_path) -> None:
    # The sign flip belongs to trend DIRECTION, not to the projection: rising
    # stress readings project to higher stress readings.
    engine = await _engine_with(tmp_path, "mental.stress", [0.3, 0.32, 0.34, 0.36])

    forecast = (await _forecast(engine, horizon_days=7)).forecasts[0]

    assert forecast.projected_value > forecast.current_value
    assert forecast.trend.direction == "declining"  # worse, for an inverted dimension
