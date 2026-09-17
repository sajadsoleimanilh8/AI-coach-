from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from nexus.api.schemas import (
    BaselineSchema,
    ClearedWeaknessSchema,
    DimensionForecastSchema,
    DimensionStateSchema,
    PersonalBaselinesResponse,
    PersonalForecastResponse,
    PersonalProfileResponse,
    PersonalProfileSetRequest,
    PersonalSignalRequest,
    PersonalStateResponse,
    PersonalWeaknessesResponse,
    ScenarioSimulateRequest,
    ScenarioSimulateResponse,
    TrendSchema,
    WeaknessSchema,
)
from nexus.api.services import get_services
from nexus.core.exceptions import InvalidSignalError
from nexus.personal.baseline import BaselineCalculator
from nexus.personal.forecast import ScenarioSimulator, StateForecaster
from nexus.personal.profile import ProfileStore
from nexus.personal.state import PersonalState, PersonalStateEngine
from nexus.personal.trends import Trend
from nexus.personal.weakness import Weakness, WeaknessEngine

router = APIRouter()


def _state_to_schema(state: PersonalState) -> PersonalStateResponse:
    return PersonalStateResponse(
        user_id=state.user_id,
        dimensions={
            dim: DimensionStateSchema(
                dimension=ds.dimension,
                value=ds.value,
                confidence=ds.confidence,
                sample_count=ds.sample_count,
                latest_at=ds.latest_at,
            )
            for dim, ds in state.dimensions.items()
        },
        computed_at=state.computed_at,
    )


def _trend_to_schema(trend: Trend | None) -> TrendSchema | None:
    if trend is None:
        return None
    return TrendSchema(
        dimension=trend.dimension,
        direction=trend.direction,
        slope=trend.slope,
        confidence=trend.confidence,
        sample_count=trend.sample_count,
    )


def _weakness_to_schema(weakness: Weakness) -> WeaknessSchema:
    return WeaknessSchema(
        dimension=weakness.dimension,
        current=weakness.current,
        baseline=weakness.baseline,
        deviation=weakness.deviation,
        priority=weakness.priority,
        confidence=weakness.confidence,
        trend=_trend_to_schema(weakness.trend),
        explanation=weakness.explanation,
    )


@router.get("/personal/{user_id}/state", response_model=PersonalStateResponse)
async def get_state(user_id: str, request: Request) -> PersonalStateResponse:
    engine: PersonalStateEngine = get_services(request).personal_state_engine
    return _state_to_schema(await engine.get_state(user_id))


@router.post("/personal/{user_id}/signal", response_model=PersonalStateResponse)
async def record_signal(
    user_id: str, payload: PersonalSignalRequest, request: Request
) -> PersonalStateResponse:
    engine: PersonalStateEngine = get_services(request).personal_state_engine
    try:
        await engine.record_signal(
            user_id=user_id,
            dimension=payload.dimension,
            value=payload.value,
            source=payload.source,
            note=payload.note,
        )
    except InvalidSignalError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _state_to_schema(await engine.get_state(user_id))


@router.get("/personal/{user_id}/baselines", response_model=PersonalBaselinesResponse)
async def get_baselines(user_id: str, request: Request) -> PersonalBaselinesResponse:
    calculator: BaselineCalculator = get_services(request).baseline_calculator
    baselines = await calculator.get_baselines(user_id)
    return PersonalBaselinesResponse(
        user_id=user_id,
        baselines={
            dim: BaselineSchema(
                dimension=b.dimension,
                value=b.value,
                sample_count=b.sample_count,
                window_days=b.window_days,
            )
            for dim, b in baselines.items()
        },
    )


@router.get("/personal/{user_id}/weaknesses", response_model=PersonalWeaknessesResponse)
async def get_weaknesses(user_id: str, request: Request) -> PersonalWeaknessesResponse:
    weakness_engine: WeaknessEngine = get_services(request).weakness_engine
    weaknesses = await weakness_engine.detect(user_id)
    return PersonalWeaknessesResponse(
        user_id=user_id, weaknesses=[_weakness_to_schema(w) for w in weaknesses]
    )


@router.get("/personal/{user_id}/profile", response_model=PersonalProfileResponse)
async def get_profile(user_id: str, request: Request) -> PersonalProfileResponse:
    profile_store: ProfileStore = get_services(request).profile_store
    return PersonalProfileResponse(user_id=user_id, profile=await profile_store.get_profile(user_id))


@router.put("/personal/{user_id}/profile", response_model=PersonalProfileResponse)
async def set_profile(
    user_id: str, payload: PersonalProfileSetRequest, request: Request
) -> PersonalProfileResponse:
    profile_store: ProfileStore = get_services(request).profile_store
    await profile_store.set_profile(user_id, payload.profile)
    return PersonalProfileResponse(user_id=user_id, profile=payload.profile)


@router.get("/personal/{user_id}/forecast", response_model=PersonalForecastResponse)
async def get_forecast(
    user_id: str, request: Request, horizon_days: int = 14
) -> PersonalForecastResponse:
    forecaster: StateForecaster = get_services(request).state_forecaster
    result = await forecaster.forecast(user_id, horizon_days=horizon_days)
    return PersonalForecastResponse(
        user_id=result.user_id,
        horizon_days=result.horizon_days,
        forecasts=[
            DimensionForecastSchema(
                dimension=f.dimension,
                current_value=f.current_value,
                projected_value=f.projected_value,
                horizon_days=f.horizon_days,
                confidence=f.confidence,
                method=f.method,
                caveat=f.caveat,
                trend=_trend_to_schema(f.trend),
            )
            for f in result.forecasts
        ],
        caveat=result.caveat,
    )


@router.post("/personal/{user_id}/simulate", response_model=ScenarioSimulateResponse)
async def simulate_scenario(
    user_id: str, payload: ScenarioSimulateRequest, request: Request
) -> ScenarioSimulateResponse:
    simulator: ScenarioSimulator = get_services(request).scenario_simulator
    result = await simulator.simulate(user_id, payload.deltas)
    return ScenarioSimulateResponse(
        user_id=result.user_id,
        deltas=result.deltas,
        weaknesses_before=result.weaknesses_before,
        weaknesses_after=result.weaknesses_after,
        changes=[
            ClearedWeaknessSchema(
                dimension=c.dimension,
                deviation_before=c.deviation_before,
                deviation_after=c.deviation_after,
                cleared=c.cleared,
            )
            for c in result.changes
        ],
        caveat=result.caveat,
    )


@router.delete("/personal/{user_id}", status_code=204)
async def delete_personal_data(user_id: str, request: Request) -> None:
    engine: PersonalStateEngine = get_services(request).personal_state_engine
    profile_store: ProfileStore = get_services(request).profile_store
    await engine.delete_all(user_id)
    await profile_store.delete_profile(user_id)
