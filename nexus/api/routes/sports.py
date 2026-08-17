from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from nexus.api.schemas import (
    CoachReportResponse,
    PreMatchAssessmentSchema,
    PreMatchCoachReportResponse,
    SportsIngestRequest,
    SportsIngestResponse,
    TacticalFindingSchema,
    VideoAnalyzeRequest,
)
from nexus.core.exceptions import ProviderUnavailableError
from nexus.personal.state import PersonalStateEngine
from nexus.sports.adapter import SportsDataAdapter
from nexus.sports.coach import CoachAssistant, CoachReport, PreMatchCoachReport
from nexus.sports.ingest import ingest_player_metrics_as_signals
from nexus.sports.video import VideoAnalysisService

router = APIRouter()


def _to_response(report: CoachReport) -> CoachReportResponse:
    return CoachReportResponse(
        match_id=report.match_id,
        findings=[
            TacticalFindingSchema(
                area=f.area,
                assessment=f.assessment,
                supporting_metrics=f.supporting_metrics,
                confidence=f.confidence,
                explanation=f.explanation,
            )
            for f in report.findings
        ],
        unavailable_metrics=report.unavailable_metrics,
        coverage=report.coverage,
        narrative=report.narrative,
        model_used=report.model_used,
    )


def _to_prematch_response(report: PreMatchCoachReport) -> PreMatchCoachReportResponse:
    assessment = report.assessment
    return PreMatchCoachReportResponse(
        player_id=report.player_id,
        match_id=report.match_id,
        assessment=PreMatchAssessmentSchema(
            player_id=assessment.player_id,
            match_id=assessment.match_id,
            physical_readiness=assessment.physical_readiness,
            fatigue_score=assessment.fatigue_score,
            recovery_score=assessment.recovery_score,
            performance_risk=assessment.performance_risk,
            workload_risk=assessment.workload_risk,
            key_positive_factors=assessment.key_positive_factors,
            key_negative_factors=assessment.key_negative_factors,
            method=assessment.method,
            schema_version=assessment.schema_version,
            computed_at=assessment.computed_at,
            data_source=assessment.data_source,
            notes=assessment.notes,
            disclaimer=assessment.disclaimer,
        ),
        narrative=report.narrative,
        model_used=report.model_used,
    )


@router.get("/sports/{match_id}/report", response_model=CoachReportResponse)
async def get_match_report(match_id: str, request: Request) -> CoachReportResponse:
    coach: CoachAssistant = request.app.state.coach_assistant
    try:
        report = await coach.build_report(match_id)
    except ProviderUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _to_response(report)


@router.get("/sports/{match_id}/player/{player_id}", response_model=CoachReportResponse)
async def get_player_report(match_id: str, player_id: int, request: Request) -> CoachReportResponse:
    coach: CoachAssistant = request.app.state.coach_assistant
    try:
        report = await coach.build_report(match_id, player_id)
    except ProviderUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _to_response(report)


@router.get(
    "/sports/{match_id}/player/{player_id}/prematch_report",
    response_model=PreMatchCoachReportResponse,
)
async def get_prematch_report(
    match_id: str, player_id: str, request: Request
) -> PreMatchCoachReportResponse:
    """Narrates the player's latest pre-match readiness assessment."""
    coach: CoachAssistant = request.app.state.coach_assistant
    try:
        report = await coach.build_prematch_report(player_id, match_id)
    except ProviderUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"No pre-match assessment submitted yet for player_id={player_id}",
        )
    return _to_prematch_response(report)


@router.post("/sports/video/analyze", response_model=CoachReportResponse)
async def analyze_video(payload: VideoAnalyzeRequest, request: Request) -> CoachReportResponse:
    """Submits footage to the football backend's own processing endpoint,
    polls until it finishes, and narrates the result through the Group C
    adapter. No vision code runs in nexus/ — that stays in backend/ and
    ai/, reached over HTTP."""
    service: VideoAnalysisService = request.app.state.video_analysis_service
    try:
        report = await service.analyze(
            video_path_or_url=payload.video_path_or_url,
            match_id=payload.match_id,
            player_id=payload.player_id,
        )
    except ProviderUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _to_response(report)


@router.post("/sports/{match_id}/ingest", response_model=SportsIngestResponse)
async def ingest_match_metrics(
    match_id: str, payload: SportsIngestRequest, request: Request
) -> SportsIngestResponse:
    adapter: SportsDataAdapter = request.app.state.sports_adapter
    state_engine: PersonalStateEngine = request.app.state.personal_state_engine
    mapping: dict[str, str] = request.app.state.settings.sports.metric_dimension_map

    try:
        analysis = await adapter.get_player_analysis(match_id, payload.player_id)
    except ProviderUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    recorded = await ingest_player_metrics_as_signals(
        state_engine, user_id=payload.user_id, analysis=analysis, mapping=mapping
    )
    return SportsIngestResponse(
        match_id=match_id, user_id=payload.user_id, signals_recorded=recorded
    )
