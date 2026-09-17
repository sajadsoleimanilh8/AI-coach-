from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from nexus.api.schemas import (
    AdjustmentSchema,
    CoachReportResponse,
    GamePlanSchema,
    OpponentStrengthSchema,
    OpponentWeaknessSchema,
    PreMatchAssessmentSchema,
    PreMatchCoachReportResponse,
    PrincipleSchema,
    ProposedShapeSchema,
    SportsIngestRequest,
    SportsIngestResponse,
    TacticalFindingSchema,
    TimelineSectionsSchema,
    VideoAnalyzeRequest,
)
from nexus.api.services import get_services
from nexus.core.exceptions import ProviderUnavailableError
from nexus.personal.state import PersonalStateEngine
from nexus.sports.adapter import SportsDataAdapter
from nexus.sports.coach import CoachAssistant, CoachReport, PreMatchCoachReport
from nexus.sports.game_plan import GamePlan
from nexus.sports.ingest import ingest_player_metrics_as_signals
from nexus.sports.timeline import TIMELINE_SECTIONS
from nexus.sports.video import VideoAnalysisService

router = APIRouter()


def _to_game_plan_schema(plan: GamePlan | None) -> GamePlanSchema | None:
    if plan is None:
        return None
    return GamePlanSchema(
        opponent_strengths=[
            OpponentStrengthSchema(
                key=s.key,
                label=s.label,
                value=s.value,
                supporting_metrics=s.supporting_metrics,
                confidence=s.confidence,
                sample_size=s.sample_size,
                evidence=s.evidence,
            )
            for s in plan.opponent_strengths
        ],
        opponent_weaknesses=[
            OpponentWeaknessSchema(
                key=w.key,
                label=w.label,
                severity=w.severity,
                value=w.value,
                supporting_metrics=w.supporting_metrics,
                confidence=w.confidence,
                sample_size=w.sample_size,
                evidence=w.evidence,
                zone=w.zone,
            )
            for w in plan.opponent_weaknesses
        ],
        proposed_shape=(
            ProposedShapeSchema(
                shape=plan.proposed_shape.shape,
                reason=plan.proposed_shape.reason,
                supporting_metrics=plan.proposed_shape.supporting_metrics,
            )
            if plan.proposed_shape is not None
            else None
        ),
        principles=[
            PrincipleSchema(
                statement=p.statement,
                grounded_in=p.grounded_in,
                supporting_metrics=p.supporting_metrics,
            )
            for p in plan.principles
        ],
        adjustments=[
            AdjustmentSchema(
                instruction=a.instruction,
                targets_weakness=a.targets_weakness,
                rationale=a.rationale,
                supporting_metrics=a.supporting_metrics,
                priority=a.priority,
            )
            for a in plan.adjustments
        ],
        uncovered_weaknesses=plan.uncovered,
        unmeasured_metrics=sorted(set(plan.unmeasured)),
        is_partial=plan.is_partial,
        partial_reason=plan.partial_reason,
    )


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
        game_plan=_to_game_plan_schema(report.game_plan),
        timeline_sections=TimelineSectionsSchema(
            available=report.timeline.available_sections if report.timeline else [],
            missing=(
                report.timeline.missing_sections
                if report.timeline
                else list(TIMELINE_SECTIONS)
            ),
        ),
        is_partial=report.is_partial,
        narrative_warnings=report.narrative_warnings,
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
    coach: CoachAssistant = get_services(request).coach_assistant
    try:
        report = await coach.build_report(match_id)
    except ProviderUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _to_response(report)


@router.get("/sports/{match_id}/player/{player_id}", response_model=CoachReportResponse)
async def get_player_report(match_id: str, player_id: int, request: Request) -> CoachReportResponse:
    coach: CoachAssistant = get_services(request).coach_assistant
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
    """Narrates the player's latest pre-match readiness assessment.

    Every number in the response is computed by the football backend's
    deterministic scorer; the LLM only writes the `narrative` field.

    player_id is a string here, unlike the integer player_id on
    /sports/{match_id}/player/{player_id} above. That is not an
    inconsistency: the tactical route's id is a ByteTrack tracking ID scoped
    to one processed video, while a pre-match questionnaire is submitted
    before any tracking exists and carries the caller's own external
    identifier. Two different ID spaces, deliberately not conflated.

    404 when the player has not submitted a questionnaire -- distinct from
    the 503 a backend outage produces, so a caller can tell "nothing to
    report yet" from "we could not find out".
    """
    coach: CoachAssistant = get_services(request).coach_assistant
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
async def analyze_video(request: Request) -> CoachReportResponse:
    """Submits footage to the football backend's own processing endpoint,
    polls until it finishes, and narrates the result through the Group C
    adapter. No vision code runs in nexus/ — that stays in backend/ and
    ai/, reached over HTTP.

    Takes either a multipart upload (`file`, plus `match_id`/`player_id`
    form fields) or the JSON body. One route rather than two because both
    are the same operation from the caller's side: hand over a video, get
    back a CoachReport. Which one is in use is read off the content type.
    """
    service: VideoAnalysisService = get_services(request).video_analysis_service
    content_type = request.headers.get("content-type", "")

    try:
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            upload = form.get("file")
            if upload is None or not hasattr(upload, "read"):
                raise HTTPException(
                    status_code=422,
                    detail="multipart request must include a 'file' part holding the video",
                )
            raw_player_id = form.get("player_id")
            report = await service.analyze_upload(
                filename=getattr(upload, "filename", None) or "upload.mp4",
                content=await upload.read(),
                match_id=str(form.get("match_id") or ""),
                player_id=int(raw_player_id) if raw_player_id else None,
                content_type=getattr(upload, "content_type", None),
            )
        else:
            payload = VideoAnalyzeRequest.model_validate(await request.json())
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
    adapter: SportsDataAdapter = get_services(request).sports_adapter
    state_engine: PersonalStateEngine = get_services(request).personal_state_engine
    mapping: dict[str, str] = get_services(request).settings.sports.metric_dimension_map

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
