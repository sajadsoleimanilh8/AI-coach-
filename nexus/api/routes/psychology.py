"""
LLM Coach route for pre-match mental readiness.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from nexus.api.schemas import PsychologyCoachReportResponse
from nexus.core.exceptions import ProviderUnavailableError
from nexus.sports.coach import CoachAssistant, PsychologyCoachReport

router = APIRouter()


def _to_response(report: PsychologyCoachReport) -> PsychologyCoachReportResponse:
    assessment = report.assessment
    return PsychologyCoachReportResponse(
        player_id=report.player_id,
        match_id=report.match_id,
        mental_readiness=assessment.mental_readiness,
        focus=assessment.focus,
        confidence=assessment.confidence,
        stress=assessment.stress,
        pressure_risk=assessment.pressure_risk,
        mental_performance_risk=assessment.mental_performance_risk,
        key_positive_factors=report.findings.key_positive_factors,
        key_negative_factors=report.findings.key_negative_factors,
        neutral_factors=report.findings.neutral_factors,
        narrative=report.narrative,
        method=assessment.method,
        confidence_level=assessment.confidence_level,
        schema_version=assessment.schema_version,
        data_source=assessment.data_source,
        historical_context=assessment.historical_context,
        disclaimer=assessment.disclaimer,
        model_used=report.model_used,
    )


@router.get(
    "/sports/psychology/{player_id}/report",
    response_model=PsychologyCoachReportResponse,
)
async def get_psychology_report(
    player_id: str, request: Request, match_id: str | None = None
) -> PsychologyCoachReportResponse:
    """Narrates the player's latest mental-readiness assessment."""
    coach: CoachAssistant = request.app.state.coach_assistant
    try:
        report = await coach.build_psychology_report(player_id, match_id)
    except ProviderUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"No psychology assessment submitted yet for player_id={player_id}",
        )
    return _to_response(report)
