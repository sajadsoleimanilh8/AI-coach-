"""
LLM Coach route for pre-match mental readiness.

Registered alongside the other sports routes in nexus/api/main.py. Kept in its
own module rather than appended to routes/sports.py because it reads a
different endpoint family on the football backend, with its own response
contract -- the same reason nexus/sports/psychology_adapter.py is a separate
client from HttpSportsDataAdapter.

Every number in the response is computed by the football backend's
deterministic engine; the LLM writes only `narrative`.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from nexus.api.schemas import PsychologyCoachReportResponse
from nexus.api.services import get_services
from nexus.core.exceptions import ProviderUnavailableError
from nexus.sports.coach import CoachAssistant, PsychologyCoachReport

router = APIRouter()


def _to_response(report: PsychologyCoachReport) -> PsychologyCoachReportResponse:
    assessment = report.assessment
    return PsychologyCoachReportResponse(
        player_id=report.player_id,
        match_id=report.match_id,
        # Copied verbatim from the assessment -- nothing on this path
        # recomputes, rounds, or rescales a value.
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
    """Narrates the player's latest mental-readiness assessment.

    player_id is a string, like the pre-match health route and unlike the
    integer player_id on /sports/{match_id}/player/{player_id}. That is not an
    inconsistency: the tactical route's id is a ByteTrack tracking ID scoped to
    one processed video, while a questionnaire is submitted before any tracking
    exists and carries the caller's own external identifier. Two different ID
    spaces, deliberately not conflated.

    404 when the player has not submitted a questionnaire -- distinct from the
    503 a backend outage produces, so a caller can tell "nothing to report yet"
    from "we could not find out".
    """
    coach: CoachAssistant = get_services(request).coach_assistant
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
