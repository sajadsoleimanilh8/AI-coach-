from __future__ import annotations

from fastapi import APIRouter, Request

from nexus.api.schemas import HealthAnalysisResponse, HealthPatternSchema, HealthScorecardResponse
from nexus.health.analyzer import HealthAnalysis, HealthAnalyzer
from nexus.health.safety import MEDICAL_DISCLAIMER

router = APIRouter()


def _to_response(analysis: HealthAnalysis) -> HealthAnalysisResponse:
    return HealthAnalysisResponse(
        user_id=analysis.user_id,
        patterns=[
            HealthPatternSchema(
                name=p.name,
                dimensions_involved=p.dimensions_involved,
                severity=p.severity,
                confidence=p.confidence,
                sample_size=p.sample_size,
                explanation=p.explanation,
            )
            for p in analysis.patterns
        ],
        scorecard=analysis.scorecard,
        data_sufficiency=analysis.data_sufficiency,
        computed_at=analysis.computed_at,
        disclaimer=MEDICAL_DISCLAIMER,
    )


@router.get("/health-intel/{user_id}/analysis", response_model=HealthAnalysisResponse)
async def get_health_analysis(user_id: str, request: Request) -> HealthAnalysisResponse:
    analyzer: HealthAnalyzer = request.app.state.health_analyzer
    return _to_response(await analyzer.analyze(user_id))


@router.get("/health-intel/{user_id}/scorecard", response_model=HealthScorecardResponse)
async def get_health_scorecard(user_id: str, request: Request) -> HealthScorecardResponse:
    analyzer: HealthAnalyzer = request.app.state.health_analyzer
    analysis = await analyzer.analyze(user_id)
    return HealthScorecardResponse(
        user_id=analysis.user_id,
        scorecard=analysis.scorecard,
        data_sufficiency=analysis.data_sufficiency,
        disclaimer=MEDICAL_DISCLAIMER,
    )
