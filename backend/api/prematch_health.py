"""
Pre-Match Health Intelligence API router.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from ai.performance_ai.match_readiness_predictor.features import FEATURE_DIRECTIONS
from ai.performance_ai.match_readiness_predictor.questionnaire import (
    PreMatchQuestionnaireInput,
    QuestionnaireValidationError,
)
from ai.performance_ai.match_readiness_predictor.score import (
    DEFAULT_SCORER,
    score_match_readiness,
)
from ai.performance_ai.match_readiness_predictor.scorer import ReadinessScorer
from backend.api.schemas import (
    PreMatchAssessmentResponse,
    PreMatchFactorResponse,
    PreMatchFeaturesResponse,
    PreMatchQuestionnaireRequest,
)
from backend.database.models import (
    Match,
    MetricMethod,
    PreMatchHealthAssessment,
    PreMatchQuestionnaire,
    RiskLevel,
)
from backend.database.session import get_db

router = APIRouter(prefix="/api/prematch_health", tags=["prematch_health"])

DISCLAIMER = (
    "Performance-readiness estimate derived from a self-reported "
    "questionnaire. Not a medical assessment, diagnosis, or injury "
    "prediction."
)


def get_readiness_scorer() -> ReadinessScorer:
    """The composition point. Swapping in an MLReadinessScorer once one
    genuinely exists is a change here and nowhere else -- and it can be
    overridden per-request in tests via FastAPI's dependency_overrides."""
    return DEFAULT_SCORER


def _to_assessment_response(row: PreMatchHealthAssessment) -> PreMatchAssessmentResponse:
    factors = [PreMatchFactorResponse(**factor) for factor in (row.factors or [])]
    return PreMatchAssessmentResponse(
        player_id=row.player_id,
        match_id=row.match_id,
        physical_readiness=row.physical_readiness,
        fatigue_score=row.fatigue_score,
        recovery_score=row.recovery_score,
        performance_risk=row.performance_risk.value,
        workload_risk=row.workload_risk.value,
        key_positive_factors=[f.detail for f in factors if f.label == "positive"],
        key_negative_factors=[f.detail for f in factors if f.label == "negative"],
        method=row.method.value,
        schema_version=row.schema_version,
        computed_at=row.computed_at,
        assessment_id=row.id,
        questionnaire_id=row.questionnaire_id,
        submission_index=row.submission_index,
        factors=factors,
        data_source=(row.features or {}).get("data_source", "self_reported"),
        notes=(row.questionnaire.questionnaire_json or {}).get(
            "caffeine_or_supplement_notes"
        )
        if row.questionnaire
        else None,
        disclaimer=DISCLAIMER,
    )


def _get_assessment_or_404(
    db: Session, player_id: str, assessment_id: str
) -> PreMatchHealthAssessment:
    """Scoped by player_id as well as id: an assessment id belonging to a
    different player must not be readable through this player's path."""
    row = (
        db.query(PreMatchHealthAssessment)
        .filter(
            PreMatchHealthAssessment.id == assessment_id,
            PreMatchHealthAssessment.player_id == player_id,
        )
        .first()
    )
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No assessment {assessment_id} for player_id={player_id}",
        )
    return row


@router.post(
    "/{player_id}/submit",
    response_model=PreMatchAssessmentResponse,
    status_code=status.HTTP_201_CREATED,
)
def submit_questionnaire(
    player_id: str,
    payload: PreMatchQuestionnaireRequest,
    db: Session = Depends(get_db),
    scorer: ReadinessScorer = Depends(get_readiness_scorer),
):
    """Validated questionnaire -> features -> assessment -> both rows persisted."""
    if payload.match_id is not None:
        if not db.get(Match, payload.match_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Match not found: {payload.match_id}. Omit match_id if the "
                f"match has not been created yet.",
            )

    try:
        questionnaire_input = PreMatchQuestionnaireInput(
            player_id=player_id,
            **payload.model_dump(),
        )
    except QuestionnaireValidationError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error

    assessment = score_match_readiness(questionnaire_input, scorer=scorer)

    questionnaire_row = PreMatchQuestionnaire(
        player_id=player_id,
        match_id=payload.match_id,
        questionnaire_json=questionnaire_input.to_storable_dict(),
    )
    db.add(questionnaire_row)
    db.flush()

    features = assessment.to_feature_vector()
    features["data_source"] = assessment.data_source

    previous_index = (
        db.query(func.coalesce(func.max(PreMatchHealthAssessment.submission_index), 0))
        .filter(PreMatchHealthAssessment.player_id == player_id)
        .scalar()
    ) or 0

    assessment_row = PreMatchHealthAssessment(
        questionnaire_id=questionnaire_row.id,
        player_id=player_id,
        submission_index=previous_index + 1,
        match_id=payload.match_id,
        features=features,
        physical_readiness=assessment.physical_readiness,
        fatigue_score=assessment.fatigue_score,
        recovery_score=assessment.recovery_score,
        performance_risk=RiskLevel(assessment.performance_risk),
        workload_risk=RiskLevel(assessment.workload_risk),
        factors=[
            {
                "dimension": f.dimension,
                "label": f.label,
                "score": f.score,
                "detail": f.detail,
            }
            for f in assessment.factors
        ],
        method=MetricMethod(assessment.method),
        schema_version=assessment.schema_version,
    )
    db.add(assessment_row)
    db.commit()
    db.refresh(assessment_row)

    return _to_assessment_response(assessment_row)


@router.get("/{player_id}/latest", response_model=PreMatchAssessmentResponse)
def get_latest_assessment(player_id: str, db: Session = Depends(get_db)):
    """Most recent assessment for this player."""
    row = (
        db.query(PreMatchHealthAssessment)
        .filter(PreMatchHealthAssessment.player_id == player_id)
        .order_by(PreMatchHealthAssessment.submission_index.desc())
        .first()
    )
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No pre-match assessment submitted yet for player_id={player_id}",
        )
    return _to_assessment_response(row)


@router.get("/{player_id}/history", response_model=list[PreMatchAssessmentResponse])
def get_assessment_history(
    player_id: str,
    limit: int = Query(default=20, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Chronological list, newest first, for trend review across matches."""
    rows = (
        db.query(PreMatchHealthAssessment)
        .filter(PreMatchHealthAssessment.player_id == player_id)
        .order_by(PreMatchHealthAssessment.submission_index.desc())
        .limit(limit)
        .all()
    )
    return [_to_assessment_response(row) for row in rows]


@router.get(
    "/{player_id}/{assessment_id}/features", response_model=PreMatchFeaturesResponse
)
def get_assessment_features(
    player_id: str, assessment_id: str, db: Session = Depends(get_db)
):
    """The normalized feature vector alone -- the shape a future ML/DL model
    would train on and predict from."""
    row = _get_assessment_or_404(db, player_id, assessment_id)
    features = dict(row.features or {})
    features.pop("data_source", None)
    return PreMatchFeaturesResponse(
        assessment_id=row.id,
        player_id=row.player_id,
        match_id=row.match_id,
        features=features,
        feature_directions=FEATURE_DIRECTIONS,
        method=row.method.value,
        schema_version=row.schema_version,
        computed_at=row.computed_at,
    )


@router.get(
    "/{player_id}/{assessment_id}/assessment", response_model=PreMatchAssessmentResponse
)
def get_assessment(player_id: str, assessment_id: str, db: Session = Depends(get_db)):
    """The full structured output for one assessment -- the same shape
    /latest returns, and exactly what the LLM Coach is handed as context."""
    return _to_assessment_response(_get_assessment_or_404(db, player_id, assessment_id))
