"""
Pre-Match Psychology Intelligence API router.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from ai.psychology_ai.constants import (
    DATA_SOURCE,
    FACTOR_NEGATIVE,
    FACTOR_POSITIVE,
)
from ai.psychology_ai.feature_extraction import (
    FEATURE_DIRECTIONS,
    QuestionnaireValidationError,
)
from ai.psychology_ai.mental_readiness.score import assess_psychology
from ai.psychology_ai.model_interface import (
    DEFAULT_MODEL,
    PsychologyReadinessModel,
    factor_phrase,
)
from backend.api.schemas import (
    PsychologyAssessmentResponse,
    PsychologyFeaturesResponse,
    PsychologyQuestionnaireRequest,
)
from backend.database.models import (
    Match,
    MetricConfidence,
    MetricMethod,
    PlayerMetric,
    PsychologyAssessment,
)
from backend.database.session import get_db

router = APIRouter(prefix="/api/psychology", tags=["psychology"])

DISCLAIMER = (
    "Mental-readiness estimate derived from a self-reported questionnaire. "
    "Not emotion detection, not a psychological or clinical assessment, and "
    "not a diagnosis."
)

_PROXY_SOURCE_METRICS: dict[str, str] = {
    "pressure_response_proxy": "press_resistance_score",
    "focus_proxy": "scanning_behavior_score",
    "confidence_proxy": "decision_making_score",
}


def get_readiness_model() -> PsychologyReadinessModel:
    """The composition point. Swapping in a trained model once one genuinely
    exists is a change here and nowhere else -- and it can be overridden
    per-request in tests via FastAPI's dependency_overrides."""
    return DEFAULT_MODEL


def _load_historical_proxies(
    db: Session, match_id: str | None, cv_player_id: int | None
) -> dict | None:
    """CV-derived corroboration for one (match_id, cv_player_id), or None."""
    if match_id is None or cv_player_id is None:
        return None

    rows = (
        db.query(PlayerMetric)
        .filter(
            PlayerMetric.match_id == match_id,
            PlayerMetric.player_id == cv_player_id,
        )
        .all()
    )
    if not rows:
        return None

    by_name = {row.metric_name: row for row in rows}
    proxies: dict = {}

    for proxy_name, metric_name in _PROXY_SOURCE_METRICS.items():
        row = by_name.get(metric_name)
        if row is None:
            continue
        proxies[proxy_name] = {
            "value": row.value,
            "confidence": row.confidence.value,
            "sample_size": row.sample_size,
            "source_metric": metric_name,
        }

    usable = [
        row.value
        for row in rows
        if row.value is not None and row.confidence != MetricConfidence.low_upstream_confidence
    ]
    if len(usable) >= 2:
        mean_value = sum(usable) / len(usable)
        variance = sum((value - mean_value) ** 2 for value in usable) / len(usable)
        spread = variance**0.5
        consistency = max(0.0, 100.0 - 2.0 * spread)
        proxies["performance_consistency"] = {
            "value": round(consistency, 2),
            "confidence": MetricConfidence.normal.value,
            "sample_size": len(usable),
            "source_metric": "within_match_spread_across_metrics",
        }
    else:
        proxies["performance_consistency"] = {
            "value": None,
            "confidence": MetricConfidence.low_sample.value,
            "sample_size": len(usable),
            "source_metric": "within_match_spread_across_metrics",
        }

    return proxies or None


def _to_response(row: PsychologyAssessment) -> PsychologyAssessmentResponse:
    factors: dict[str, str] = dict(row.factors or {})
    return PsychologyAssessmentResponse(
        player_id=row.player_id,
        match_id=row.match_id,
        cv_player_id=row.cv_player_id,
        mental_readiness=int(row.mental_readiness),
        focus=int(row.focus),
        confidence=int(row.confidence),
        stress=int(row.stress),
        pressure_risk=row.pressure_risk,
        mental_performance_risk=row.mental_performance_risk,
        factors=factors,
        key_positive_factors=[
            factor_phrase(dimension, label)
            for dimension, label in factors.items()
            if label == FACTOR_POSITIVE
        ],
        key_negative_factors=[
            factor_phrase(dimension, label)
            for dimension, label in factors.items()
            if label == FACTOR_NEGATIVE
        ],
        method=row.method.value,
        confidence_level=row.confidence_level.value,
        sample_size=(row.sub_scores or {}).get("sample_size", 0),
        schema_version=row.schema_version,
        submitted_at=row.submitted_at,
        computed_at=row.computed_at,
        assessment_id=row.assessment_id,
        submission_index=row.submission_index,
        data_source=DATA_SOURCE,
        historical_context=(row.sub_scores or {}).get("historical_context", []),
        disclaimer=DISCLAIMER,
    )


def _get_assessment_or_404(
    db: Session, player_id: str, assessment_id: str
) -> PsychologyAssessment:
    """Scoped by player_id as well as id: an assessment id belonging to a
    different player must not be readable through this player's path."""
    row = (
        db.query(PsychologyAssessment)
        .filter(
            PsychologyAssessment.assessment_id == assessment_id,
            PsychologyAssessment.player_id == player_id,
        )
        .first()
    )
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No psychology assessment {assessment_id} for player_id={player_id}",
        )
    return row


@router.post(
    "/{player_id}/submit",
    response_model=PsychologyAssessmentResponse,
    status_code=status.HTTP_201_CREATED,
)
def submit_questionnaire(
    player_id: str,
    payload: PsychologyQuestionnaireRequest,
    db: Session = Depends(get_db),
    model: PsychologyReadinessModel = Depends(get_readiness_model),
):
    """Validated questionnaire -> features -> assessment -> persisted."""
    if payload.match_id is not None:
        if not db.get(Match, payload.match_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Match not found: {payload.match_id}. Omit match_id if the "
                f"match has not been created yet.",
            )

    responses = payload.model_dump(exclude={"match_id", "cv_player_id"})
    historical = _load_historical_proxies(db, payload.match_id, payload.cv_player_id)

    try:
        assessment = assess_psychology(responses, model=model, historical=historical)
    except QuestionnaireValidationError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error

    if assessment["mental_readiness"] is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Questionnaire could not be scored: "
            f"{assessment['sub_scores'].get('missing_features')}",
        )

    previous_index = (
        db.query(func.coalesce(func.max(PsychologyAssessment.submission_index), 0))
        .filter(PsychologyAssessment.player_id == player_id)
        .scalar()
    ) or 0

    row = PsychologyAssessment(
        player_id=player_id,
        cv_player_id=payload.cv_player_id,
        match_id=payload.match_id,
        submission_index=previous_index + 1,
        responses=responses,
        features=assessment["features"],
        mental_readiness=float(assessment["mental_readiness"]),
        focus=float(assessment["focus"]),
        confidence=float(assessment["confidence"]),
        stress=float(assessment["stress"]),
        pressure_risk=assessment["pressure_risk"],
        mental_performance_risk=assessment["mental_performance_risk"],
        factors=assessment["factors"],
        sub_scores={
            **assessment["sub_scores"],
            "sample_size": assessment["sample_size"],
            "historical_context": assessment["historical_context"],
            "component_metrics": assessment["component_metrics"],
        },
        method=MetricMethod(assessment["method"]),
        confidence_level=MetricConfidence(assessment["confidence_level"]),
        schema_version=assessment["schema_version"],
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return _to_response(row)




@router.get("/{player_id}/latest", response_model=PsychologyAssessmentResponse)
def get_latest_assessment(
    player_id: str,
    match_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """Most recent assessment for this player, optionally filtered to a match."""
    query = db.query(PsychologyAssessment).filter(
        PsychologyAssessment.player_id == player_id
    )
    if match_id is not None:
        query = query.filter(PsychologyAssessment.match_id == match_id)

    row = query.order_by(PsychologyAssessment.submission_index.desc()).first()
    if not row:
        detail = f"No psychology assessment submitted yet for player_id={player_id}"
        if match_id is not None:
            detail += f" and match_id={match_id}"
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)
    return _to_response(row)


@router.get("/{player_id}/history", response_model=list[PsychologyAssessmentResponse])
def get_assessment_history(
    player_id: str,
    match_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Past assessments, most recent first, for trend review across matches."""
    query = db.query(PsychologyAssessment).filter(
        PsychologyAssessment.player_id == player_id
    )
    if match_id is not None:
        query = query.filter(PsychologyAssessment.match_id == match_id)

    rows = (
        query.order_by(PsychologyAssessment.submission_index.desc()).limit(limit).all()
    )
    return [_to_response(row) for row in rows]


@router.get(
    "/{player_id}/{assessment_id}/features", response_model=PsychologyFeaturesResponse
)
def get_assessment_features(
    player_id: str, assessment_id: str, db: Session = Depends(get_db)
):
    """The normalized feature vector alone -- the shape a future ML/DL model
    would train on and predict from."""
    row = _get_assessment_or_404(db, player_id, assessment_id)
    return PsychologyFeaturesResponse(
        assessment_id=row.assessment_id,
        player_id=row.player_id,
        match_id=row.match_id,
        features=dict(row.features or {}),
        feature_directions=FEATURE_DIRECTIONS,
        method=row.method.value,
        schema_version=row.schema_version,
        computed_at=row.computed_at,
    )


@router.get("/{player_id}/{assessment_id}", response_model=PsychologyAssessmentResponse)
def get_assessment(player_id: str, assessment_id: str, db: Session = Depends(get_db)):
    """The full structured output for one assessment -- the same shape /latest
    returns, and exactly what the LLM Coach is handed as context."""
    return _to_response(_get_assessment_or_404(db, player_id, assessment_id))
