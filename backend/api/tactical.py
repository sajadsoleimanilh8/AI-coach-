"""
Tactical Intelligence API router.
Implementation Spec §5.1.
"""

from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.api.cache import get_cached_metrics, set_cached_metrics
from backend.api.schemas import TeamMetricResponse
from backend.database.models import TeamMetric
from backend.database.session import get_db

router = APIRouter(prefix="/api/tactical", tags=["tactical"])
team_intel_router = APIRouter(prefix="/api/team_intelligence", tags=["team_intelligence"])

DEFAULT_TEAM_SCOPE = "unassigned"


@router.get("/formation/{match_id}", response_model=TeamMetricResponse)
def get_formation(
    match_id: str,
    team_id: str = Query(default=DEFAULT_TEAM_SCOPE),
    db: Session = Depends(get_db),
):
    scope = f"formation:{team_id}"
    cached = get_cached_metrics(match_id, scope)
    if cached:
        return cached

    metric = (
        db.query(TeamMetric)
        .filter(
            TeamMetric.match_id == match_id,
            TeamMetric.team_id == team_id,
            TeamMetric.metric_name == "formation",
        )
        .first()
    )

    if not metric:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No formation metric computed yet for match_id={match_id}, team_id={team_id}. "
                   f"Upload and process a video for this match first.",
        )

    res = {
        "metric_id": metric.metric_id,
        "match_id": metric.match_id,
        "team_id": metric.team_id,
        "metric_name": metric.metric_name,
        "value": metric.value,
        "method": metric.method.value,
        "confidence": metric.confidence.value,
        "confidence_score": metric.confidence_score,
        "sample_size": metric.sample_size,
        "sub_scores": metric.sub_scores,
        "computed_at": metric.computed_at,
        "schema_version": metric.schema_version,
    }

    set_cached_metrics(match_id, scope, res)
    return res


@router.get("/team_shape/{match_id}", response_model=list[TeamMetricResponse])
def get_team_shape(
    match_id: str,
    team_id: str = Query(default=DEFAULT_TEAM_SCOPE),
    db: Session = Depends(get_db),
):
    scope = f"team_shape:{team_id}"
    cached = get_cached_metrics(match_id, scope)
    if cached:
        return cached

    shape_names = ["compactness_score", "formation_stability_score", "pressing_intensity_score", "weak_zone_map"]
    metrics = (
        db.query(TeamMetric)
        .filter(
            TeamMetric.match_id == match_id,
            TeamMetric.team_id == team_id,
            TeamMetric.metric_name.in_(shape_names),
        )
        .all()
    )

    results = [
        {
            "metric_id": m.metric_id,
            "match_id": m.match_id,
            "team_id": m.team_id,
            "metric_name": m.metric_name,
            "value": m.value,
            "method": m.method.value,
            "confidence": m.confidence.value,
            "confidence_score": m.confidence_score,
            "sample_size": m.sample_size,
            "sub_scores": m.sub_scores,
            "computed_at": m.computed_at,
            "schema_version": m.schema_version,
        }
        for m in metrics
    ]

    set_cached_metrics(match_id, scope, results)
    return results


@team_intel_router.get("/{match_id}", response_model=list[TeamMetricResponse])
def get_team_intelligence(match_id: str, db: Session = Depends(get_db)):
    scope = "all_team_metrics"
    cached = get_cached_metrics(match_id, scope)
    if cached:
        return cached

    metrics = db.query(TeamMetric).filter(TeamMetric.match_id == match_id).all()
    results = [
        {
            "metric_id": m.metric_id,
            "match_id": m.match_id,
            "team_id": m.team_id,
            "metric_name": m.metric_name,
            "value": m.value,
            "method": m.method.value,
            "confidence": m.confidence.value,
            "confidence_score": m.confidence_score,
            "sample_size": m.sample_size,
            "sub_scores": m.sub_scores,
            "computed_at": m.computed_at,
            "schema_version": m.schema_version,
        }
        for m in metrics
    ]

    set_cached_metrics(match_id, scope, results)
    return results
