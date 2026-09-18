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

# Which team a request is about.
#
# The pipeline writes team-level metrics under "team-home"/"team-away" when
# jersey-colour team assignment produces a split, and under "unassigned" only
# when it does not (see backend/pipeline/team_scoring.py). These routes used to
# default team_id to "unassigned", so a request that did not name a team found
# nothing for every match where the split worked -- which is the normal case.
#
# Now team_id is optional. Named: exactly that team. Omitted: the teams that
# actually exist for this match. /formation returns one metric, so it takes
# the first of TEAM_PREFERENCE that has a row (the response's team_id says
# which); /team_shape returns a list, so it returns every team's rows.
TEAM_PREFERENCE = ("team-home", "team-away", "unassigned")

TEAM_ID_QUERY = Query(
    default=None,
    description="team-home, team-away or unassigned. Omit to use the teams this match actually has.",
)

SHAPE_METRICS = ["compactness_score", "formation_stability_score", "pressing_intensity_score", "weak_zone_map"]


def _serialize(metric: TeamMetric) -> dict:
    return {
        "metric_id": metric.metric_id,
        "match_id": metric.match_id,
        "team_id": metric.team_id,
        "metric_name": metric.metric_name,
        "value": metric.value,  # TeamMetric.value property resolves value_numeric/value_label
        "method": metric.method.value,
        "confidence": metric.confidence.value,
        "confidence_score": metric.confidence_score,
        "sample_size": metric.sample_size,
        "sub_scores": metric.sub_scores,
        "computed_at": metric.computed_at,
        "schema_version": metric.schema_version,
    }


def _team_rank(team_id: str | None) -> tuple[int, str]:
    """Order teams by TEAM_PREFERENCE, anything unexpected last (alphabetically)."""
    if team_id in TEAM_PREFERENCE:
        return TEAM_PREFERENCE.index(team_id), ""
    return len(TEAM_PREFERENCE), team_id or ""


@router.get("/formation/{match_id}", response_model=TeamMetricResponse)
def get_formation(
    match_id: str,
    team_id: str | None = TEAM_ID_QUERY,
    db: Session = Depends(get_db),
):
    scope = f"formation:{team_id or 'auto'}"
    cached = get_cached_metrics(match_id, scope)
    if cached:
        return cached

    query = db.query(TeamMetric).filter(
        TeamMetric.match_id == match_id,
        TeamMetric.metric_name == "formation",
    )
    if team_id is not None:
        query = query.filter(TeamMetric.team_id == team_id)
    candidates = sorted(query.all(), key=lambda m: _team_rank(m.team_id))

    if not candidates:
        # No fallback value. A made-up formation label next to a
        # "low_upstream_confidence" badge is more misleading than an honest
        # "not computed yet": it looks like a measurement when none exists.
        # So a missing row is a 404, never a placeholder formation.
        which = f", team_id={team_id}" if team_id else ""
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No formation metric computed yet for match_id={match_id}{which}. "
                   f"Upload and process a video for this match first.",
        )

    res = _serialize(candidates[0])
    set_cached_metrics(match_id, scope, res)
    return res


@router.get("/team_shape/{match_id}", response_model=list[TeamMetricResponse])
def get_team_shape(
    match_id: str,
    team_id: str | None = TEAM_ID_QUERY,
    db: Session = Depends(get_db),
):
    scope = f"team_shape:{team_id or 'auto'}"
    cached = get_cached_metrics(match_id, scope)
    if cached:
        return cached

    query = db.query(TeamMetric).filter(
        TeamMetric.match_id == match_id,
        TeamMetric.metric_name.in_(SHAPE_METRICS),
    )
    if team_id is not None:
        query = query.filter(TeamMetric.team_id == team_id)
    metrics = sorted(query.all(), key=lambda m: (_team_rank(m.team_id), m.metric_name))

    results = [_serialize(m) for m in metrics]
    set_cached_metrics(match_id, scope, results)
    return results


@team_intel_router.get("/{match_id}", response_model=list[TeamMetricResponse])
def get_team_intelligence(match_id: str, db: Session = Depends(get_db)):
    scope = "all_team_metrics"
    cached = get_cached_metrics(match_id, scope)
    if cached:
        return cached

    metrics = db.query(TeamMetric).filter(TeamMetric.match_id == match_id).all()
    results = [_serialize(m) for m in metrics]

    set_cached_metrics(match_id, scope, results)
    return results
