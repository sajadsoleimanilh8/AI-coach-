"""Match-scoped, transparent what-if simulation API.

This is deterministic recomputation over persisted tracking data, not
reinforcement learning and not an outcome predictor.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ai.computer_vision.tactical_analysis.attacking_direction import infer_attacking_directions
from ai.simulation_ai.what_if_analysis.engine import Intervention, SimulationError, simulate
from backend.api.schemas import SimulationRequest, SimulationResponse
from backend.database.models import Match, PlayerTracking, TeamMetric
from backend.database.session import get_db

router = APIRouter(prefix="/api/simulation", tags=["simulation"])


def _split_persisted_tracks(rows: list[PlayerTracking]) -> dict[str, dict[int, list[PlayerTracking]]]:
    by_player: dict[int, list[PlayerTracking]] = defaultdict(list)
    for row in rows:
        by_player[row.player_id].append(row)

    teams: dict[str, dict[int, list[PlayerTracking]]] = defaultdict(dict)
    for player_id, points in by_player.items():
        counts = Counter(str(p.team_id) for p in points if p.team_id is not None)
        if counts:
            teams[counts.most_common(1)[0][0]][player_id] = points
    return dict(teams)


@router.post("/{match_id}", response_model=SimulationResponse)
def run_simulation(match_id: str, payload: SimulationRequest, db: Session = Depends(get_db)):
    if db.get(Match, match_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Match not found")

    tracking = (
        db.query(PlayerTracking)
        .filter(PlayerTracking.match_id == match_id)
        .order_by(PlayerTracking.frame_id)
        .all()
    )
    team_metrics = db.query(TeamMetric).filter(TeamMetric.match_id == match_id).all()
    teams = _split_persisted_tracks(tracking)

    provenance: dict[str, dict[str, dict]] = defaultdict(dict)
    real_inputs = []
    assignment_confidences = []
    for metric in team_metrics:
        provenance[str(metric.team_id)][metric.metric_name] = {
            "metric_id": metric.metric_id,
            "method": metric.method.value,
            "confidence": metric.confidence.value,
        }
        real_inputs.append({
            "metric_id": metric.metric_id,
            "team_id": metric.team_id,
            "metric_name": metric.metric_name,
            "value": metric.value,
            "method": metric.method.value,
            "confidence": metric.confidence.value,
            "sample_size": metric.sample_size,
        })
        raw_assignment = (metric.sub_scores or {}).get("team_assignment_confidence")
        if isinstance(raw_assignment, (int, float)):
            assignment_confidences.append(float(raw_assignment))

    # Older runs did not persist this upstream confidence. Zero deliberately
    # activates the scorers' low-upstream-confidence gates instead of guessing.
    assignment_confidence = (
        min(assignment_confidences) if assignment_confidences else 0.0
    )
    interventions = [Intervention(**item.dict()) for item in payload.interventions]
    try:
        result = simulate(
            teams,
            interventions,
            team_assignment_confidence=assignment_confidence,
            directions=infer_attacking_directions({pid: pts for team in teams.values() for pid, pts in team.items()}),
            baseline_provenance=dict(provenance),
        ).as_dict()
    except SimulationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    result["match_id"] = match_id
    result["real_input_metrics"] = real_inputs
    result["team_assignment_confidence"] = (
        assignment_confidence if assignment_confidences else None
    )
    if not assignment_confidences:
        result["unavailable"].append(
            "team-assignment confidence was not persisted for this run; simulation "
            "outputs remain gated rather than assuming a trustworthy team split"
        )
    possession_inputs = [m for m in real_inputs if "possession" in m["metric_name"]]
    if not possession_inputs:
        result["unavailable"].append(
            "team possession is not persisted as a TeamMetric, so possession cannot be simulated"
        )
    return result
