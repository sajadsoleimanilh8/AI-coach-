"""Stage 7: per-team intelligence metrics.

Moved verbatim out of backend/pipeline/runner.py; run_pipeline() there is
the orchestrator that calls these in order.
"""


from __future__ import annotations

import logging
from collections import Counter, defaultdict

from ai.computer_vision.tactical_analysis.formation_detection import detect_formation
from ai.team_intelligence.formation_stability.team_shape import compute_compactness, compute_formation_stability
from ai.team_intelligence.pressing_structure_analysis.pressing import compute_pressing_intensity
from ai.team_intelligence.weak_zone_detection.weak_zones import compute_weak_zones
from backend.database.models import (
    Match,
)

logger = logging.getLogger(__name__)

def _dominant_team_id(points) -> str | None:
    """The team a track belongs to, as the modal non-null team_id over its
    own points. Per-detection assignment can flicker on a bad crop; the
    mode is stable against that without inventing an assignment for a track
    that genuinely never got one."""
    counts = Counter(str(p.team_id) for p in points if p.team_id is not None)
    return counts.most_common(1)[0][0] if counts else None


def _split_by_team(trajectories: dict) -> dict[str, dict]:
    """{team_id: {player_id: points}}. Tracks with no team assignment are
    dropped rather than pooled into either side -- an unassigned track is
    usually an official or a bad crop, and putting it in a team's shape
    would corrupt exactly the metric being fixed here."""
    out: dict[str, dict] = defaultdict(dict)
    for player_id, points in trajectories.items():
        tid = _dominant_team_id(points)
        if tid is not None:
            out[tid][player_id] = points
    return dict(out)


def _mean_positions_per_player(team_traj: dict) -> list[tuple[float, float]]:
    """One (x, y) per PLAYER -- their mean position over the clip.

    detect_formation() matches N points against 10 formation slots with the
    Hungarian algorithm, so it needs one point per player. The previous
    call passed `all_positions[:10]`, the first ten *trajectory points* of
    the pooled population: typically ten consecutive samples of one or two
    players in the opening frames, matched against a full-team template.
    """
    out = []
    for points in team_traj.values():
        xs = [(p.pitch_x_m, p.pitch_y_m) for p in points if p.pitch_x_m is not None]
        if xs:
            out.append((sum(v[0] for v in xs) / len(xs), sum(v[1] for v in xs) / len(xs)))
    return out


def _score_team_intelligence(match: Match, trajectories: dict, team_assignment_confidence: float,
                             directions=None) -> list[dict]:
    """Per-team team-level metrics.

    Each metric is computed once per team over that team's own tracks, never
    over every tracked player pooled: pooled "compactness" only measures how
    much of the pitch is in frame, and a mixed-team point cloud cannot match a
    formation. Every returned dict carries `team_id` so the rows are
    distinguishable in `team_metrics`.

    Weak zones additionally receive the OPPOSING team's positions, because
    a zone this team leaves empty is only a weakness if the opposition
    actually occupies it -- see compute_weak_zones()'s docstring.

    When team assignment produced nothing usable (no track has a team_id --
    the normal outcome when calibration is invalid), one "unassigned" row
    per metric is emitted through the same gated functions, so the API
    still receives an honest low-confidence answer rather than an empty
    list that the frontend would render as a missing panel.
    """
    by_team = _split_by_team(trajectories)

    if not by_team:
        # Route the empty population through the real functions so the
        # gates produce their own honest low-confidence dicts.
        empty: list[tuple[float, float]] = []
        rows = [
            detect_formation(player_positions=empty,
                             team_assignment_confidence=team_assignment_confidence),
            compute_compactness(empty, team_assignment_confidence=team_assignment_confidence),
            compute_formation_stability([], team_assignment_confidence=team_assignment_confidence),
            compute_weak_zones(empty, team_assignment_confidence=team_assignment_confidence),
            compute_pressing_intensity([], team_assignment_confidence=team_assignment_confidence),
        ]
        for r in rows:
            r["team_id"] = "unassigned"
            r.setdefault("sub_scores", {})["reason_no_team_split"] = (
                "no tracked player carried a team_id; team assignment "
                "and/or calibration did not produce a usable split"
            )
        return rows

    metrics: list[dict] = []
    team_ids = sorted(by_team)
    for tid in team_ids:
        own = by_team[tid]
        own_positions = [(p.pitch_x_m, p.pitch_y_m)
                         for points in own.values() for p in points
                         if p.pitch_x_m is not None]
        opp_positions = [(p.pitch_x_m, p.pitch_y_m)
                         for other in team_ids if other != tid
                         for points in by_team[other].values() for p in points
                         if p.pitch_x_m is not None]

        direction = directions.for_team(tid) if directions is not None else "left_to_right"
        formation_kwargs = {}
        if direction in ("left_to_right", "right_to_left"):
            formation_kwargs["attacking_direction"] = direction

        rows = [
            detect_formation(
                player_positions=_mean_positions_per_player(own),
                team_assignment_confidence=team_assignment_confidence,
                **formation_kwargs,
            ),
            compute_compactness(own_positions,
                                team_assignment_confidence=team_assignment_confidence),
            compute_formation_stability(_positions_by_frame(own),
                                        team_assignment_confidence=team_assignment_confidence),
            compute_weak_zones(own_positions,
                               team_assignment_confidence=team_assignment_confidence,
                               opponent_positions_m=opp_positions or None),
            # Pressing still needs per-frame defender-to-ball distances
            # scoped to the pressing team. Possession is not yet attributed
            # to a team over time, so this stays an honest empty input and
            # the function's own low_sample gate answers for it.
            compute_pressing_intensity([],
                                       team_assignment_confidence=team_assignment_confidence),
        ]
        for r in rows:
            r["team_id"] = tid
            sub = r.setdefault("sub_scores", {})
            sub["n_players_in_team"] = len(own)
            sub["team_assignment_confidence"] = team_assignment_confidence
            if directions is not None:
                sub["attacking_direction"] = direction
                if direction == "unknown":
                    sub["attacking_direction_reason"] = directions.reason
        metrics.extend(rows)

    if len(team_ids) == 1:
        for r in metrics:
            r.setdefault("sub_scores", {})["single_team_only"] = (
                "only one team_id was present, so opponent-relative context "
                "(weak-zone exposure) could not be computed"
            )
    return metrics


def _positions_by_frame(trajectories: dict) -> list[list[tuple]]:
    """Per-frame position lists, ordered by real frame_id.

    Ordered by real frame_id, not by point-list index (same reason as
    _build_players_by_frame()). compute_formation_stability() measures how much
    a shape moves BETWEEN consecutive entries, so misaligned frames would
    manufacture apparent movement out of an occlusion gap and report the team's
    shape as less stable than it is.
    """
    by_frame: dict[int, list[tuple]] = {}
    for points in trajectories.values():
        for p in points:
            if p.pitch_x_m is not None:
                by_frame.setdefault(p.frame_id, []).append((p.pitch_x_m, p.pitch_y_m))
    return [by_frame[fid] for fid in sorted(by_frame)]
