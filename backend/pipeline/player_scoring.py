"""Stage 8: per-player intelligence metrics and the inputs derived for them.

Moved verbatim out of backend/pipeline/runner.py; run_pipeline() there is
the orchestrator that calls these in order.
"""


from __future__ import annotations

import logging
import math

from ai.computer_vision.tactical_analysis.attacking_direction import (
    UNKNOWN as DIRECTION_UNKNOWN,
)
from ai.computer_vision.tactical_analysis.constants import (
    PASS_DIRECTION_SECTORS,
)
from ai.player_intelligence.body_orientation_score.score import score_body_orientation
from ai.player_intelligence.decision_making_score.score import score_decision_making
from ai.player_intelligence.defensive_positioning.score import score_defensive_positioning
from ai.player_intelligence.finishing_efficiency_score.score import score_finishing_efficiency
from ai.player_intelligence.first_touch_score.score import score_first_touch
from ai.player_intelligence.off_ball_movement.score import score_off_ball_movement
from ai.player_intelligence.passing_vision_score.score import score_passing_vision
from ai.player_intelligence.press_resistance_score.score import score_press_resistance
from ai.player_intelligence.scanning_behavior.score import score_scanning_behavior
from backend.database.models import (
    Event,
    Match,
)
from backend.pipeline.team_scoring import _dominant_team_id

logger = logging.getLogger(__name__)

def _build_players_by_frame(trajectories: dict) -> dict[int, list[dict]]:
    """
    {real frame_id: [{player_id, team_id, pitch_x_m, pitch_y_m}, ...]}.

    Keyed by real frame_id, not by each player's point-list INDEX. An index is
    exact only while every tracked player has one TrackingPoint per frame with
    no gaps. ByteTrack drops a track through an occlusion and re-acquires it
    later, so from the first gap onward one player's index i and another's
    index i refer to different frames -- and every consumer of this structure
    (proximity, off-ball movement, possession context) would compare players
    who were never on the pitch at the same moment.

    `TrackingPoint.frame_id` is the real frame number and was already
    populated by enrich_with_pitch_coordinates(); keying on it makes
    "same key" mean "same frame" unconditionally. Frames where a given
    player has no point simply do not list that player, which is the
    honest representation of an occlusion.
    """
    players_by_frame: dict[int, list[dict]] = {}
    for _player_id, points in trajectories.items():
        for p in points:
            players_by_frame.setdefault(p.frame_id, []).append({
                "player_id": p.player_id,
                "team_id": p.team_id,
                "pitch_x_m": p.pitch_x_m,
                "pitch_y_m": p.pitch_y_m,
            })
    return players_by_frame


def _nearest_other_player_distance(player_id: int, x: float, y: float, frame_others: list[dict]) -> float | None:
    """Distance to the closest OTHER tracked player at this frame. Same
    documented nearest-other-player-as-opponent-stand-in approximation as
    _enrich_first_touch_metadata() uses -- not yet upgraded to filter by
    real team_id (see that function's NOTE for why)."""
    others = [p for p in frame_others if p["player_id"] != player_id and p["pitch_x_m"] is not None]
    if not others:
        return None
    return min(((p["pitch_x_m"] - x) ** 2 + (p["pitch_y_m"] - y) ** 2) ** 0.5 for p in others)


def _compute_off_ball_inputs(player_id: int, points: list, players_by_frame: dict[int, list[dict]]) -> dict:
    """
    Derives score_off_ball_movement()'s inputs from real trajectory data (see
    _score_player_intelligence below).

    Honesty notes on what IS and ISN'T computed here:
      - total_path_length_m / net_displacement_m: real, computed directly
        from this player's own trajectory -- no team info needed.
      - avg_distance_gained_from_marker: real, but built on the same
        nearest-other-player stand-in for "opponent" as first-touch
        pressure (see _nearest_other_player_distance's docstring) --
        averages the positive frame-to-frame change in that distance
        (separation gained), zero contribution on frames where the
        player closed distance instead of gaining it.
      - off_ball_frames_in_attacking_third / total_off_ball_frames_while_
        team_in_possession: deliberately left at 0/0. Team assignment now
        exists (real team_id, see team_assignment.py), so (a) is
        available; (b) -- possession windows scoped to a team -- still
        isn't computed anywhere in this pipeline, so this sub-score isn't
        wired up to real data yet either way. Passing 0/0 makes
        score_off_ball_movement()'s safe_ratio(..., default=None) return
        None for this sub-score honestly, rather than guessing a
        plausible-looking count. A natural fast-follow once team-scoped
        possession windows exist, out of scope for this pass.
      - sample_size_phases: count of valid consecutive-frame pairs used
        for the space-creation computation above. A real, grounded
        number tied to actual measured data -- but an approximation of
        "off-ball phases" in the sports-science sense (a phase would
        normally be one continuous off-the-ball stretch during a
        team-possession window); described as such rather than presented
        as a validated phase count.
    """
    valid_points = [p for p in points if p.pitch_x_m is not None]

    total_path_length_m = 0.0
    for prev, nxt in zip(valid_points, valid_points[1:]):
        total_path_length_m += ((nxt.pitch_x_m - prev.pitch_x_m) ** 2 + (nxt.pitch_y_m - prev.pitch_y_m) ** 2) ** 0.5

    net_displacement_m = 0.0
    if len(valid_points) >= 2:
        first, last = valid_points[0], valid_points[-1]
        net_displacement_m = ((last.pitch_x_m - first.pitch_x_m) ** 2 + (last.pitch_y_m - first.pitch_y_m) ** 2) ** 0.5

    separation_gains: list[float] = []
    for p_now, p_next in zip(points, points[1:]):
        if p_now.pitch_x_m is None or p_next.pitch_x_m is None:
            continue
        # Look players up by real frame_id, and compare consecutive FRAMES, not
        # consecutive points: after an occlusion gap p_now and p_next can be
        # seconds apart, and the separation change across that gap is not an
        # off-ball movement the player made.
        if p_next.frame_id != p_now.frame_id + 1:
            continue
        dist_now = _nearest_other_player_distance(
            player_id, p_now.pitch_x_m, p_now.pitch_y_m,
            players_by_frame.get(p_now.frame_id, []))
        dist_next = _nearest_other_player_distance(
            player_id, p_next.pitch_x_m, p_next.pitch_y_m,
            players_by_frame.get(p_next.frame_id, []))
        if dist_now is None or dist_next is None:
            continue
        gain = dist_next - dist_now
        separation_gains.append(max(gain, 0.0))

    avg_distance_gained_from_marker = sum(separation_gains) / len(separation_gains) if separation_gains else 0.0

    return {
        "avg_distance_gained_from_marker": avg_distance_gained_from_marker,
        "off_ball_frames_in_attacking_third": 0,
        "total_off_ball_frames_while_team_in_possession": 0,
        "net_displacement_m": net_displacement_m,
        "total_path_length_m": total_path_length_m,
        "sample_size_phases": len(separation_gains),
    }


def _compute_decision_times(possession_segments: list[dict]) -> dict[int, list[float]]:
    """
    One decision-time sample per touch = time until that touch's
    possession segment ends (the next segment's timestamp minus this
    one's) -- fed to score_decision_making()'s decision_speed sub-score,
    same scale press_resistance_score already uses for its own
    decision_speed. A player's LAST touch in the match/window has no
    "next" entry to measure against and is dropped from their samples
    entirely, not imputed -- same drop-not-impute convention as
    _enrich_first_touch_metadata().
    """
    times: dict[int, list[float]] = {}
    for i in range(len(possession_segments) - 1):
        pid = possession_segments[i]["player_id"]
        dt = possession_segments[i + 1]["timestamp"] - possession_segments[i]["timestamp"]
        times.setdefault(pid, []).append(dt)
    return times


def _compute_passing_vision_inputs(
    player_id: int, events_by_type_and_player: dict[str, dict[int, list[Event]]],
    attacking_direction: str = DIRECTION_UNKNOWN,
) -> dict:
    """
    completed_passes / turnovers_lost: real event counts -- both only
    classify correctly once team assignment (Stage 1.5) has run, which is
    why score_passing_vision() gates on team_assignment_confidence rather
    than treating these as always-safe inputs (see
    ai/computer_vision/pass_detection/pass_heuristics.py's team1==team2
    check).
    pass_sector_counts: real, bucketed from each pass event's
    metadata_json start_x/start_y -> Event.pitch_x_m/pitch_y_m via atan2,
    into PASS_DIRECTION_SECTORS compass buckets.

    forward_passes: uses the attacking direction inferred for THIS player's
    team (see tactical_analysis/attacking_direction.py), never a fixed
    "increasing x is forward", which is right for only one of the two teams and
    for neither after the ends switch.

    When the direction is `unknown` -- which is the normal case on real
    broadcast footage, because inferring it needs pitch coordinates and
    calibration is invalid there -- `forward_passes` is returned as None
    rather than as a number derived from a coin-flip. score_passing_vision()
    receives None and reports that sub-score honestly instead of scoring a
    direction it does not know.
    """
    passes = events_by_type_and_player.get("pass", {}).get(player_id, [])
    turnovers_lost = events_by_type_and_player.get("turnover", {}).get(player_id, [])

    sector_counts = [0] * PASS_DIRECTION_SECTORS
    known_direction = attacking_direction in ("left_to_right", "right_to_left")
    forward_passes: int | None = 0 if known_direction else None

    for e in passes:
        meta = e.metadata_json or {}
        start_x, start_y = meta.get("start_x"), meta.get("start_y")
        end_x, end_y = e.pitch_x_m, e.pitch_y_m
        if start_x is None or start_y is None or end_x is None or end_y is None:
            continue
        angle = math.atan2(end_y - start_y, end_x - start_x)
        sector = int(((angle + math.pi) / (2 * math.pi)) * PASS_DIRECTION_SECTORS) % PASS_DIRECTION_SECTORS
        sector_counts[sector] += 1
        if known_direction:
            advanced = (end_x > start_x) if attacking_direction == "left_to_right" else (end_x < start_x)
            if advanced:
                forward_passes += 1

    return {
        "completed_passes": len(passes),
        "turnovers_lost": len(turnovers_lost),
        "pass_sector_counts": sector_counts,
        "forward_passes": forward_passes,
        "attacking_direction": attacking_direction,
    }


def _compute_finishing_inputs(
    player_id: int, events_by_type_and_player: dict[str, dict[int, list[Event]]]
) -> list[dict]:
    """This player's shot events' real location/trajectory-projection
    fields, for score_finishing_efficiency()'s per-shot loop."""
    shots = events_by_type_and_player.get("shot", {}).get(player_id, [])
    return [
        {
            "pitch_x_m": e.pitch_x_m,
            "pitch_y_m": e.pitch_y_m,
            "projected_y_at_goal": (e.metadata_json or {}).get("projected_y_at_goal"),
        }
        for e in shots
    ]


def _score_player_intelligence(
    match: Match, trajectories: dict, events: list[Event], homography_confidence: float, fps: float,
    team_assignment_confidence: float, possession_segments: list[dict],
    directions=None,
):
    results = []
    events_by_type_and_player: dict[str, dict[int, list[Event]]] = {}
    for e in events:
        if e.player_id is not None:
            events_by_type_and_player.setdefault(e.event_type, {}).setdefault(e.player_id, []).append(e)

    events_by_player: dict[int, list[dict]] = {
        pid: [{"homography_confidence": e.homography_confidence, "metadata_json": e.metadata_json} for e in evs]
        for pid, evs in events_by_type_and_player.get("first_touch", {}).items()
    }

    players_by_frame = _build_players_by_frame(trajectories)
    decision_times_by_player = _compute_decision_times(possession_segments)

    for player_id, points in trajectories.items():
        player_events = events_by_player.get(player_id, [])
        first_touch = score_first_touch(player_events, homography_confidence=homography_confidence)
        results.append((first_touch, player_id))

        # Press Resistance and Defensive Positioning both require real
        # team-scoped aggregates (possessions under pressure, defensive
        # line deviation, etc.) that depend on knowing who's on which
        # team -- team_assignment_confidence is now a real, per-run
        # measurement from Stage 1.5 (assign_teams()), so these correctly
        # report low_upstream_confidence only when that measurement is
        # genuinely low, not unconditionally.
        press_resistance = score_press_resistance(team_assignment_confidence=team_assignment_confidence)
        results.append((press_resistance, player_id))

        defensive_positioning = score_defensive_positioning(team_assignment_confidence=team_assignment_confidence)
        results.append((defensive_positioning, player_id))

        # Pass the real off-ball inputs. Called with homography_confidence
        # alone, every other input defaults to 0 and the metric returns
        # "low_sample" without measuring anything. See
        # _compute_off_ball_inputs() above for what is real vs. still honestly
        # unmeasured.
        off_ball_inputs = _compute_off_ball_inputs(player_id, points, players_by_frame)
        off_ball = score_off_ball_movement(homography_confidence=homography_confidence, **off_ball_inputs)
        results.append((off_ball, player_id))

        # Body orientation / scanning behavior. Both are sparse --
        # only the stride-sampled frames have a real reading (see
        # POSE_SAMPLE_STRIDE / _estimate_orientations) -- so most matches/
        # players will land on "low_sample" until enough readings
        # accumulate, same honest pattern as everything else here.
        orientation_values = [p.body_orientation_deg for p in points if p.body_orientation_deg is not None]
        body_orientation = score_body_orientation(orientation_values)
        results.append((body_orientation, player_id))

        orientation_readings = [
            (p.frame_id / fps, p.body_orientation_deg)
            for p in points
            if p.body_orientation_deg is not None
        ]
        tracked_duration_s = (points[-1].frame_id - points[0].frame_id) / fps if len(points) >= 2 else 0.0
        scanning = score_scanning_behavior(orientation_readings, tracked_duration_s=tracked_duration_s)
        results.append((scanning, player_id))

        # Passing Vision / Decision Making: both gated on
        # team_assignment_confidence, same reasoning as Press Resistance/
        # Defensive Positioning above -- their real event counts only
        # classify correctly once team_id is real (see
        # _compute_passing_vision_inputs()'s docstring).
        player_team = _dominant_team_id(points)
        passing_inputs = _compute_passing_vision_inputs(
            player_id, events_by_type_and_player,
            attacking_direction=(directions.for_team(player_team)
                                 if directions is not None else DIRECTION_UNKNOWN),
        )
        # Carried for provenance, not a scorer input.
        passing_direction = passing_inputs.pop("attacking_direction")
        passing_vision = score_passing_vision(team_assignment_confidence=team_assignment_confidence, **passing_inputs)
        passing_vision.setdefault("sub_scores", {})["attacking_direction"] = passing_direction
        results.append((passing_vision, player_id))

        player_decision_times = decision_times_by_player.get(player_id, [])
        avg_decision_time = (
            sum(player_decision_times) / len(player_decision_times) if player_decision_times else None
        )
        n_pass = len(events_by_type_and_player.get("pass", {}).get(player_id, []))
        n_shot = len(events_by_type_and_player.get("shot", {}).get(player_id, []))
        n_turnover = len(events_by_type_and_player.get("turnover", {}).get(player_id, []))
        decision_making = score_decision_making(
            successful_actions=n_pass + n_shot,
            lost_actions=n_turnover,
            avg_decision_time=avg_decision_time,
            team_assignment_confidence=team_assignment_confidence,
        )
        results.append((decision_making, player_id))

        # Finishing Efficiency: gated on homography_confidence, NOT
        # team_assignment_confidence -- shot location/attribution doesn't
        # depend on team_id (see score_finishing_efficiency()'s docstring).
        shots = _compute_finishing_inputs(player_id, events_by_type_and_player)
        finishing = score_finishing_efficiency(shots=shots, homography_confidence=homography_confidence)
        results.append((finishing, player_id))

    return results
