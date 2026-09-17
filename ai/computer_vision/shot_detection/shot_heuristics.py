"""
Shot detection heuristics.
Implementation Spec §2.

PHASE 3 CHANGES
    1. The goal mouth is no longer ASSUMED. `goalpost_v1` is trained and
       registered, its detections are already carried per frame on
       `FrameData.goalposts` and projected to pitch metres by
       runner._project_frame_geometry(). `build_goal_mouths()` turns those
       observations into a measured goal-mouth rectangle, gated on
       detection confidence the same way pitch coordinates are gated on
       `calibration.valid`. When no goal was detected with enough
       confidence, the nominal geometry is used and every emitted event
       says so in `goal_geometry_source`, so a consumer can tell a measured
       result from a nominal one.
    2. `attacking_direction` is per-team and may be unknown. There is no
       left_to_right default: without a direction there is no "towards
       goal", and events are emitted as `uncertain` rather than scored
       against a guessed end.
    3. Fast ball movement near a goal line is not automatically a shot. It
       is separated into shot / cross / clearance, and anything that does
       not clearly satisfy one is labelled `uncertain` rather than forced
       into a category.

METHOD
    heuristic_proxy throughout. Velocity thresholds and geometric windows
    are proxies for intent; none of this is a trained event model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ai.computer_vision.tactical_analysis.constants import (
    GOAL_WIDTH_M,
    PENALTY_AREA_DEPTH_M,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    SHOT_VELOCITY_MIN_MS,
)

#: A goalpost detection below this confidence is not used to define the
#: goal mouth. Mirrors the calibration gate's intent: a low-confidence
#: geometric anchor is worse than a nominal one, because it looks measured.
GOALPOST_CONFIDENCE_MIN = 0.5

#: How far outside the posts a trajectory may project and still count as
#: on-target-ish. Kept from the original implementation (±2 m) -- it
#: absorbs extrapolation error over a long distance, not shot accuracy.
GOAL_MOUTH_TOLERANCE_M = 2.0

#: A ball crossing the goal line this far outside the posts, from a wide
#: position, reads as a cross rather than a wayward shot. Half the pitch
#: width minus the goal is the whole rest of the byline, so this is
#: deliberately generous.
CROSS_MIN_OFFSET_M = 4.0

#: Lateral distance from the touchline within which a delivery is "wide".
WIDE_CHANNEL_M = 18.0


@dataclass
class GoalMouth:
    """A goal mouth in pitch coordinates, measured or nominal."""

    side: str                  # "left" (x=0) | "right" (x=PITCH_LENGTH_M)
    x_m: float
    y_min_m: float
    y_max_m: float
    source: str                # "detected" | "nominal"
    confidence: float = 0.0
    n_observations: int = 0

    @property
    def y_center_m(self) -> float:
        return (self.y_min_m + self.y_max_m) / 2.0

    def contains(self, y: float, tolerance: float = GOAL_MOUTH_TOLERANCE_M) -> bool:
        return (self.y_min_m - tolerance) <= y <= (self.y_max_m + tolerance)


def nominal_goal_mouth(side: str) -> GoalMouth:
    """The goal mouth implied by the pitch constants. Used only when no
    goalpost was detected confidently; always labelled `nominal`."""
    half = GOAL_WIDTH_M / 2.0
    centre = PITCH_WIDTH_M / 2.0
    return GoalMouth(
        side=side,
        x_m=0.0 if side == "left" else PITCH_LENGTH_M,
        y_min_m=centre - half,
        y_max_m=centre + half,
        source="nominal",
    )


def build_goal_mouths(frame_data, min_confidence: float = GOALPOST_CONFIDENCE_MIN) -> dict[str, GoalMouth]:
    """Measures each goal mouth from the goalpost model's detections.

    Only detections that (a) clear `min_confidence`, (b) were projected to
    pitch metres -- which requires that frame's calibration to have been
    valid -- and (c) were assigned a side, are used. The median of the
    observed y-extent is taken rather than the mean so one badly-projected
    frame cannot drag the mouth across the pitch.

    Returns only the sides that were actually measured; callers fall back
    to `nominal_goal_mouth()` for the rest and record which they used.
    """
    by_side: dict[str, list[tuple[float, float, float]]] = {"left": [], "right": []}
    for f in frame_data or []:
        if not getattr(f, "calibration", None) or not f.calibration.valid:
            continue
        for gp in getattr(f, "goalposts", None) or []:
            if gp.side not in ("left", "right"):
                continue
            if gp.confidence < min_confidence:
                continue
            if gp.pitch_x_m is None or gp.pitch_y_m is None:
                continue
            # Half the projected post width, in metres, either side of the
            # detected centre. The detection is a box around the mouth, so
            # its own width is the best available estimate of the span.
            by_side[gp.side].append((float(gp.pitch_x_m), float(gp.pitch_y_m), float(gp.confidence)))

    def _median(vals: list[float]) -> float:
        s = sorted(vals)
        return s[len(s) // 2]

    mouths: dict[str, GoalMouth] = {}
    for side, obs in by_side.items():
        if not obs:
            continue
        xs = [o[0] for o in obs]
        ys = [o[1] for o in obs]
        confs = [o[2] for o in obs]
        cy = _median(ys)
        half = GOAL_WIDTH_M / 2.0
        mouths[side] = GoalMouth(
            side=side,
            x_m=_median(xs),
            # The detected centre positions the mouth; its WIDTH stays the
            # regulation 7.32 m rather than being read from the projected
            # box, because a box width projected through a homography at a
            # shallow angle is far noisier than the known real width.
            y_min_m=cy - half,
            y_max_m=cy + half,
            source="detected",
            confidence=_median(confs),
            n_observations=len(obs),
        )
    return mouths


def _target_side(attacking_direction: str) -> str | None:
    if attacking_direction == "left_to_right":
        return "right"
    if attacking_direction == "right_to_left":
        return "left"
    return None


def _classify(
    x1: float, y1: float, dx: float, dy: float, velocity: float,
    mouth: GoalMouth, own_mouth: GoalMouth | None,
) -> tuple[str, str, float | None]:
    """Returns (event_type, reason, projected_y_at_goal).

    Deliberately conservative: everything that is not clearly one thing is
    `uncertain`. A wrong label is worse than an honest refusal here,
    because finishing efficiency divides by the shot count.
    """
    toward_goal = (mouth.x_m - x1) * dx > 0
    if not toward_goal or abs(dx) < 1e-5:
        return "uncertain", "fast_ball_not_heading_at_target_goal", None

    # A clearance is checked BEFORE the goal-directed cases, because on a
    # one-dimensional x axis "away from our own goal" and "towards their
    # goal" are the same sign. What separates a clearance from a shot is
    # not direction but ORIGIN: it starts in the defending team's own
    # penalty area, ~90 m from the goal it is nominally travelling at.
    if own_mouth is not None and abs(x1 - own_mouth.x_m) <= PENALTY_AREA_DEPTH_M:
        return "clearance", "fast_ball_struck_from_inside_own_penalty_area", None

    t = (mouth.x_m - x1) / dx
    if t <= 0:
        return "uncertain", "goal_line_behind_ball_travel", None
    y_at_goal = y1 + t * dy

    in_attacking_half = abs(x1 - mouth.x_m) <= PITCH_LENGTH_M / 2.0
    if not in_attacking_half:
        # Struck from its own half but outside its own box -- could be a
        # long clearance, a switch of play, or a speculative effort. Not
        # separable on this evidence, so it is not labelled.
        return "uncertain", "origin_outside_attacking_half", round(y_at_goal, 2)

    if mouth.contains(y_at_goal):
        return "shot", "trajectory_intersects_detected_goal_mouth", round(y_at_goal, 2)

    # Misses the mouth. A delivery from a wide channel that crosses the
    # byline well outside the posts is a cross, not a bad shot.
    offset = min(abs(y_at_goal - mouth.y_min_m), abs(y_at_goal - mouth.y_max_m))
    from_wide = min(y1, PITCH_WIDTH_M - y1) <= WIDE_CHANNEL_M
    if offset >= CROSS_MIN_OFFSET_M and from_wide:
        return "cross", "wide_origin_crossing_byline_outside_posts", round(y_at_goal, 2)

    return "uncertain", "misses_mouth_but_origin_not_wide", round(y_at_goal, 2)


def detect_shots(
    ball_positions: list[dict],
    fps: float = 25.0,
    attacking_direction: str = "unknown",
    goal_mouths: dict[str, GoalMouth] | None = None,
    direction_by_team: dict[str, str] | None = None,
) -> list[dict]:
    """
    Detects goal-directed ball events from velocity plus goal-mouth geometry.

    Args:
        ball_positions: dicts with frame_id, timestamp, pitch_x_m, pitch_y_m,
            homography_confidence, and optionally player_id/team_id.
        attacking_direction: fallback direction when the ball carrier's team
            is unknown. "unknown" means no direction is assumed.
        goal_mouths: measured mouths from build_goal_mouths(); missing sides
            fall back to nominal geometry, recorded per event.
        direction_by_team: per-team directions, preferred over the single
            fallback when the ball point carries a team_id.

    Returns:
        Event dicts whose `event_type` is one of shot | cross | clearance |
        uncertain.
    """
    events: list[dict] = []
    if len(ball_positions) < 2:
        return events

    mouths = dict(goal_mouths or {})
    dt = 1.0 / fps

    for b1, b2 in zip(ball_positions, ball_positions[1:]):
        x1, y1 = b1.get("pitch_x_m"), b1.get("pitch_y_m")
        x2, y2 = b2.get("pitch_x_m"), b2.get("pitch_y_m")
        if x1 is None or y1 is None or x2 is None or y2 is None:
            continue

        dx, dy = x2 - x1, y2 - y1
        velocity = math.hypot(dx, dy) / dt
        if velocity < SHOT_VELOCITY_MIN_MS:
            continue

        team_id = b1.get("team_id") or b2.get("team_id")
        direction = attacking_direction
        if direction_by_team and team_id is not None:
            direction = direction_by_team.get(str(team_id), attacking_direction)

        side = _target_side(direction)
        if side is None:
            # No direction for this team: a fast ball is real, but which
            # goal it is heading for is unknown, so it cannot be a shot.
            events.append(_event("uncertain", b1, b2, velocity, None,
                                 "attacking_direction_unknown", None, direction))
            continue

        mouth = mouths.get(side) or nominal_goal_mouth(side)
        own = mouths.get("left" if side == "right" else "right") or \
            nominal_goal_mouth("left" if side == "right" else "right")

        event_type, reason, y_at_goal = _classify(x1, y1, dx, dy, velocity, mouth, own)
        events.append(_event(event_type, b1, b2, velocity, mouth, reason, y_at_goal, direction))

    return events


def _event(event_type, b1, b2, velocity, mouth, reason, y_at_goal, direction) -> dict:
    return {
        "event_type": event_type,
        "player_id": b1.get("player_id") or b2.get("player_id"),
        "team_id": b1.get("team_id"),
        "timestamp": b2.get("timestamp", 0.0),
        "pitch_x_m": b2.get("pitch_x_m"),
        "pitch_y_m": b2.get("pitch_y_m"),
        "homography_confidence": b2.get("homography_confidence", 1.0),
        "metadata_json": {
            "ball_velocity_ms": round(velocity, 2),
            "projected_y_at_goal": y_at_goal,
            "classification_reason": reason,
            "attacking_direction": direction,
            # Provenance: whether the geometry this was judged against was
            # measured by goalpost_v1 or assumed from the pitch constants.
            "goal_geometry_source": mouth.source if mouth else "none",
            "goal_mouth_confidence": round(mouth.confidence, 3) if mouth else None,
            "goal_mouth_observations": mouth.n_observations if mouth else 0,
            "goal_side": mouth.side if mouth else None,
            "method": "heuristic_proxy",
        },
    }
