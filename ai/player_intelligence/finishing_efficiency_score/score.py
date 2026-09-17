"""
Finishing Efficiency Score implementation.
Analysis Logic Design v3 (docs/data analysis.md §10).

IMPORTANT: this is LOCATION-based shot quality (distance from goal, angle
within the goal mouth, placement of the extrapolated shot trajectory), NOT
a true conversion rate. No goal/miss/on-target outcome exists anywhere in
this pipeline's data -- ai/computer_vision/shot_detection/shot_heuristics.py's
`projected_y_at_goal` is only ever used internally to decide whether a fast
ball movement qualifies as a "shot" at all, never persisted or observed as
a real outcome. Presenting this as a conversion/accuracy rate would be
exactly the "confident-looking garbage" failure mode this codebase's own
honesty principles warn against (see backend/api/player_intelligence.py's
PLAYER_NAME_MAP fix comment for the established tone) -- so both this
docstring and the "method" semantics below must keep that distinction
explicit to any caller/UI presenting this score.
"""

from __future__ import annotations

import math

from ai.common.metrics import Confidence, MetricResult, metric_result
from ai.computer_vision.tactical_analysis.constants import (
    GOAL_WIDTH_M,
    HOMOGRAPHY_CONFIDENCE_MIN,
    MAX_REALISTIC_SHOOTING_ANGLE_DEG,
    MIN_SAMPLE_EVENTS,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    SCHEMA_VERSION,
    SHOT_DISTANCE_TOLERANCE_M,
)
from ai.computer_vision.tactical_analysis.utils import clip, gaussian_score


def _shooting_angle_deg(sx: float, sy: float, goal_x: float, post1_y: float, post2_y: float) -> float:
    """Classic shooting-angle-subtended-by-the-goal-mouth feature (a
    standard xG-model input): the angle, in degrees, between the vectors
    from the shot location to each goal post. Wider angle = more of the
    goal is visible/reachable from that spot = better chance, all else
    equal."""
    v1x, v1y = goal_x - sx, post1_y - sy
    v2x, v2y = goal_x - sx, post2_y - sy
    mag1 = math.hypot(v1x, v1y)
    mag2 = math.hypot(v2x, v2y)
    if mag1 == 0 or mag2 == 0:
        return MAX_REALISTIC_SHOOTING_ANGLE_DEG  # standing right on a post -- treat as maximally open
    cos_angle = clip((v1x * v2x + v1y * v2y) / (mag1 * mag2), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def score_finishing_efficiency(
    shots: list[dict] | None = None,
    homography_confidence: float = 1.0,
    attacking_direction: str = "left_to_right",
) -> MetricResult:
    """
    Computes Finishing Efficiency Score per docs/data analysis.md §10.
    Gated on homography_confidence >= HOMOGRAPHY_CONFIDENCE_MIN (0.6), NOT
    team_assignment_confidence -- shot location/attribution comes straight
    from ball-trajectory pitch coordinates and doesn't depend on team_id,
    matching first_touch_score/off_ball_movement_score's existing gating
    choice rather than press_resistance_score/defensive_positioning_score's.

    Args:
        shots: this player's shot events, each
            {"pitch_x_m": float, "pitch_y_m": float, "projected_y_at_goal": float}.

    Returns PlayerMetric dict.
    """
    shots = shots or []
    sample_size = len(shots)

    if homography_confidence < HOMOGRAPHY_CONFIDENCE_MIN:
        return metric_result(
            "finishing_efficiency_score",
            None,
            method="heuristic_proxy",
            confidence="low_upstream_confidence",
            sample_size=sample_size,
            sub_scores={
                "distance": None,
                "angle": None,
                "placement": None,
            },
            schema_version=SCHEMA_VERSION,
        )

    goal_x = PITCH_LENGTH_M if attacking_direction == "left_to_right" else 0.0
    goal_center_y = PITCH_WIDTH_M / 2.0
    half_goal = GOAL_WIDTH_M / 2.0
    post1_y, post2_y = goal_center_y - half_goal, goal_center_y + half_goal

    distance_vals: list[float] = []
    angle_vals: list[float] = []
    placement_vals: list[float] = []

    for shot in shots:
        sx, sy = shot.get("pitch_x_m"), shot.get("pitch_y_m")
        if sx is None or sy is None:
            continue
        distance_to_goal_m = math.hypot(goal_x - sx, goal_center_y - sy)
        distance_vals.append(gaussian_score(distance_to_goal_m, ideal=0.0, tolerance=SHOT_DISTANCE_TOLERANCE_M))

        angle_deg = _shooting_angle_deg(sx, sy, goal_x, post1_y, post2_y)
        angle_vals.append(100.0 * clip(angle_deg / MAX_REALISTIC_SHOOTING_ANGLE_DEG, 0.0, 1.0))

        proj_y = shot.get("projected_y_at_goal")
        if proj_y is not None:
            placement_vals.append(gaussian_score(proj_y, ideal=goal_center_y, tolerance=half_goal))

    distance_score = sum(distance_vals) / len(distance_vals) if distance_vals else None
    angle_score = sum(angle_vals) / len(angle_vals) if angle_vals else None
    placement_score = sum(placement_vals) / len(placement_vals) if placement_vals else None

    components = {
        "distance": (distance_score, 0.35),
        "angle": (angle_score, 0.35),
        "placement": (placement_score, 0.30),
    }

    valid = {k: (v, w) for k, (v, w) in components.items() if v is not None}

    if not valid or sample_size < MIN_SAMPLE_EVENTS:
        final_val = None
        confidence_enum: Confidence = "low_sample"
    else:
        weight_sum = sum(w for _, w in valid.values())
        final_val = sum(v * w for v, w in valid.values()) / weight_sum
        confidence_enum = "normal"

    return metric_result(
        "finishing_efficiency_score",
        round(final_val, 1) if final_val is not None else None,
        method="heuristic_proxy",
        confidence=confidence_enum,
        sample_size=sample_size,
        sub_scores={
            "distance": round(distance_score, 1) if distance_score is not None else None,
            "angle": round(angle_score, 1) if angle_score is not None else None,
            "placement": round(placement_score, 1) if placement_score is not None else None,
        },
        schema_version=SCHEMA_VERSION,
    )
