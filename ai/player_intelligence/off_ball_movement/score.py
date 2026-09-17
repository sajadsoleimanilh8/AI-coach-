"""
Off-Ball Movement Score implementation (§3.1 of Phase 3 Spec).
"""

from __future__ import annotations

from ai.common.metrics import Confidence, MetricResult, metric_result
from ai.computer_vision.tactical_analysis.constants import (
    HOMOGRAPHY_CONFIDENCE_MIN,
    MAX_USEFUL_SEPARATION_GAIN_M,
    MIN_SAMPLE_EVENTS,
    SCHEMA_VERSION,
)
from ai.computer_vision.tactical_analysis.utils import clip, safe_ratio


def score_off_ball_movement(
    avg_distance_gained_from_marker: float = 0.0,
    off_ball_frames_in_attacking_third: int = 0,
    total_off_ball_frames_while_team_in_possession: int = 0,
    net_displacement_m: float = 0.0,
    total_path_length_m: float = 0.0,
    sample_size_phases: int = 0,
    homography_confidence: float = 1.0,
) -> MetricResult:
    """
    Computes Off-Ball Movement Score per §3.1.

    Returns PlayerMetric dict.
    """
    if homography_confidence < HOMOGRAPHY_CONFIDENCE_MIN:
        return metric_result(
            "off_ball_movement_score",
            None,
            method="heuristic_proxy",
            confidence="low_upstream_confidence",
            sample_size=sample_size_phases,
            sub_scores={
                "space_creation": None,
                "attacking_presence": None,
                "efficiency": None,
            },
            schema_version=SCHEMA_VERSION,
        )

    # 1. Space Creation (50%)
    space_raw = safe_ratio(avg_distance_gained_from_marker, MAX_USEFUL_SEPARATION_GAIN_M, default=0.0)
    space_creation_score = 100.0 * clip(space_raw, 0.0, 1.0) if space_raw is not None else None

    # 2. Attacking Presence (30%)
    presence_raw = safe_ratio(
        off_ball_frames_in_attacking_third, total_off_ball_frames_while_team_in_possession, default=None
    )
    attacking_presence_score = 100.0 * presence_raw if presence_raw is not None else None

    # 3. Efficiency (20%)
    efficiency_raw = safe_ratio(net_displacement_m, total_path_length_m, default=None)
    efficiency_score = 100.0 * efficiency_raw if efficiency_raw is not None else None

    components = {
        "space_creation": (space_creation_score, 0.50),
        "attacking_presence": (attacking_presence_score, 0.30),
        "efficiency": (efficiency_score, 0.20),
    }

    valid = {k: (v, w) for k, (v, w) in components.items() if v is not None}

    if not valid or sample_size_phases < MIN_SAMPLE_EVENTS:
        final_val = None
        confidence_enum: Confidence = "low_sample"
    else:
        weight_sum = sum(w for _, w in valid.values())
        final_val = sum(v * w for v, w in valid.values()) / weight_sum
        confidence_enum = "normal"

    return metric_result(
        "off_ball_movement_score",
        round(final_val, 1) if final_val is not None else None,
        method="heuristic_proxy",
        confidence=confidence_enum,
        sample_size=sample_size_phases,
        sub_scores={
            "space_creation": round(space_creation_score, 1) if space_creation_score is not None else None,
            "attacking_presence": round(attacking_presence_score, 1) if attacking_presence_score is not None else None,
            "efficiency": round(efficiency_score, 1) if efficiency_score is not None else None,
        },
        schema_version=SCHEMA_VERSION,
    )
