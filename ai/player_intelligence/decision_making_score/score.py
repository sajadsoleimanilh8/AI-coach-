"""
Decision Making Score implementation.
Analysis Logic Design v3 (docs/data_analysis.md §9).
"""

from __future__ import annotations

from ai.computer_vision.tactical_analysis.constants import (
    DECISION_TIME_MAX_S,
    DECISION_TIME_MIN_S,
    MIN_SAMPLE_EVENTS,
    SCHEMA_VERSION,
    TEAM_ASSIGNMENT_CONFIDENCE_MIN,
)
from ai.computer_vision.tactical_analysis.utils import clip, safe_ratio


def score_decision_making(
    successful_actions: int = 0,
    lost_actions: int = 0,
    avg_decision_time: float | None = None,
    team_assignment_confidence: float = 0.0,
) -> dict:
    """
    Computes Decision Making Score per docs/data_analysis.md §9.
    Gated on team_assignment_confidence >= TEAM_ASSIGNMENT_CONFIDENCE_MIN
    (0.5). avg_decision_time itself is team-agnostic (it's the time
    between receiving the ball and the next possession-ending event,
    """
    if team_assignment_confidence < TEAM_ASSIGNMENT_CONFIDENCE_MIN:
        return {
            "metric_name": "decision_making_score",
            "value": None,
            "method": "heuristic_proxy",
            "confidence": "low_upstream_confidence",
            "sample_size": successful_actions + lost_actions,
            "sub_scores": {
                "retention": None,
                "decision_speed": None,
            },
            "schema_version": SCHEMA_VERSION,
        }

    retention_raw = safe_ratio(successful_actions, successful_actions + lost_actions)
    retention_score = 100.0 * retention_raw if retention_raw is not None else None

    if avg_decision_time is not None:
        decision_speed_score = 100.0 * clip(
            (DECISION_TIME_MAX_S - avg_decision_time) / (DECISION_TIME_MAX_S - DECISION_TIME_MIN_S), 0.0, 1.0
        )
    else:
        decision_speed_score = None

    components = {
        "retention": (retention_score, 0.50),
        "decision_speed": (decision_speed_score, 0.50),
    }

    valid = {k: (v, w) for k, (v, w) in components.items() if v is not None}
    sample_size = successful_actions + lost_actions

    if not valid or sample_size < MIN_SAMPLE_EVENTS:
        final_val = None
        confidence_enum = "low_sample"
    else:
        weight_sum = sum(w for _, w in valid.values())
        final_val = sum(v * w for v, w in valid.values()) / weight_sum
        confidence_enum = "normal"

    return {
        "metric_name": "decision_making_score",
        "value": round(final_val, 1) if final_val is not None else None,
        "method": "heuristic_proxy",
        "confidence": confidence_enum,
        "sample_size": sample_size,
        "sub_scores": {
            "retention": round(retention_score, 1) if retention_score is not None else None,
            "decision_speed": round(decision_speed_score, 1) if decision_speed_score is not None else None,
        },
        "schema_version": SCHEMA_VERSION,
    }
