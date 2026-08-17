"""
Press Resistance Score implementation.
Analysis Logic Design v3 (docs/data_analysis.md §2).
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


def score_press_resistance(
    possessions_under_pressure: int = 0,
    possessions_retained: int = 0,
    pressure_events: int = 0,
    successful_escapes: int = 0,
    attempted_passes_under_pressure: int = 0,
    completed_passes_under_pressure: int = 0,
    sum_success_density: float = 0.0,
    sum_density: float = 0.0,
    avg_decision_time: float | None = None,
    team_assignment_confidence: float = 0.0,
) -> dict:
    """
    Computes Press Resistance Score per docs/data_analysis.md §2.
    Gated on team_assignment_confidence >= TEAM_ASSIGNMENT_CONFIDENCE_MIN (0.5).
    """
    if team_assignment_confidence < TEAM_ASSIGNMENT_CONFIDENCE_MIN:
        return {
            "metric_name": "press_resistance_score",
            "value": None,
            "method": "heuristic_proxy",
            "confidence": "low_upstream_confidence",
            "sample_size": pressure_events,
            "sub_scores": {
                "retention": None,
                "escape": None,
                "pass_accuracy": None,
                "density": None,
                "decision_speed": None,
            },
            "schema_version": SCHEMA_VERSION,
        }

    retention_raw = safe_ratio(possessions_retained, possessions_under_pressure)
    retention_score = 100.0 * retention_raw if retention_raw is not None else None

    escape_raw = safe_ratio(successful_escapes, pressure_events)
    escape_score = 100.0 * escape_raw if escape_raw is not None else None

    pass_acc_raw = safe_ratio(completed_passes_under_pressure, attempted_passes_under_pressure)
    pass_accuracy_score = 100.0 * pass_acc_raw if pass_acc_raw is not None else None

    density_raw = safe_ratio(sum_success_density, sum_density)
    density_score = 100.0 * density_raw if density_raw is not None else None

    if avg_decision_time is not None:
        decision_speed_score = 100.0 * clip(
            (DECISION_TIME_MAX_S - avg_decision_time) / (DECISION_TIME_MAX_S - DECISION_TIME_MIN_S), 0.0, 1.0
        )
    else:
        decision_speed_score = None

    components = {
        "retention": (retention_score, 0.30),
        "escape": (escape_score, 0.25),
        "pass_accuracy": (pass_accuracy_score, 0.20),
        "density": (density_score, 0.15),
        "decision_speed": (decision_speed_score, 0.10),
    }

    valid = {k: (v, w) for k, (v, w) in components.items() if v is not None}

    if not valid or pressure_events < MIN_SAMPLE_EVENTS:
        final_val = None
        confidence_enum = "low_sample"
    else:
        weight_sum = sum(w for _, w in valid.values())
        final_val = sum(v * w for v, w in valid.values()) / weight_sum
        confidence_enum = "normal"

    return {
        "metric_name": "press_resistance_score",
        "value": round(final_val, 1) if final_val is not None else None,
        "method": "heuristic_proxy",
        "confidence": confidence_enum,
        "sample_size": pressure_events,
        "sub_scores": {
            "retention": round(retention_score, 1) if retention_score is not None else None,
            "escape": round(escape_score, 1) if escape_score is not None else None,
            "pass_accuracy": round(pass_accuracy_score, 1) if pass_accuracy_score is not None else None,
            "density": round(density_score, 1) if density_score is not None else None,
            "decision_speed": round(decision_speed_score, 1) if decision_speed_score is not None else None,
        },
        "schema_version": SCHEMA_VERSION,
    }
