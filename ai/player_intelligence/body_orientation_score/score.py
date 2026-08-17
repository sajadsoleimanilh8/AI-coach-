"""
Body Orientation Score.
"""

from __future__ import annotations

from ai.computer_vision.tactical_analysis.constants import (
    IDEAL_SQUARENESS_DEG,
    MIN_SAMPLE_EVENTS,
    SCHEMA_VERSION,
    SQUARENESS_TOLERANCE_DEG,
)
from ai.computer_vision.tactical_analysis.utils import gaussian_score


def _squareness_deg(orientation_deg: float) -> float:
    """
    Maps a 0-360 shoulder-line angle to a 0-90 "squareness" value:
    0 = shoulders parallel to the pitch's long axis (x-axis),
    90 = shoulders perpendicular to it (fully side-on).
    """
    half_turn = orientation_deg % 180.0
    return min(half_turn, 180.0 - half_turn)


def score_body_orientation(
    orientation_readings: list[float] | None = None,
) -> dict:
    """
    Computes Body Orientation Score.
    """
    orientation_readings = orientation_readings or []
    sample_size = len(orientation_readings)

    if sample_size < MIN_SAMPLE_EVENTS:
        return {
            "metric_name": "body_orientation_score",
            "value": None,
            "method": "heuristic_proxy",
            "confidence": "low_sample",
            "sample_size": sample_size,
            "sub_scores": {"avg_squareness_deg": None},
            "schema_version": SCHEMA_VERSION,
        }

    squareness_values = [_squareness_deg(a) for a in orientation_readings]
    avg_squareness = sum(squareness_values) / len(squareness_values)

    value = round(gaussian_score(avg_squareness, IDEAL_SQUARENESS_DEG, SQUARENESS_TOLERANCE_DEG), 1)

    return {
        "metric_name": "body_orientation_score",
        "value": value,
        "method": "heuristic_proxy",
        "confidence": "normal",
        "sample_size": sample_size,
        "sub_scores": {"avg_squareness_deg": round(avg_squareness, 1)},
        "schema_version": SCHEMA_VERSION,
    }
