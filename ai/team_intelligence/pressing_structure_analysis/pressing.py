"""
Pressing Intensity analysis.
Analysis Logic Design v3 & Implementation Spec §1.2.
"""

from __future__ import annotations

import numpy as np

from ai.common.metrics import MetricResult, metric_result
from ai.computer_vision.tactical_analysis.constants import (
    SCHEMA_VERSION,
    TEAM_ASSIGNMENT_CONFIDENCE_MIN,
    pressure_level,
)


def compute_pressing_intensity(
    frames_defender_to_ball_distances: list[list[float]],
    team_assignment_confidence: float = 0.0,
) -> MetricResult:
    """
    Computes pressing intensity score when opposing team has ball.
    Mean pressure_level across defending team players.

    Returns TeamMetric dict with metric_name="pressing_intensity_score".
    """
    if team_assignment_confidence < TEAM_ASSIGNMENT_CONFIDENCE_MIN or len(frames_defender_to_ball_distances) == 0:
        return metric_result(
            "pressing_intensity_score",
            None,
            method="heuristic_proxy",
            confidence="low_upstream_confidence" if team_assignment_confidence < TEAM_ASSIGNMENT_CONFIDENCE_MIN else "low_sample",
            sample_size=len(frames_defender_to_ball_distances),
            sub_scores={"mean_pressure_level": None},
            schema_version=SCHEMA_VERSION,
        )

    frame_pressures = []
    for dists in frames_defender_to_ball_distances:
        pressures = [pressure_level(d) for d in dists if d is not None]
        valid_pressures = [p for p in pressures if p is not None]
        if valid_pressures:
            frame_pressures.append(float(np.mean(valid_pressures)))

    if not frame_pressures:
        return metric_result(
            "pressing_intensity_score",
            None,
            method="heuristic_proxy",
            confidence="low_sample",
            sample_size=0,
            sub_scores={"mean_pressure_level": None},
            schema_version=SCHEMA_VERSION,
        )

    mean_pressure = float(np.mean(frame_pressures))
    intensity_score = round(mean_pressure * 100.0, 1)

    return metric_result(
        "pressing_intensity_score",
        intensity_score,
        method="heuristic_proxy",
        confidence="normal",
        sample_size=len(frame_pressures),
        sub_scores={
            "mean_pressure_level": round(mean_pressure, 3),
            "raw_intensity": intensity_score,
        },
        schema_version=SCHEMA_VERSION,
    )
