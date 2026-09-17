"""
Body Orientation Score.

Distinct from, and complementary to, scanning_behavior_score
(ai/player_intelligence/scanning_behavior/score.py): scanning measures
CHANGE in body orientation over time (how often a player checks their
surroundings), this measures the AVERAGE STANCE itself -- specifically,
how close to a "half-turned"/open body shape the player plays with,
rather than square-on or fully side-on. This is a standard coaching cue
(receive on the "half-turn" to see both the ball and forward space at
once), not invented for this project, but the specific
IDEAL_SQUARENESS_DEG/SQUARENESS_TOLERANCE_DEG constants are an
uncalibrated starting point -- see constants.py's comment.

Reads: PlayerTracking.body_orientation_deg time series.
Writes: PlayerMetric(metric_name="body_orientation_score", ...).

Known methodological limitation (stated here rather than hidden): a
shoulder-line angle alone is symmetric -- it cannot distinguish "facing
one way" from "facing the opposite way" along that line (e.g. a player
squared up facing north reads identically to one squared up facing
south). "Squareness" (how aligned the shoulders are with the pitch's long
axis) is well-defined regardless of which of the two directions the
player is actually facing, which is why this module scores squareness
rather than a specific facing direction -- extending this to a true
"is the player facing the ball" score would need an additional landmark
(e.g. nose/hip asymmetry) that pose.py does not currently extract.
"""

from __future__ import annotations

from ai.common.metrics import MetricResult, metric_result
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

    Uses modulo 180 first since a shoulder line has no inherent direction
    (see module docstring's methodological-limitation note) -- 10deg and
    190deg describe the same physical shoulder-line orientation.
    """
    half_turn = orientation_deg % 180.0
    return min(half_turn, 180.0 - half_turn)


def score_body_orientation(
    orientation_readings: list[float] | None = None,
) -> MetricResult:
    """
    Computes Body Orientation Score.

    Args:
        orientation_readings: list of body_orientation_deg values for this
            player across their tracked frames, already filtered to real
            (non-None) readings only.

    Returns:
        PlayerMetric dict per the Standard Output Contract.
    """
    orientation_readings = orientation_readings or []
    sample_size = len(orientation_readings)

    if sample_size < MIN_SAMPLE_EVENTS:
        return metric_result(
            "body_orientation_score",
            None,
            method="heuristic_proxy",
            confidence="low_sample",
            sample_size=sample_size,
            sub_scores={"avg_squareness_deg": None},
            schema_version=SCHEMA_VERSION,
        )

    squareness_values = [_squareness_deg(a) for a in orientation_readings]
    avg_squareness = sum(squareness_values) / len(squareness_values)

    value = round(gaussian_score(avg_squareness, IDEAL_SQUARENESS_DEG, SQUARENESS_TOLERANCE_DEG), 1)

    return metric_result(
        "body_orientation_score",
        value,
        method="heuristic_proxy",
        confidence="normal",
        sample_size=sample_size,
        sub_scores={"avg_squareness_deg": round(avg_squareness, 1)},
        schema_version=SCHEMA_VERSION,
    )
