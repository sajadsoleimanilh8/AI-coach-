"""
Scanning Behavior Score.

Grounded in real sports-science literature (McGuckian et al.'s work on
"visual exploratory activity" / scanning frequency in football) rather
than invented for this project: elite players check their surroundings
(shoulder/head rotation) at a measurably higher rate than lower-level
players, particularly off the ball. This module counts scan events from
consecutive body_orientation_deg pose readings and rates per minute.

Reads: PlayerTracking.body_orientation_deg / body_orientation_confidence
    time series (see ai/computer_vision/pose_estimation/pose.py for how
    these are produced, ai/computer_vision/player_tracking/trajectory.py's
    attach_body_orientation() for how they land on a trajectory).
Writes: PlayerMetric(metric_name="scanning_behavior_score", ...).

Honesty notes (matching this codebase's existing scoring modules):
  - A "scan" requires two CONSECUTIVE usable readings (both above
    MIN_LANDMARK_VISIBILITY, enforced upstream by
    pose.shoulder_line_angle() already returning None below that
    threshold -- this module trusts that gate, it doesn't re-check
    visibility itself).
  - Readings more than SCAN_MAX_GAP_S apart are not compared -- a large
    gap usually means pose wasn't sampled that frame (see
    POSE_SAMPLE_STRIDE in backend/pipeline/runner.py) or play was
    stopped, not one continuous head-check motion.
  - SCANS_PER_MINUTE_TARGET is an uncalibrated placeholder (see
    constants.py's comment) -- ranking is meaningful, the absolute number
    is not yet validated against real footage.
"""

from __future__ import annotations

from ai.common.metrics import MetricResult, metric_result
from ai.computer_vision.tactical_analysis.constants import (
    MIN_SAMPLE_EVENTS,
    SCAN_ANGLE_THRESHOLD_DEG,
    SCAN_MAX_GAP_S,
    SCANS_PER_MINUTE_TARGET,
    SCHEMA_VERSION,
)
from ai.computer_vision.tactical_analysis.utils import clip, safe_ratio


def _angular_delta_deg(a: float, b: float) -> float:
    """Smallest angular difference between two 0-360 angles, e.g. the
    difference between 350deg and 10deg is 20deg, not 340deg."""
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def count_scans(orientation_readings: list[tuple[float, float]]) -> int:
    """
    Args:
        orientation_readings: chronological list of (timestamp_s, orientation_deg)
            for ONE player, already filtered to only entries where a real
            (non-None) orientation was measured -- callers should not pass
            unmeasured (None) readings in here.

    Returns:
        Number of consecutive-pair transitions where the angular change
        exceeds SCAN_ANGLE_THRESHOLD_DEG within SCAN_MAX_GAP_S.
    """
    if len(orientation_readings) < 2:
        return 0

    scans = 0
    for (t_prev, angle_prev), (t_next, angle_next) in zip(orientation_readings, orientation_readings[1:]):
        gap_s = t_next - t_prev
        if gap_s <= 0 or gap_s > SCAN_MAX_GAP_S:
            continue  # not a continuous reading pair -- see module docstring
        if _angular_delta_deg(angle_prev, angle_next) >= SCAN_ANGLE_THRESHOLD_DEG:
            scans += 1
    return scans


def score_scanning_behavior(
    orientation_readings: list[tuple[float, float]] | None = None,
    tracked_duration_s: float = 0.0,
) -> MetricResult:
    """
    Computes Scanning Behavior Score.

    Args:
        orientation_readings: chronological (timestamp_s, orientation_deg)
            pairs for this player, already filtered to real (non-None)
            readings only (see count_scans()'s docstring).
        tracked_duration_s: total time this player was tracked with at
            least occasional pose sampling -- the denominator for a
            per-minute rate. Not the same as match duration; a player who
            was only pose-sampled for 3 minutes of a 90-minute match
            should be rated on those 3 minutes, not silently diluted
            against 90.

    Returns:
        PlayerMetric dict per the Standard Output Contract.
    """
    orientation_readings = sorted(orientation_readings or [], key=lambda r: r[0])
    scan_count = count_scans(orientation_readings)

    # sample_size here is the number of usable orientation readings, not
    # the scan count -- this is what tells a consumer "how much pose data
    # actually went into this," matching first_touch_score's sample_size
    # meaning ("raw event count behind this score"), not conflating it
    # with the derived scan count.
    sample_size = len(orientation_readings)

    minutes = safe_ratio(tracked_duration_s, 60.0, default=None)
    scans_per_minute = safe_ratio(scan_count, minutes, default=None)

    if scans_per_minute is None or sample_size < MIN_SAMPLE_EVENTS:
        return metric_result(
            "scanning_behavior_score",
            None,
            method="heuristic_proxy",
            confidence="low_sample",
            sample_size=sample_size,
            sub_scores={
                "scan_count": scan_count,
                "scans_per_minute": round(scans_per_minute, 2) if scans_per_minute is not None else None,
            },
            schema_version=SCHEMA_VERSION,
        )

    value = round(100.0 * clip(scans_per_minute / SCANS_PER_MINUTE_TARGET, 0.0, 1.0), 1)

    return metric_result(
        "scanning_behavior_score",
        value,
        method="heuristic_proxy",
        confidence="normal",
        sample_size=sample_size,
        sub_scores={
            "scan_count": scan_count,
            "scans_per_minute": round(scans_per_minute, 2),
        },
        schema_version=SCHEMA_VERSION,
    )
