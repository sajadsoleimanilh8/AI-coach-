"""
Scanning Behavior Score.
"""

from __future__ import annotations

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
    """
    if len(orientation_readings) < 2:
        return 0

    scans = 0
    for (t_prev, angle_prev), (t_next, angle_next) in zip(orientation_readings, orientation_readings[1:]):
        gap_s = t_next - t_prev
        if gap_s <= 0 or gap_s > SCAN_MAX_GAP_S:
            continue
        if _angular_delta_deg(angle_prev, angle_next) >= SCAN_ANGLE_THRESHOLD_DEG:
            scans += 1
    return scans


def score_scanning_behavior(
    orientation_readings: list[tuple[float, float]] | None = None,
    tracked_duration_s: float = 0.0,
) -> dict:
    """
    Computes Scanning Behavior Score.
    """
    orientation_readings = sorted(orientation_readings or [], key=lambda r: r[0])
    scan_count = count_scans(orientation_readings)

    sample_size = len(orientation_readings)

    minutes = safe_ratio(tracked_duration_s, 60.0, default=None)
    scans_per_minute = safe_ratio(scan_count, minutes, default=None)

    if scans_per_minute is None or sample_size < MIN_SAMPLE_EVENTS:
        return {
            "metric_name": "scanning_behavior_score",
            "value": None,
            "method": "heuristic_proxy",
            "confidence": "low_sample",
            "sample_size": sample_size,
            "sub_scores": {
                "scan_count": scan_count,
                "scans_per_minute": round(scans_per_minute, 2) if scans_per_minute is not None else None,
            },
            "schema_version": SCHEMA_VERSION,
        }

    value = round(100.0 * clip(scans_per_minute / SCANS_PER_MINUTE_TARGET, 0.0, 1.0), 1)

    return {
        "metric_name": "scanning_behavior_score",
        "value": value,
        "method": "heuristic_proxy",
        "confidence": "normal",
        "sample_size": sample_size,
        "sub_scores": {
            "scan_count": scan_count,
            "scans_per_minute": round(scans_per_minute, 2),
        },
        "schema_version": SCHEMA_VERSION,
    }
