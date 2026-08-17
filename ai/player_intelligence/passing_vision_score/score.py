"""
Passing Vision Score implementation.
Analysis Logic Design v3 (docs/data_analysis.md §8).
"""

from __future__ import annotations

import math

from ai.computer_vision.tactical_analysis.constants import (
    MIN_SAMPLE_EVENTS,
    PASS_DIRECTION_SECTORS,
    SCHEMA_VERSION,
    TEAM_ASSIGNMENT_CONFIDENCE_MIN,
)
from ai.computer_vision.tactical_analysis.utils import safe_ratio


def score_passing_vision(
    completed_passes: int = 0,
    turnovers_lost: int = 0,
    pass_sector_counts: list[int] | None = None,
    forward_passes: int | None = 0,
    team_assignment_confidence: float = 0.0,
) -> dict:
    """
    Computes Passing Vision Score per docs/data_analysis.md §8.
    Gated on team_assignment_confidence >= TEAM_ASSIGNMENT_CONFIDENCE_MIN
    (0.5) -- NOT just because team info is generally nice to have, but
    because completed_passes/turnovers_lost themselves only classify
    """
    if team_assignment_confidence < TEAM_ASSIGNMENT_CONFIDENCE_MIN:
        return {
            "metric_name": "passing_vision_score",
            "value": None,
            "method": "heuristic_proxy",
            "confidence": "low_upstream_confidence",
            "sample_size": completed_passes + turnovers_lost,
            "sub_scores": {
                "completion_rate": None,
                "angle_diversity": None,
                "forward_pass_ratio": None,
            },
            "schema_version": SCHEMA_VERSION,
        }

    completion_raw = safe_ratio(completed_passes, completed_passes + turnovers_lost)
    completion_rate_score = 100.0 * completion_raw if completion_raw is not None else None

    if pass_sector_counts and sum(pass_sector_counts) > 0:
        total = sum(pass_sector_counts)
        entropy = -sum(
            (c / total) * math.log(c / total) for c in pass_sector_counts if c > 0
        )
        entropy = max(0.0, entropy)
        max_entropy = math.log(PASS_DIRECTION_SECTORS)
        angle_diversity_score = 100.0 * (entropy / max_entropy) if max_entropy > 0 else None
    else:
        angle_diversity_score = None

    if forward_passes is None:
        forward_pass_ratio_score = None
    else:
        forward_raw = safe_ratio(forward_passes, completed_passes)
        forward_pass_ratio_score = 100.0 * forward_raw if forward_raw is not None else None

    components = {
        "completion_rate": (completion_rate_score, 0.40),
        "angle_diversity": (angle_diversity_score, 0.30),
        "forward_pass_ratio": (forward_pass_ratio_score, 0.30),
    }

    valid = {k: (v, w) for k, (v, w) in components.items() if v is not None}
    sample_size = completed_passes + turnovers_lost

    if not valid or sample_size < MIN_SAMPLE_EVENTS:
        final_val = None
        confidence_enum = "low_sample"
    else:
        weight_sum = sum(w for _, w in valid.values())
        final_val = sum(v * w for v, w in valid.values()) / weight_sum
        confidence_enum = "normal"

    return {
        "metric_name": "passing_vision_score",
        "value": round(final_val, 1) if final_val is not None else None,
        "method": "heuristic_proxy",
        "confidence": confidence_enum,
        "sample_size": sample_size,
        "sub_scores": {
            "completion_rate": round(completion_rate_score, 1) if completion_rate_score is not None else None,
            "angle_diversity": round(angle_diversity_score, 1) if angle_diversity_score is not None else None,
            "forward_pass_ratio": round(forward_pass_ratio_score, 1) if forward_pass_ratio_score is not None else None,
        },
        "schema_version": SCHEMA_VERSION,
    }
