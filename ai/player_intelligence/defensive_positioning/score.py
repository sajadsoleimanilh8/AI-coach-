"""
Defensive Positioning Score implementation (§3.2 of Phase 3 Spec).
"""

from __future__ import annotations

from ai.common.metrics import Confidence, MetricResult, metric_result
from ai.computer_vision.tactical_analysis.constants import (
    IDEAL_DEFENSIVE_SPACING_M,
    MAX_ACCEPTABLE_LINE_DEVIATION_M,
    MIN_SAMPLE_EVENTS,
    REACTION_SPEED_REFERENCE_MS,
    SCHEMA_VERSION,
    SPACING_TOLERANCE_M,
    TEAM_ASSIGNMENT_CONFIDENCE_MIN,
)
from ai.computer_vision.tactical_analysis.utils import clip, gaussian_score, safe_ratio


def score_defensive_positioning(
    avg_deviation_from_defensive_line_m: float = 0.0,
    nearest_teammate_distance_m: float | None = None,
    closing_speed_toward_threat_ms: float | None = None,
    sample_size_defensive_phases: int = 0,
    team_assignment_confidence: float = 0.0,
) -> MetricResult:
    """
    Computes Defensive Positioning Score per §3.2.
    Gated on team_assignment_confidence >= TEAM_ASSIGNMENT_CONFIDENCE_MIN (0.5).

    Returns PlayerMetric dict.
    """
    if team_assignment_confidence < TEAM_ASSIGNMENT_CONFIDENCE_MIN:
        return metric_result(
            "defensive_positioning_score",
            None,
            method="heuristic_proxy",
            confidence="low_upstream_confidence",
            sample_size=sample_size_defensive_phases,
            sub_scores={
                "line_discipline": None,
                "spacing": None,
                "reaction": None,
            },
            schema_version=SCHEMA_VERSION,
        )

    # 1. Line Discipline (40%)
    line_discipline_score = 100.0 * (
        1.0 - clip(avg_deviation_from_defensive_line_m / MAX_ACCEPTABLE_LINE_DEVIATION_M, 0.0, 1.0)
    )

    # 2. Cover / Balance Spacing (35%)
    spacing_score = gaussian_score(
        nearest_teammate_distance_m, ideal=IDEAL_DEFENSIVE_SPACING_M, tolerance=SPACING_TOLERANCE_M
    ) if nearest_teammate_distance_m is not None else None

    # 3. Reaction to Threat (25%)
    reaction_raw = safe_ratio(closing_speed_toward_threat_ms, REACTION_SPEED_REFERENCE_MS, default=None)
    reaction_score = 100.0 * clip(reaction_raw, 0.0, 1.0) if reaction_raw is not None else None

    components = {
        "line_discipline": (line_discipline_score, 0.40),
        "spacing": (spacing_score, 0.35),
        "reaction": (reaction_score, 0.25),
    }

    valid = {k: (v, w) for k, (v, w) in components.items() if v is not None}

    if not valid or sample_size_defensive_phases < MIN_SAMPLE_EVENTS:
        final_val = None
        confidence_enum: Confidence = "low_sample"
    else:
        weight_sum = sum(w for _, w in valid.values())
        final_val = sum(v * w for v, w in valid.values()) / weight_sum
        confidence_enum = "normal"

    return metric_result(
        "defensive_positioning_score",
        round(final_val, 1) if final_val is not None else None,
        method="heuristic_proxy",
        confidence=confidence_enum,
        sample_size=sample_size_defensive_phases,
        sub_scores={
            "line_discipline": round(line_discipline_score, 1) if line_discipline_score is not None else None,
            "spacing": round(spacing_score, 1) if spacing_score is not None else None,
            "reaction": round(reaction_score, 1) if reaction_score is not None else None,
        },
        schema_version=SCHEMA_VERSION,
    )
