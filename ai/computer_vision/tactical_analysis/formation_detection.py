"""
Formation Detection module.
Uses Hungarian algorithm (scipy.optimize.linear_sum_assignment) to match average outfield player positions
against candidate formation templates.
Analysis Logic Design v3 (docs/data_analysis.md §3).
"""

from __future__ import annotations
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

from ai.computer_vision.tactical_analysis.constants import (
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    SCHEMA_VERSION,
    TEAM_ASSIGNMENT_CONFIDENCE_MIN,
)
from ai.computer_vision.tactical_analysis.formation_templates import FORMATION_TEMPLATES

def _derive_normalization_constant() -> float:
    """The slot-ambiguity radius, in metres, derived from the templates."""
    gaps: list[float] = []
    for slots in FORMATION_TEMPLATES.values():
        pts = np.array([(x * PITCH_LENGTH_M, y * PITCH_WIDTH_M) for x, y in slots],
                       dtype=np.float64)
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        gaps.extend(d.min(axis=1).tolist())
    return float(np.median(gaps) / 2.0)


NORMALIZATION_CONSTANT = _derive_normalization_constant()


def detect_formation(
    player_positions: list[tuple[float, float]],
    team_assignment_confidence: float = 0.0,
    low_confidence_player_count: int = 0,
    normalization_constant: float = NORMALIZATION_CONSTANT,
    attacking_direction: str = "left_to_right",
) -> dict:
    """
    Detects the best matching formation for a given set of average outfield player positions (pitch_x_m, pitch_y_m).
    """
    if team_assignment_confidence < TEAM_ASSIGNMENT_CONFIDENCE_MIN or low_confidence_player_count > 2:
        return {
            "metric_name": "formation",
            "value": None,
            "method": "heuristic_proxy",
            "confidence": "low_upstream_confidence",
            "confidence_score": None,
            "sample_size": len(player_positions),
            "sub_scores": {"reason": "low_team_assignment_confidence"},
            "schema_version": SCHEMA_VERSION,
        }

    n_players = len(player_positions)
    if n_players < 8:
        return {
            "metric_name": "formation",
            "value": None,
            "method": "heuristic_proxy",
            "confidence": "low_sample",
            "confidence_score": None,
            "sample_size": n_players,
            "sub_scores": {"reason": "insufficient_outfield_players"},
            "schema_version": SCHEMA_VERSION,
        }

    player_pts = np.array(player_positions, dtype=np.float64)

    best_template = None
    best_confidence_score = -1.0
    best_distance_score = float("inf")

    for name, norm_slots in FORMATION_TEMPLATES.items():
        slots_m = np.array(
            [(x_frac * PITCH_LENGTH_M, y_frac * PITCH_WIDTH_M) for x_frac, y_frac in norm_slots],
            dtype=np.float64,
        )

        if attacking_direction == "right_to_left":
            slots_m[:, 0] = PITCH_LENGTH_M - slots_m[:, 0]

        target_slots = slots_m[:n_players] if len(slots_m) >= n_players else slots_m

        cost_matrix = np.linalg.norm(player_pts[:, None, :] - target_slots[None, :, :], axis=2)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        distance_score = float(cost_matrix[row_ind, col_ind].sum())

        confidence_score = 100.0 * math.exp(-distance_score / (n_players * normalization_constant))

        if confidence_score > best_confidence_score:
            best_confidence_score = confidence_score
            best_template = name
            best_distance_score = distance_score

    confidence_enum = "normal"

    return {
        "metric_name": "formation",
        "value": best_template,
        "method": "heuristic_proxy",
        "confidence": confidence_enum,
        "confidence_score": round(best_confidence_score / 100.0, 4),
        "sample_size": n_players,
        "sub_scores": {
            "template_matching_distance": round(best_distance_score, 2),
            "matching_confidence_pct": round(best_confidence_score, 2),
        },
        "schema_version": SCHEMA_VERSION,
    }
