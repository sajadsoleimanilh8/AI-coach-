"""
Team Shape metrics: Compactness and Formation Stability.
Analysis Logic Design v3 & Implementation Spec §1.2.
"""

from __future__ import annotations
import numpy as np
from scipy.spatial import ConvexHull

from ai.computer_vision.tactical_analysis.constants import SCHEMA_VERSION, TEAM_ASSIGNMENT_CONFIDENCE_MIN
from ai.computer_vision.tactical_analysis.utils import clip


def compute_compactness(
    player_positions_m: list[tuple[float, float]],
    team_assignment_confidence: float = 0.0,
) -> dict:
    """
    Computes compactness via Convex Hull area of outfield players.
    """
    if team_assignment_confidence < TEAM_ASSIGNMENT_CONFIDENCE_MIN or len(player_positions_m) < 3:
        return {
            "metric_name": "compactness_score",
            "value": None,
            "method": "deterministic",
            "confidence": "low_upstream_confidence" if team_assignment_confidence < TEAM_ASSIGNMENT_CONFIDENCE_MIN else "low_sample",
            "sample_size": len(player_positions_m),
            "sub_scores": {"hull_area_m2": None},
            "schema_version": SCHEMA_VERSION,
        }

    pts = np.array(player_positions_m, dtype=np.float64)
    try:
        hull = ConvexHull(pts)
        area_m2 = float(hull.volume)
    except Exception:
        area_m2 = 1000.0

    norm_score = clip(100.0 * (1.0 - (area_m2 - 60.0) / (2000.0 - 60.0)), 0.0, 100.0)

    return {
        "metric_name": "compactness_score",
        "value": round(norm_score, 1),
        "method": "deterministic",
        "confidence": "normal",
        "sample_size": len(player_positions_m),
        "sub_scores": {
            "hull_area_m2": round(area_m2, 1),
            "raw_compactness": round(norm_score, 1),
        },
        "schema_version": SCHEMA_VERSION,
    }


def compute_formation_stability(
    frames_player_positions: list[list[tuple[float, float]]],
    team_assignment_confidence: float = 0.0,
) -> dict:
    """
    Computes formation stability as frame-to-frame variance of player positions relative to team centroid.
    """
    if team_assignment_confidence < TEAM_ASSIGNMENT_CONFIDENCE_MIN or len(frames_player_positions) < 2:
        return {
            "metric_name": "formation_stability_score",
            "value": None,
            "method": "deterministic",
            "confidence": "low_upstream_confidence" if team_assignment_confidence < TEAM_ASSIGNMENT_CONFIDENCE_MIN else "low_sample",
            "sample_size": len(frames_player_positions),
            "sub_scores": {"position_variance_m2": None},
            "schema_version": SCHEMA_VERSION,
        }

    all_relative_vectors = []
    for frame_pts in frames_player_positions:
        if len(frame_pts) == 0:
            continue
        pts = np.array(frame_pts, dtype=np.float64)
        centroid = np.mean(pts, axis=0)
        rel = pts - centroid
        all_relative_vectors.append(rel)

    if len(all_relative_vectors) == 0:
        return {
            "metric_name": "formation_stability_score",
            "value": None,
            "method": "deterministic",
            "confidence": "low_sample",
            "sample_size": 0,
            "sub_scores": {"position_variance_m2": None},
            "schema_version": SCHEMA_VERSION,
        }

    variances = [float(np.var(rel, axis=0).sum()) for rel in all_relative_vectors]
    mean_variance = float(np.mean(variances))

    stability_score = clip(100.0 * (1.0 - mean_variance / 25.0), 0.0, 100.0)

    return {
        "metric_name": "formation_stability_score",
        "value": round(stability_score, 1),
        "method": "deterministic",
        "confidence": "normal",
        "sample_size": len(frames_player_positions),
        "sub_scores": {
            "position_variance_m2": round(mean_variance, 2),
            "raw_stability": round(stability_score, 1),
        },
        "schema_version": SCHEMA_VERSION,
    }
