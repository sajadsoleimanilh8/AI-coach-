"""
Team Shape metrics: Compactness and Formation Stability.
Analysis Logic Design v3 & Implementation Spec §1.2.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from ai.common.metrics import MetricResult, metric_result
from ai.computer_vision.tactical_analysis.constants import SCHEMA_VERSION, TEAM_ASSIGNMENT_CONFIDENCE_MIN
from ai.computer_vision.tactical_analysis.utils import clip


def compute_compactness(
    player_positions_m: list[tuple[float, float]],
    team_assignment_confidence: float = 0.0,
) -> MetricResult:
    """
    Computes compactness via Convex Hull area of outfield players.

    Returns TeamMetric dict with metric_name="compactness_score".
    """
    if team_assignment_confidence < TEAM_ASSIGNMENT_CONFIDENCE_MIN or len(player_positions_m) < 3:
        return metric_result(
            "compactness_score",
            None,
            method="deterministic",
            confidence="low_upstream_confidence" if team_assignment_confidence < TEAM_ASSIGNMENT_CONFIDENCE_MIN else "low_sample",
            sample_size=len(player_positions_m),
            sub_scores={"hull_area_m2": None},
            schema_version=SCHEMA_VERSION,
        )

    pts = np.array(player_positions_m, dtype=np.float64)
    try:
        hull = ConvexHull(pts)
        area_m2 = float(hull.volume)  # 2D volume == area
    except Exception:  # noqa: BLE001 - degenerate hull (collinear points) uses a fallback area
        area_m2 = 1000.0

    # FIFA pitch reference bounds: 60 m² (extremely compact) to 2000 m² (spread out)
    # Smaller area = more compact = higher score
    norm_score = clip(100.0 * (1.0 - (area_m2 - 60.0) / (2000.0 - 60.0)), 0.0, 100.0)

    return metric_result(
        "compactness_score",
        round(norm_score, 1),
        method="deterministic",
        confidence="normal",
        sample_size=len(player_positions_m),
        sub_scores={
            "hull_area_m2": round(area_m2, 1),
            "raw_compactness": round(norm_score, 1),
        },
        schema_version=SCHEMA_VERSION,
    )


def compute_formation_stability(
    frames_player_positions: list[list[tuple[float, float]]],
    team_assignment_confidence: float = 0.0,
) -> MetricResult:
    """
    Computes formation stability as frame-to-frame variance of player positions relative to team centroid.

    Returns TeamMetric dict with metric_name="formation_stability_score".
    """
    if team_assignment_confidence < TEAM_ASSIGNMENT_CONFIDENCE_MIN or len(frames_player_positions) < 2:
        return metric_result(
            "formation_stability_score",
            None,
            method="deterministic",
            confidence="low_upstream_confidence" if team_assignment_confidence < TEAM_ASSIGNMENT_CONFIDENCE_MIN else "low_sample",
            sample_size=len(frames_player_positions),
            sub_scores={"position_variance_m2": None},
            schema_version=SCHEMA_VERSION,
        )

    # For each frame, compute relative vectors (x - cx, y - cy)
    all_relative_vectors = []
    for frame_pts in frames_player_positions:
        if len(frame_pts) == 0:
            continue
        pts = np.array(frame_pts, dtype=np.float64)
        centroid = np.mean(pts, axis=0)
        rel = pts - centroid
        all_relative_vectors.append(rel)

    if len(all_relative_vectors) == 0:
        return metric_result(
            "formation_stability_score",
            None,
            method="deterministic",
            confidence="low_sample",
            sample_size=0,
            sub_scores={"position_variance_m2": None},
            schema_version=SCHEMA_VERSION,
        )

    # Concatenate variances
    variances = [float(np.var(rel, axis=0).sum()) for rel in all_relative_vectors]
    mean_variance = float(np.mean(variances))

    # Scale to 0-100 score: variance of 0 = 100 stability, variance >= 25 m² = 0 stability
    stability_score = clip(100.0 * (1.0 - mean_variance / 25.0), 0.0, 100.0)

    return metric_result(
        "formation_stability_score",
        round(stability_score, 1),
        method="deterministic",
        confidence="normal",
        sample_size=len(frames_player_positions),
        sub_scores={
            "position_variance_m2": round(mean_variance, 2),
            "raw_stability": round(stability_score, 1),
        },
        schema_version=SCHEMA_VERSION,
    )
