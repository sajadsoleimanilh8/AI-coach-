"""
The 32-keypoint pitch model: keypoint index -> pitch coordinate in metres.
"""

from __future__ import annotations

import math

from ai.computer_vision.tactical_analysis.constants import PITCH_LENGTH_M, PITCH_WIDTH_M

L = PITCH_LENGTH_M
W = PITCH_WIDTH_M
_CY = W / 2.0

PENALTY_AREA_DEPTH_M = 16.5
_PA_TOP, _PA_BOT = _CY - 20.16, _CY + 20.16
GOAL_AREA_DEPTH_M = 5.5
_GA_TOP, _GA_BOT = _CY - 9.16, _CY + 9.16
PENALTY_SPOT_DIST_M = 11.0
CENTRE_CIRCLE_RADIUS_M = 9.15
_ARC = math.sqrt(CENTRE_CIRCLE_RADIUS_M ** 2
                 - (PENALTY_AREA_DEPTH_M - PENALTY_SPOT_DIST_M) ** 2)

PITCH_KEYPOINTS_32: dict[int, tuple[float, float]] = {
    0: (0.0, 0.0),
    1: (0.0, _PA_TOP),
    2: (0.0, _GA_TOP),
    3: (0.0, _GA_BOT),
    4: (0.0, _PA_BOT),
    5: (0.0, W),
    6: (GOAL_AREA_DEPTH_M, _GA_TOP),
    7: (GOAL_AREA_DEPTH_M, _GA_BOT),
    8: (PENALTY_SPOT_DIST_M, _CY),
    9: (PENALTY_AREA_DEPTH_M, _PA_TOP),
    10: (PENALTY_AREA_DEPTH_M, _CY - _ARC),
    11: (PENALTY_AREA_DEPTH_M, _CY + _ARC),
    12: (PENALTY_AREA_DEPTH_M, _PA_BOT),
    13: (L / 2, 0.0),
    14: (L / 2, _CY - CENTRE_CIRCLE_RADIUS_M),
    15: (L / 2, _CY + CENTRE_CIRCLE_RADIUS_M),
    16: (L / 2, W),
    17: (L - PENALTY_AREA_DEPTH_M, _PA_TOP),
    18: (L - PENALTY_AREA_DEPTH_M, _CY - _ARC),
    19: (L - PENALTY_AREA_DEPTH_M, _CY + _ARC),
    20: (L - PENALTY_AREA_DEPTH_M, _PA_BOT),
    21: (L - PENALTY_SPOT_DIST_M, _CY),
    22: (L - GOAL_AREA_DEPTH_M, _GA_TOP),
    23: (L - GOAL_AREA_DEPTH_M, _GA_BOT),
    24: (L, 0.0),
    25: (L, _PA_TOP),
    26: (L, _GA_TOP),
    27: (L, _GA_BOT),
    28: (L, _PA_BOT),
    29: (L, W),
    30: (L / 2 - CENTRE_CIRCLE_RADIUS_M, _CY),
    31: (L / 2 + CENTRE_CIRCLE_RADIUS_M, _CY),
}

PITCH_KEYPOINT_NAMES: dict[int, str] = {
    0: "left_corner_far", 1: "left_goalline_penalty_top",
    2: "left_goalline_goalarea_top", 3: "left_goalline_goalarea_bottom",
    4: "left_goalline_penalty_bottom", 5: "left_corner_near",
    6: "left_goalarea_top_inner", 7: "left_goalarea_bottom_inner",
    8: "left_penalty_spot",
    9: "left_penaltyarea_top_inner", 10: "left_penalty_arc_top",
    11: "left_penalty_arc_bottom", 12: "left_penaltyarea_bottom_inner",
    13: "halfway_far", 14: "centre_circle_top",
    15: "centre_circle_bottom", 16: "halfway_near",
    17: "right_penaltyarea_top_inner", 18: "right_penalty_arc_top",
    19: "right_penalty_arc_bottom", 20: "right_penaltyarea_bottom_inner",
    21: "right_penalty_spot",
    22: "right_goalarea_top_inner", 23: "right_goalarea_bottom_inner",
    24: "right_corner_far", 25: "right_goalline_penalty_top",
    26: "right_goalline_goalarea_top", 27: "right_goalline_goalarea_bottom",
    28: "right_goalline_penalty_bottom", 29: "right_corner_near",
    30: "centre_circle_left", 31: "centre_circle_right",
}

FLIP_IDX = [24, 25, 26, 27, 28, 29, 22, 23, 21, 17, 18, 19, 20, 13, 14, 15,
            16, 9, 10, 11, 12, 8, 6, 7, 0, 1, 2, 3, 4, 5, 31, 30]


def keypoints_to_correspondences(
    keypoints: dict[int, tuple[float, float]],
    min_confidence: float = 0.0,
    confidences: dict[int, float] | None = None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[int]]:
    """
    Turns detected keypoints into (pixel_pts, pitch_pts, indices) ready for
    homography.compute_homography().
    """
    pixel_pts: list[tuple[float, float]] = []
    pitch_pts: list[tuple[float, float]] = []
    used: list[int] = []
    for idx, (px, py) in sorted(keypoints.items()):
        if idx not in PITCH_KEYPOINTS_32:
            continue
        if confidences is not None and confidences.get(idx, 0.0) < min_confidence:
            continue
        pixel_pts.append((float(px), float(py)))
        pitch_pts.append(PITCH_KEYPOINTS_32[idx])
        used.append(idx)
    return pixel_pts, pitch_pts, used


def verify_table() -> list[str]:
    """Self-check: every index present once, and the flip symmetry in
    FLIP_IDX actually mirrors this table about x = L/2. Returns a list of
    problems (empty when consistent)."""
    problems = []
    missing = set(range(32)) - set(PITCH_KEYPOINTS_32)
    if missing:
        problems.append(f"missing keypoint indices: {sorted(missing)}")
    for i, j in enumerate(FLIP_IDX):
        if i not in PITCH_KEYPOINTS_32 or j not in PITCH_KEYPOINTS_32:
            continue
        xi, yi = PITCH_KEYPOINTS_32[i]
        xj, yj = PITCH_KEYPOINTS_32[j]
        if abs((L - xi) - xj) > 1e-6 or abs(yi - yj) > 1e-6:
            problems.append(
                f"flip mismatch {i}<->{j}: ({xi:.2f},{yi:.2f}) vs ({xj:.2f},{yj:.2f})")
    return problems
