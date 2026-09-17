"""
The 32-keypoint pitch model: keypoint index -> pitch coordinate in metres.

WHY THIS FILE EXISTS
    The calibration dataset (datasets/processed/field_datasets/2 -- note the
    directory name is misleading, see configs/datasets.yaml) supplies 32
    INDEXED pitch keypoints. The existing manual flow
    (manual_calibration.py) collects 17 NAMED landmarks
    (constants.CALIBRATION_POINT_ORDER). These are different schemes with
    different orderings; there is no 1:1 correspondence.

    homography.compute_homography() needs (pixel, pitch-metre) pairs. This
    table is the missing link that turns the model's keypoint indices into
    pitch metres. Without it the calibration model's output is unusable.

HOW THIS MAPPING WAS ESTABLISHED (not guessed)
    Guessing it would have been catastrophic and silent: a wrong index
    assignment still produces a homography, just one that maps players to
    the wrong pitch locations, and reprojection error alone would not
    obviously flag it.

    Instead it was derived empirically and then verified:

    1. Keypoint indices were rendered onto a training frame and read off
       visually to identify four unambiguous anchors (both far corners, and
       the halfway line's two ends).
    2. A homography from those anchors projected every other keypoint into
       pitch metres, aggregated as a median across 23 images with all four
       anchors visible. The resulting cluster centres landed on canonical
       pitch landmarks (penalty spots, penalty-arc intersections, goal-area
       corners) to within ~1-2 m.
    3. The flip_idx published in the dataset's own data.yaml independently
       confirms the symmetry structure: 0<->24, 1<->25 ... 30<->31, with
       13/14/15/16 self-symmetric -- exactly the four halfway-line points
       this table assigns to x = 52.5.
    4. VERIFICATION: refitting with RANSAC using this full table across all
       255 training images gives a median reprojection error of
       **0.247 m** (mean 0.257 m, 90th pct 0.396 m, 100% of images under
       1.0 m). A mis-assigned index scheme cannot produce sub-metre
       agreement across 255 independent camera poses.

       Reproduce with: python -m scripts.validate_pitch_keypoints

COORDINATE FRAME
    x in [0, 105] runs goal line to goal line; y in [0, 68] runs touchline
    to touchline, y = 0 being the touchline furthest from a standard main
    camera. Matches constants.PITCH_LENGTH_M / PITCH_WIDTH_M and the frame
    already used by REFERENCE_POINTS.
"""

from __future__ import annotations

import math

from ai.computer_vision.tactical_analysis.constants import PITCH_LENGTH_M, PITCH_WIDTH_M

L = PITCH_LENGTH_M          # 105.0
W = PITCH_WIDTH_M           # 68.0
_CY = W / 2.0               # 34.0 -- pitch centre line

# Penalty area: 16.5 m deep, 40.32 m wide, centred.
PENALTY_AREA_DEPTH_M = 16.5
_PA_TOP, _PA_BOT = _CY - 20.16, _CY + 20.16          # 13.84 / 54.16
# Goal area ("6-yard box"): 5.5 m deep, 18.32 m wide, centred.
GOAL_AREA_DEPTH_M = 5.5
_GA_TOP, _GA_BOT = _CY - 9.16, _CY + 9.16            # 24.84 / 43.16
PENALTY_SPOT_DIST_M = 11.0
CENTRE_CIRCLE_RADIUS_M = 9.15
# Where the penalty arc crosses the penalty-area line:
# sqrt(r^2 - (depth - spot)^2) from the centre line.
_ARC = math.sqrt(CENTRE_CIRCLE_RADIUS_M ** 2
                 - (PENALTY_AREA_DEPTH_M - PENALTY_SPOT_DIST_M) ** 2)   # ~7.31

#: keypoint index -> (x_m, y_m)
PITCH_KEYPOINTS_32: dict[int, tuple[float, float]] = {
    # ---- left goal line (x = 0), top to bottom -----------------------
    0: (0.0, 0.0),                       # far-side corner
    1: (0.0, _PA_TOP),                   # penalty area meets goal line
    2: (0.0, _GA_TOP),                   # goal area meets goal line
    3: (0.0, _GA_BOT),
    4: (0.0, _PA_BOT),
    5: (0.0, W),                         # near-side corner
    # ---- left goal area + penalty spot -------------------------------
    6: (GOAL_AREA_DEPTH_M, _GA_TOP),
    7: (GOAL_AREA_DEPTH_M, _GA_BOT),
    8: (PENALTY_SPOT_DIST_M, _CY),       # penalty spot
    # ---- left penalty area line (x = 16.5) ---------------------------
    9: (PENALTY_AREA_DEPTH_M, _PA_TOP),
    10: (PENALTY_AREA_DEPTH_M, _CY - _ARC),   # penalty arc, top
    11: (PENALTY_AREA_DEPTH_M, _CY + _ARC),   # penalty arc, bottom
    12: (PENALTY_AREA_DEPTH_M, _PA_BOT),
    # ---- halfway line (self-symmetric under horizontal flip) ---------
    13: (L / 2, 0.0),                              # far touchline
    14: (L / 2, _CY - CENTRE_CIRCLE_RADIUS_M),     # centre circle, top
    15: (L / 2, _CY + CENTRE_CIRCLE_RADIUS_M),     # centre circle, bottom
    16: (L / 2, W),                                # near touchline
    # ---- right penalty area line (x = 88.5) --------------------------
    17: (L - PENALTY_AREA_DEPTH_M, _PA_TOP),
    18: (L - PENALTY_AREA_DEPTH_M, _CY - _ARC),
    19: (L - PENALTY_AREA_DEPTH_M, _CY + _ARC),
    20: (L - PENALTY_AREA_DEPTH_M, _PA_BOT),
    # ---- right penalty spot + goal area ------------------------------
    21: (L - PENALTY_SPOT_DIST_M, _CY),
    22: (L - GOAL_AREA_DEPTH_M, _GA_TOP),
    23: (L - GOAL_AREA_DEPTH_M, _GA_BOT),
    # ---- right goal line (x = 105) -----------------------------------
    24: (L, 0.0),
    25: (L, _PA_TOP),
    26: (L, _GA_TOP),
    27: (L, _GA_BOT),
    28: (L, _PA_BOT),
    29: (L, W),
    # ---- centre circle left/right extremes ---------------------------
    30: (L / 2 - CENTRE_CIRCLE_RADIUS_M, _CY),
    31: (L / 2 + CENTRE_CIRCLE_RADIUS_M, _CY),
}

#: Human-readable names, for the calibration debug overlay and QA reports.
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

#: Horizontal-flip symmetry, copied from the dataset's own data.yaml. Used
#: to sanity-check this table, and required for correct fliplr augmentation
#: during training (see configs/datasets.yaml).
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

    Keypoints whose index is not in PITCH_KEYPOINTS_32, or whose confidence
    is below `min_confidence`, are dropped rather than defaulted -- an
    unreliable landmark is worse than a missing one, because it biases the
    fit while looking like extra evidence.
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
