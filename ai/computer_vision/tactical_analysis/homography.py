"""
Camera (pixel) -> pitch (meters) homography.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class HomographyResult:
    """Everything downstream code needs to trust (or distrust) this transform."""

    H: np.ndarray
    reprojection_error_m: float
    max_reprojection_error_m: float
    confidence: float
    n_points: int
    per_point_errors_m: list = field(default_factory=list)
    n_inliers: int | None = None
    inlier_ratio: float | None = None
    inlier_reprojection_error_m: float | None = None
    inlier_mask: list = field(default_factory=list)


def compute_homography(
    pixel_pts: np.ndarray | list,
    pitch_pts: np.ndarray | list,
    method: int = 0,
    ransac_reproj_threshold: float = 3.0,
) -> HomographyResult:
    """
    Fit a pixel -> pitch-meters homography from corresponding point pairs.
    """
    pixel_pts = np.asarray(pixel_pts, dtype=np.float64).reshape(-1, 2)
    pitch_pts = np.asarray(pitch_pts, dtype=np.float64).reshape(-1, 2)

    if len(pixel_pts) != len(pitch_pts):
        raise ValueError(
            f"pixel_pts ({len(pixel_pts)}) and pitch_pts ({len(pitch_pts)}) "
            "must have the same number of points"
        )
    if len(pixel_pts) < 4:
        raise ValueError(
            f"Need at least 4 point correspondences to fit a homography, got {len(pixel_pts)}"
        )

    kwargs = {}
    if method == cv2.RANSAC:
        kwargs["ransacReprojThreshold"] = ransac_reproj_threshold

    H, mask = cv2.findHomography(pixel_pts, pitch_pts, method=method, **kwargs)

    if H is None:
        raise ValueError(
            "cv2.findHomography failed to fit a matrix -- check that points "
            "aren't collinear or duplicated"
        )

    projected = _apply_homography(pixel_pts, H)
    per_point_errors = np.linalg.norm(projected - pitch_pts, axis=1)

    mean_error = float(np.mean(per_point_errors))
    max_error = float(np.max(per_point_errors))

    n_inliers: int | None = None
    inlier_ratio: float | None = None
    inlier_error: float | None = None
    inlier_mask: list = []
    if method == cv2.RANSAC and mask is not None:
        inliers = mask.reshape(-1).astype(bool)
        inlier_mask = inliers.tolist()
        n_inliers = int(inliers.sum())
        inlier_ratio = float(n_inliers) / float(len(pixel_pts))
        inlier_error = (float(np.mean(per_point_errors[inliers]))
                        if n_inliers > 0 else mean_error)

    scored_error = inlier_error if inlier_error is not None else mean_error

    return HomographyResult(
        H=H,
        reprojection_error_m=mean_error,
        max_reprojection_error_m=max_error,
        confidence=homography_confidence(scored_error),
        n_points=len(pixel_pts),
        per_point_errors_m=per_point_errors.tolist(),
        n_inliers=n_inliers,
        inlier_ratio=inlier_ratio,
        inlier_reprojection_error_m=(inlier_error if inlier_error is not None
                                     else mean_error),
        inlier_mask=inlier_mask,
    )


def _apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to an (N, 2) array of points."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points, H)
    return transformed.reshape(-1, 2)


def pixel_to_pitch(x: float, y: float, H: np.ndarray) -> tuple[float, float]:
    """
    Transform a single pixel coordinate to pitch meters.
    """
    result = _apply_homography(np.array([[x, y]]), H)
    px, py = result[0]
    return float(px), float(py)


def pixels_to_pitch(points: np.ndarray | list, H: np.ndarray) -> np.ndarray:
    """
    Batch version of pixel_to_pitch. Use this for whole tracking arrays
    (e.g. all player positions in a frame, or a full trajectory) instead of
    looping pixel_to_pitch() point by point -- one cv2 call instead of N.
    """
    return _apply_homography(np.asarray(points, dtype=np.float64), H)


def point_spread(
    pixel_pts: np.ndarray | list, frame_size: tuple[int, int]
) -> float:
    """
    Convex-hull area of the calibration points, as a fraction of frame area.
    """
    width, height = frame_size
    if width <= 0 or height <= 0:
        return 0.0

    pts = np.asarray(pixel_pts, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0

    try:
        hull = cv2.convexHull(pts)
    except cv2.error:
        return 0.0

    return float(cv2.contourArea(hull)) / float(width * height)


def homography_geometry_problems(
    H: np.ndarray,
    frame_size: tuple[int, int],
    keypoints_px: np.ndarray | list | None = None,
) -> list[str]:
    """
    Structural sanity checks on a fitted pixel -> pitch homography.
    """
    from ai.computer_vision.tactical_analysis.constants import (
        HOMOGRAPHY_MAX_OUT_OF_BOUNDS_M,
        HOMOGRAPHY_MAX_VIEW_SPAN_M,
        HOMOGRAPHY_MIN_DETERMINANT,
        HOMOGRAPHY_MIN_VIEW_SPAN_M,
        HOMOGRAPHY_REJECT_MIRRORED,
        PITCH_LENGTH_M,
        PITCH_WIDTH_M,
    )

    problems: list[str] = []
    width, height = frame_size
    if width <= 0 or height <= 0:
        return ["frame size is non-positive, geometry cannot be checked"]

    matrix = np.asarray(H, dtype=np.float64)
    if matrix.shape != (3, 3):
        return [f"homography is not 3x3 (shape {matrix.shape})"]
    if not np.isfinite(matrix).all():
        return ["homography contains non-finite entries"]

    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < HOMOGRAPHY_MIN_DETERMINANT:
        return [f"degenerate homography: |det| {abs(determinant):.3e} < "
                f"{HOMOGRAPHY_MIN_DETERMINANT:.0e}, the matrix collapses the "
                "image plane and is not invertible"]

    probe_pts = None
    if keypoints_px is not None:
        pts = np.asarray(keypoints_px, dtype=np.float64).reshape(-1, 2)
        if len(pts) >= 3:
            probe_pts = pts
    if probe_pts is not None:
        x0, y0 = float(probe_pts[:, 0].min()), float(probe_pts[:, 1].min())
        x1, y1 = float(probe_pts[:, 0].max()), float(probe_pts[:, 1].max())
        region = "the calibration-keypoint region"
    else:
        x0, x1 = width * 0.2, width * 0.8
        y0, y1 = height * 0.2, height * 0.8
        region = "the frame's central region"
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return ["calibration keypoints span no area, geometry cannot be checked"]

    corners = np.array([[x0, y1], [x1, y1], [x1, y0], [x0, y0]], dtype=np.float64)
    try:
        projected = _apply_homography(corners, matrix)
    except cv2.error as exc:
        return [f"projecting the probe region failed: {exc}"]
    if not np.isfinite(projected).all():
        return [f"{region} projects to non-finite pitch coordinates -- the "
                "homography maps part of it through its horizon"]

    shoelace = 0.0
    for i in range(4):
        x1, y1 = projected[i]
        x2, y2 = projected[(i + 1) % 4]
        shoelace += x1 * y2 - x2 * y1
    signed_area = shoelace / 2.0
    if HOMOGRAPHY_REJECT_MIRRORED and signed_area > 0:
        problems.append(
            f"mirrored pitch geometry: {region} projects with reversed "
            "winding, so pitch x/y are flipped relative to the camera")

    span_x = float(projected[:, 0].max() - projected[:, 0].min())
    span_y = float(projected[:, 1].max() - projected[:, 1].min())
    span = max(span_x, span_y)
    if span < HOMOGRAPHY_MIN_VIEW_SPAN_M:
        problems.append(
            f"implausible projected scale: {region} maps to {span:.1f} m "
            f"< {HOMOGRAPHY_MIN_VIEW_SPAN_M} m, a collapsed fit")
    elif span > HOMOGRAPHY_MAX_VIEW_SPAN_M:
        problems.append(
            f"implausible projected scale: {region} maps to {span:.1f} m "
            f"> {HOMOGRAPHY_MAX_VIEW_SPAN_M} m, a near-degenerate fit "
            "projecting toward the horizon")

    overshoot = max(
        float(-projected[:, 0].min()),
        float(projected[:, 0].max() - PITCH_LENGTH_M),
        float(-projected[:, 1].min()),
        float(projected[:, 1].max() - PITCH_WIDTH_M),
    )
    if overshoot > HOMOGRAPHY_MAX_OUT_OF_BOUNDS_M:
        problems.append(
            f"projected pitch outside plausible bounds: {region} lands up "
            f"to {overshoot:.0f} m beyond the pitch rectangle "
            f"(> {HOMOGRAPHY_MAX_OUT_OF_BOUNDS_M} m)")

    crosses = []
    for i in range(4):
        a = projected[(i + 1) % 4] - projected[i]
        b = projected[(i + 2) % 4] - projected[(i + 1) % 4]
        crosses.append(a[0] * b[1] - a[1] * b[0])
    if not (all(c >= 0 for c in crosses) or all(c <= 0 for c in crosses)):
        problems.append(
            "projected frame is not convex -- the homography folds the image "
            "across its horizon line")

    return problems


def homography_confidence(reprojection_error_m: float, scale_m: float = 2.0) -> float:
    """
    Map reprojection error (pitch meters) to a 0-1 confidence score.
    """
    if reprojection_error_m < 0:
        raise ValueError("reprojection_error_m cannot be negative")
    confidence = math.exp(-reprojection_error_m / scale_m)
    return max(0.0, min(1.0, confidence))
