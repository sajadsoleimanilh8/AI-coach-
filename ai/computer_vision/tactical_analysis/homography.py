"""
Camera (pixel) -> pitch (meters) homography.

Implements the contract required by docs/data analysis.md (Analysis Logic
Design v3) §0.2: every homography transform must carry a confidence score
derived from reprojection error, not just report coordinates. Downstream
consumers (First Touch, Press Resistance, Formation Detection, Injury Risk)
treat pitch_x_m/pitch_y_m as unusable when homography_confidence is below
HOMOGRAPHY_CONFIDENCE_MIN (0.6) -- see constants.py.

Typical flow:
    1. Run calibrate_pitch.py once per match/camera-angle to click 4-8+
       reference points in a representative frame -> saves a calibration
       JSON with pixel_pts, pitch_pts, H, reprojection_error, confidence.
    2. Load that JSON at analysis time, call pixel_to_pitch() (or the batch
       variant) on every tracked player/ball pixel position for that match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class HomographyResult:
    """Everything downstream code needs to trust (or distrust) this transform."""

    H: np.ndarray                     # 3x3 homography matrix, pixel -> pitch meters
    reprojection_error_m: float       # mean per-point error over ALL points, in pitch meters
    max_reprojection_error_m: float   # worst single-point error, in pitch meters
    confidence: float                 # 0-1, see homography_confidence()
    n_points: int
    per_point_errors_m: list = field(default_factory=list)
    # RANSAC consensus. None when RANSAC did not run (method=0, an exact fit
    # through every point), which is NOT the same as "RANSAC ran and every
    # point was an inlier" -- consumers must be able to tell those apart, so
    # the unknown case stays None rather than defaulting to n_points.
    n_inliers: int | None = None
    inlier_ratio: float | None = None
    # Mean error over the consensus set only. Equal to reprojection_error_m
    # when RANSAC did not run.
    inlier_reprojection_error_m: float | None = None
    #: Boolean mask over the input correspondences, True = inlier.
    inlier_mask: list = field(default_factory=list)


def compute_homography(
    pixel_pts: np.ndarray | list,
    pitch_pts: np.ndarray | list,
    method: int = 0,
    ransac_reproj_threshold: float = 3.0,
) -> HomographyResult:
    """
    Fit a pixel -> pitch-meters homography from corresponding point pairs.

    Args:
        pixel_pts: (N, 2) array of (x, y) pixel coordinates in the source frame.
        pitch_pts: (N, 2) array of (x, y) pitch coordinates in meters
            (use constants.REFERENCE_POINTS for known landmarks).
        method: 0 for exact least-squares/DLT fit (use when N == 4 and every
            point is trusted, e.g. clean corner-flag picks). Pass
            cv2.RANSAC for N > 4 with possibly-noisy clicks, which will
            down-weight outlier points automatically.
        ransac_reproj_threshold: only used when method=cv2.RANSAC; max
            allowed reprojection error (in pitch meters) for a point to be
            treated as an inlier.

    Returns:
        HomographyResult with the fitted matrix and an honest error estimate
        computed by reprojecting every input point and comparing against its
        known pitch position.

    CONFIDENCE, AND WHY IT IS MEASURED ON THE CONSENSUS SET WHEN RANSAC RUNS
        `reprojection_error_m` is, and remains, the mean over EVERY input
        point. `confidence` is derived from the mean over the RANSAC
        consensus set instead, and equals the all-point value when RANSAC did
        not run (method=0), so the exact-fit path is bit-for-bit unchanged.

        The reason for the split: when a caller passes points it already
        knows may contain outliers -- which is exactly why it passed
        cv2.RANSAC -- scoring the fit on the outliers it explicitly asked to
        have rejected describes a homography that was never used. On the
        calibration model's broadcast output that made every frame score 7-25
        m and fail the gate, including frames whose consensus set was
        sub-metre.

        This is only safe because the consensus set is SIZE-GATED downstream
        (HOMOGRAPHY_MIN_INLIERS / HOMOGRAPHY_MIN_INLIER_RATIO in
        CalibrationState.evaluate). Four points always agree perfectly, so
        without those gates this would be a confidence-manufacturing machine.
        `n_inliers` and `inlier_ratio` are reported so that gate can be
        applied, and both errors are kept so the discrepancy stays visible.

    Raises:
        ValueError: fewer than 4 point correspondences given (a homography
            has 8 degrees of freedom and needs at least 4 non-collinear
            point pairs to be well-posed).
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

    # Reproject every input pixel point through H and compare to its known
    # pitch position. Measured directly rather than trusting cv2's internal
    # bookkeeping, so the numbers below describe the matrix actually returned.
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
        # An empty consensus set cannot happen when findHomography returned a
        # matrix, but guard rather than divide by zero if cv2 ever changes.
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

    Returns (pitch_x_m, pitch_y_m). Matches the field naming used by
    PlayerTracking / Event in docs/database_schema.md.
    """
    result = _apply_homography(np.array([[x, y]]), H)
    px, py = result[0]
    return float(px), float(py)


def pixels_to_pitch(points: np.ndarray | list, H: np.ndarray) -> np.ndarray:
    """
    Batch version of pixel_to_pitch. Use this for whole tracking arrays
    (e.g. all player positions in a frame, or a full trajectory) instead of
    looping pixel_to_pitch() point by point -- one cv2 call instead of N.

    Args:
        points: (N, 2) array of (x, y) pixel coordinates.
        H: 3x3 homography matrix from compute_homography().

    Returns:
        (N, 2) array of (pitch_x_m, pitch_y_m).
    """
    return _apply_homography(np.asarray(points, dtype=np.float64), H)


def point_spread(
    pixel_pts: np.ndarray | list, frame_size: tuple[int, int]
) -> float:
    """
    Convex-hull area of the calibration points, as a fraction of frame area.

    WHY THIS IS NOT REDUNDANT WITH reprojection_error_m / confidence
        Reprojection error measures how consistently the points agree with
        EACH OTHER. It is lowest when they are clustered, because a small
        patch of the image is nearly affine and almost any homography fits
        it. Four landmarks bunched in one corner therefore produce
        near-zero error and ~1.0 confidence while the matrix is wildly
        wrong everywhere else in the frame -- which is where the players
        are. This function measures COVERAGE instead, the thing confidence
        structurally cannot see.

    Args:
        pixel_pts: (N, 2) array of (x, y) source points in pixels.
        frame_size: (width, height) of the frame those pixels came from.

    Returns:
        Hull area / frame area, in [0, 1]. Returns 0.0 for fewer than 3
        points, for degenerate (collinear or coincident) sets, and for a
        non-positive frame size -- all of which are "no measurable spread",
        not "spread unknown". A collinear set genuinely has zero area and
        cannot constrain a homography, so 0.0 is the honest answer rather
        than a special case.

        Values may exceed nothing but 1.0 is not clamped away artificially;
        points outside the frame bounds (a mis-detected landmark) can push
        this above 1.0, and that is worth seeing rather than hiding.
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

    # contourArea returns 0.0 for collinear hulls, which is exactly the
    # answer we want for a degenerate point set.
    return float(cv2.contourArea(hull)) / float(width * height)


def homography_geometry_problems(
    H: np.ndarray,
    frame_size: tuple[int, int],
    keypoints_px: np.ndarray | list | None = None,
) -> list[str]:
    """
    Structural sanity checks on a fitted pixel -> pitch homography.

    WHAT THIS CATCHES THAT REPROJECTION ERROR CANNOT
        Reprojection error asks "do these correspondences agree with each
        other". A homography built from landmarks whose IDENTITIES were
        swapped -- the mirrored-pitch failure documented in
        scripts/validate_auto_calibration.py -- agrees with itself perfectly
        and is wrong by the width of the pitch. Error is structurally blind
        to it, because the mistake is in the labels, not the residuals.

        These checks look at the matrix's effect on a REGION OF THE IMAGE
        instead, where that mistake is visible: project a probe rectangle
        into pitch metres and ask whether the result is a shape a camera
        could actually be seeing.

    WHICH REGION IS PROBED, AND WHY NOT THE FRAME CORNERS
        The probe is the bounding box of `keypoints_px` when given. Probing
        the FRAME's corners instead looks obvious and is wrong: the top edge
        of a broadcast frame is crowd and sky, which lies above the pitch
        plane's horizon, so it projects to hundreds of metres off-pitch under
        a perfectly good homography. Verified against the synthetic camera in
        tests/test_homography.py, whose corners project to -782 m while the
        fit is exact to 3e-7 m. The calibration keypoints, by contrast, ARE
        pitch landmarks and so are guaranteed to lie on the plane the
        homography is defined for.

        With no keypoints available the probe falls back to the frame's
        central 60%, which is far more likely to be pitch than its corners
        but is still only a fallback.

    Returns a list of human-readable problems, EMPTY when the homography is
    geometrically plausible. Returning the reasons rather than a bool is the
    point -- "rejected" without a why is what this codebase already refuses
    to emit elsewhere.

    Note this is a NECESSARY-condition test, not a sufficient one. Passing it
    means the fit is not obviously impossible; it does not mean the fit is
    accurate. It is one gate among several, never a substitute for the
    confidence and consensus gates.
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

    # Probe rectangle, traversed bottom-left, bottom-right, top-right,
    # top-left -- a counter-clockwise circuit in image coordinates (y grows
    # downward), which is what makes the winding test below meaningful.
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

    # -- mirroring -------------------------------------------------------
    # Signed area (shoelace). A pixel->pitch map that preserves orientation
    # keeps the sign; a mirrored one flips it, and every player then lands on
    # the wrong side of the pitch while every residual stays small.
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

    # -- degenerate / implausible extent ---------------------------------
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

    # -- gross out-of-bounds ---------------------------------------------
    # Overshoot past the touchlines is normal (stands, technical area).
    # Hundreds of metres off the pitch is not.
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

    # -- convexity -------------------------------------------------------
    # A projective map sends a convex quad to a convex quad. A self-
    # intersecting result means part of the probe crossed the horizon, and
    # pitch coordinates on the far side of it are meaningless.
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

    Uses the same exp(-error / scale) shape as Formation Detection's
    similarity confidence in docs/data analysis.md §3, for consistency
    across the codebase's confidence formulas: 0 error -> confidence 1.0,
    error growing relative to `scale_m` decays confidence toward 0.

    `scale_m` = 2.0 means: a 2m average reprojection error yields ~0.37
    confidence, comfortably below HOMOGRAPHY_CONFIDENCE_MIN (0.6), which a
    professional broadcast-camera calibration should never actually hit if
    the clicked points are accurate. Retune scale_m only after checking
    real calibration data, same caveat as NORMALIZATION_CONSTANT in the
    Formation Detection calibration procedure.

    Reference points, for intuition:
        error = 0.0m  -> confidence = 1.00
        error = 0.5m  -> confidence = 0.78
        error = 1.0m  -> confidence = 0.61
        error = 2.0m  -> confidence = 0.37
        error = 4.0m  -> confidence = 0.14
    """
    if reprojection_error_m < 0:
        raise ValueError("reprojection_error_m cannot be negative")
    confidence = math.exp(-reprojection_error_m / scale_m)
    return max(0.0, min(1.0, confidence))
