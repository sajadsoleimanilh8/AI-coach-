"""
Sanity checks for the pixel <-> pitch homography.

No real video needed: we synthesize a plausible broadcast-camera view by
applying a known forward (pitch -> pixel) perspective transform to the
pitch reference points, feed the resulting "clicked pixels" into
compute_homography(), and then check that:

  1. Reprojection error / confidence behave sanely (low error -> high conf,
     added click noise -> lower confidence).
  2. Real-world distances (penalty box width ~40.32m, center circle
     diameter ~18.3m, six-yard box depth ~5.5m) measured on *held-out*
     points (not used to fit H) come back close to their known values --
     this is the actual "did we get a usable transform" test, since
     reprojection error alone can look fine while distances are still off
     if the point set is degenerate.
  3. The point-spread gate rejects clustered landmarks that nonetheless fit
     with high confidence, and does not replace the confidence gate.

This used to be one script-style `run()` with nine numbered checks behind a
single pytest wrapper, so the first failing assertion hid every check after it
and pytest reported one test. Each check is now its own test; the assertions
are unchanged.

Run: pytest ai/computer_vision/tactical_analysis/tests/test_homography.py -v
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from ai.computer_vision.frame_data import CalibrationState
from ai.computer_vision.tactical_analysis.constants import (
    CENTER_CIRCLE_RADIUS_M,
    HOMOGRAPHY_CONFIDENCE_MIN,
    HOMOGRAPHY_MIN_POINT_SPREAD,
    PENALTY_AREA_WIDTH_M,
    REFERENCE_POINTS,
    SIX_YARD_DEPTH_M,
)
from ai.computer_vision.tactical_analysis.homography import (
    compute_homography,
    homography_confidence,
    pixel_to_pitch,
    pixels_to_pitch,
    point_spread,
)

#: The synthetic camera in make_synthetic_forward_homography() targets a
#: 1920x1080 frame; point_spread() needs that to express hull area as a
#: fraction. (width, height), matching cv2/frame_size convention elsewhere.
FRAME_SIZE = (1920, 1080)

#: Points an operator would actually click; used to FIT the homography.
CALIB_NAMES = [
    "corner_bottom_left",
    "corner_top_left",
    "halfway_bottom",
    "halfway_top",
    "left_penalty_area_bottom_near",
    "left_penalty_area_top_near",
]

#: NOT used to fit H; used only to check recovered real-world distances.
HOLDOUT_NAMES = [
    "left_penalty_area_bottom_far",
    "left_penalty_area_top_far",
    "left_six_yard_bottom_near",
    "left_six_yard_bottom_far",
    "center_circle_top",
    "center_circle_bottom",
]


def make_synthetic_forward_homography() -> np.ndarray:
    """
    A hand-built pitch (meters) -> pixel perspective transform standing in
    for a real broadcast/tactical camera: pitch recedes into the distance,
    so far points compress toward a vanishing region, near points spread out.
    This is the *inverse* of what compute_homography() will fit.
    """
    # Map 4 known pitch corners of the left half of the pitch to plausible
    # pixel positions in a 1920x1080 frame, with the standard "camera looking
    # down the touchline" perspective compression on the far side.
    pitch_corners = np.array(
        [
            [0.0, 0.0],     # near-left corner  -> bottom-left of frame, spread out
            [0.0, 68.0],    # near-right corner -> bottom-right of frame
            [52.5, 0.0],    # halfway near      -> compressed toward center, higher up
            [52.5, 68.0],   # halfway far
        ],
        dtype=np.float64,
    )
    pixel_corners = np.array(
        [
            [200.0, 1000.0],
            [1720.0, 1000.0],
            [760.0, 300.0],
            [1160.0, 300.0],
        ],
        dtype=np.float64,
    )
    H_forward, _ = cv2.findHomography(pitch_corners, pixel_corners, method=0)
    return H_forward


def pitch_to_pixel(points_m: np.ndarray, H_forward: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_m, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H_forward).reshape(-1, 2)


def euclidean(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# ---------------------------------------------------------------------------
# Shared synthetic scene. Module-scoped: every check reads the same fit, which
# is what the original single run() did.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def H_forward() -> np.ndarray:
    return make_synthetic_forward_homography()


@pytest.fixture(scope="module")
def calib(H_forward):
    pitch = np.array([REFERENCE_POINTS[n] for n in CALIB_NAMES])
    return pitch, pitch_to_pixel(pitch, H_forward)


@pytest.fixture(scope="module")
def holdout_pixels(H_forward) -> np.ndarray:
    pitch = np.array([REFERENCE_POINTS[n] for n in HOLDOUT_NAMES])
    return pitch_to_pixel(pitch, H_forward)


@pytest.fixture(scope="module")
def clean_result(calib):
    pitch, pixels = calib
    return compute_homography(pixels, pitch, method=0)


@pytest.fixture(scope="module")
def clustered(H_forward):
    """Four landmarks bunched in a 2m x 4m patch by the left penalty spot."""
    pitch = np.array(
        [[10.0, 32.0], [12.0, 32.0], [12.0, 36.0], [10.0, 36.0]],
        dtype=np.float64,
    )
    pixels = pitch_to_pixel(pitch, H_forward)
    return pixels, compute_homography(pixels, pitch, method=0)


# ---------------------------------------------------------------------------
# 1-6: the fit itself
# ---------------------------------------------------------------------------

def test_clean_clicks_give_low_error_and_high_confidence(clean_result):
    assert clean_result.reprojection_error_m < 0.01, "clean synthetic points should reproject almost exactly"
    assert clean_result.confidence > 0.99, "near-zero error should yield near-1.0 confidence"


def test_known_real_world_distances_on_held_out_points(clean_result, holdout_pixels):
    recovered = dict(zip(HOLDOUT_NAMES, pixels_to_pitch(holdout_pixels, clean_result.H)))

    # Penalty box width: near/far edges share the box-width span across y
    box_width = euclidean(recovered["left_penalty_area_bottom_far"], recovered["left_penalty_area_top_far"])
    assert abs(box_width - PENALTY_AREA_WIDTH_M) < 0.05, "penalty box width off by >5cm"

    # Center circle diameter (top point to bottom point, straight through center)
    diameter = euclidean(recovered["center_circle_top"], recovered["center_circle_bottom"])
    assert abs(diameter - 2 * CENTER_CIRCLE_RADIUS_M) < 0.05, "center circle diameter off by >5cm"

    six_yard_depth = euclidean(recovered["left_six_yard_bottom_near"], recovered["left_six_yard_bottom_far"])
    assert abs(six_yard_depth - SIX_YARD_DEPTH_M) < 0.05, "six-yard depth off by >5cm"


def test_noisy_clicks_degrade_confidence(calib, clean_result):
    """Simulates imprecise manual point-picking: +-3px jitter."""
    pitch, pixels = calib
    rng = np.random.default_rng(42)
    noisy = compute_homography(pixels + rng.normal(scale=3.0, size=pixels.shape), pitch, method=0)
    assert noisy.reprojection_error_m > clean_result.reprojection_error_m, (
        "noisy clicks should produce strictly higher error than clean clicks"
    )
    assert noisy.confidence < clean_result.confidence, "noisy clicks should reduce confidence"


def test_single_point_function_matches_batch_function(clean_result, holdout_pixels):
    x, y = holdout_pixels[0]
    single = pixel_to_pitch(float(x), float(y), clean_result.H)
    batch = pixels_to_pitch(holdout_pixels, clean_result.H)[0]
    assert abs(single[0] - batch[0]) < 1e-9 and abs(single[1] - batch[1]) < 1e-9


def test_too_few_points_raises_value_error(calib):
    pitch, pixels = calib
    with pytest.raises(ValueError):
        compute_homography(pixels[:3], pitch[:3])


def test_confidence_formula_straddles_the_usability_threshold():
    """~1.8m average error sits around the 'unusable' threshold (0.6) used
    everywhere else in the pipeline; 0.3m is clearly usable, 5m clearly not."""
    assert homography_confidence(0.3) > HOMOGRAPHY_CONFIDENCE_MIN
    assert homography_confidence(5.0) < HOMOGRAPHY_CONFIDENCE_MIN


# ---------------------------------------------------------------------------
# 7-9: the point-spread gate, and that it is ADDITIONAL to the confidence gate
# ---------------------------------------------------------------------------

def test_clustered_points_are_rejected_despite_high_confidence(clustered):
    """The premise of the spread gate: bunched landmarks fit a homography
    almost perfectly AMONG THEMSELVES, so confidence is near 1.0 while the
    matrix says nothing about the rest of the frame. Proves both halves:
    confidence really fails to catch it, and spread does."""
    pixels, result = clustered
    assert result.confidence > HOMOGRAPHY_CONFIDENCE_MIN, (
        "premise of this test: a clustered fit must LOOK confident, "
        "otherwise the spread gate would be redundant"
    )
    assert point_spread(pixels, FRAME_SIZE) < HOMOGRAPHY_MIN_POINT_SPREAD

    state = CalibrationState(
        H=result.H,
        confidence=result.confidence,
        reprojection_error_m=result.reprojection_error_m,
        n_points=result.n_points,
    ).evaluate(keypoints_px=[tuple(p) for p in pixels], frame_size=FRAME_SIZE)
    assert not state.valid, "clustered points must be rejected"
    assert "spread" in (state.invalid_reason or "").lower(), (
        f"rejection must name the spread gate, got: {state.invalid_reason}"
    )


def test_spread_gate_is_skipped_when_frame_size_is_unknown(clustered):
    """By documented design: hull area is meaningless without a frame area."""
    pixels, result = clustered
    state = CalibrationState(
        H=result.H, confidence=result.confidence, n_points=result.n_points,
    ).evaluate(keypoints_px=[tuple(p) for p in pixels], frame_size=None)
    assert state.valid, (
        "with frame_size=None the spread gate must be skipped, not silently "
        "rejecting every calibration"
    )


def test_well_spread_points_are_accepted(calib, clean_result):
    _, pixels = calib
    assert point_spread(pixels, FRAME_SIZE) > HOMOGRAPHY_MIN_POINT_SPREAD

    state = CalibrationState(
        H=clean_result.H,
        confidence=clean_result.confidence,
        reprojection_error_m=clean_result.reprojection_error_m,
        n_points=clean_result.n_points,
    ).evaluate(keypoints_px=[tuple(p) for p in pixels], frame_size=FRAME_SIZE)
    assert state.valid, (
        f"well-spread, high-confidence points must be accepted, got: {state.invalid_reason}"
    )
    assert state.invalid_reason is None


def test_degenerate_point_sets_have_zero_spread(calib):
    """A collinear set has genuinely zero area and cannot constrain a
    homography -- 0.0 is the honest answer, not a special case."""
    _, pixels = calib
    collinear = np.array([[100.0, 100.0], [200.0, 200.0], [300.0, 300.0], [400.0, 400.0]])
    assert point_spread(collinear, FRAME_SIZE) == 0.0
    assert point_spread(pixels[:2], FRAME_SIZE) == 0.0  # < 3 points


def test_confidence_gate_still_rejects_well_spread_points(calib, clean_result):
    """The spread gate is ADDITIONAL to the confidence gate, not a
    replacement: well-spread points with poor confidence are still rejected,
    and the reason still names the confidence gate."""
    _, pixels = calib
    low_conf = homography_confidence(5.0)
    assert low_conf < HOMOGRAPHY_CONFIDENCE_MIN
    state = CalibrationState(
        H=clean_result.H,
        confidence=low_conf,
        reprojection_error_m=5.0,
        n_points=clean_result.n_points,
    ).evaluate(keypoints_px=[tuple(p) for p in pixels], frame_size=FRAME_SIZE)
    assert not state.valid, "low confidence must still be rejected"
    assert "HOMOGRAPHY_CONFIDENCE_MIN" in (state.invalid_reason or ""), (
        f"rejection must still name the confidence gate, got: {state.invalid_reason}"
    )


def test_confidence_failure_is_reported_before_spread_failure(clustered):
    """A fit that fails both gates reports the confidence failure -- the
    pre-existing ordering."""
    pixels, result = clustered
    state = CalibrationState(
        H=result.H, confidence=homography_confidence(5.0), n_points=4,
    ).evaluate(keypoints_px=[tuple(p) for p in pixels], frame_size=FRAME_SIZE)
    assert not state.valid
    assert "HOMOGRAPHY_CONFIDENCE_MIN" in (state.invalid_reason or "")


def test_missing_homography_is_rejected():
    assert not CalibrationState(H=None).evaluate(frame_size=FRAME_SIZE).valid
