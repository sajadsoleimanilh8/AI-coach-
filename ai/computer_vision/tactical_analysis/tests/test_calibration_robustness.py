"""
The gates added when the 0/3300-valid calibration failure was investigated.

WHAT THESE PIN, AND WHAT THEY DELIBERATELY DO NOT CLAIM
    Each test below pins one mechanism: preprocessing parity, RANSAC
    consensus accounting, the consensus gates, geometric plausibility, the
    bounded temporal fallback, and geometry-preserving smoothing.

    None of them claims the automatic calibration is ACCURATE on
    out-of-domain footage. It is not -- see the WHAT REMAINS BROKEN note in
    auto_calibration.py, which records that fits scoring 8/8 inliers at
    0.31 m reprojection error were still metres wrong when rendered against
    the painted pitch lines. These tests pin that the machinery behaves as
    documented, not that the model is good enough.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from ai.computer_vision.frame_data import (
    CalibrationSource,
    CalibrationState,
    CameraMotion,
    CameraState,
)
from ai.computer_vision.tactical_analysis.auto_calibration import (
    AutoCalibrator,
    blend_homographies,
    extract_keypoints,
    preprocess_frame,
)
from ai.computer_vision.tactical_analysis.constants import (
    HOMOGRAPHY_MIN_INLIERS,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
)
from ai.computer_vision.tactical_analysis.homography import (
    compute_homography,
    homography_geometry_problems,
)

FRAME_SIZE = (1920, 1080)


def _broadcast_h():
    """A plausible pixel -> pitch-metre homography for a 1920x1080 frame."""
    pixel = np.array([[200.0, 200.0], [1700.0, 220.0],
                      [1750.0, 900.0], [150.0, 880.0]])
    pitch = np.array([[0.0, 0.0], [PITCH_LENGTH_M, 0.0],
                      [PITCH_LENGTH_M, PITCH_WIDTH_M], [0.0, PITCH_WIDTH_M]])
    H, _ = cv2.findHomography(pixel, pitch, method=0)
    return H, pixel


# ----------------------------------------------------------------------
# Preprocessing parity -- the root cause of the 0/3300 failure
# ----------------------------------------------------------------------

def test_stretch_square_reshapes_and_reports_the_inverse_scale():
    """The mapping back to original pixels must be exact, or every keypoint
    lands in the wrong place and the homography is silently wrong."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    image, sx, sy = preprocess_frame(frame, 960, "stretch_square")

    assert image.shape[:2] == (960, 960), "must match the 960x960 training shape"
    assert sx == pytest.approx(1280 / 960)
    assert sy == pytest.approx(720 / 960)

    # A keypoint at the centre of the square image must map back to the
    # centre of the original frame.
    assert (480 * sx, 480 * sy) == pytest.approx((640.0, 360.0))


def test_native_preprocessing_is_the_identity():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    image, sx, sy = preprocess_frame(frame, 960, "native")
    assert image is frame
    assert (sx, sy) == (1.0, 1.0)


def test_unknown_preprocess_mode_falls_back_to_native_without_raising():
    """A config typo must not take down a pipeline run."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    image, sx, sy = preprocess_frame(frame, 960, "streeetch")
    assert image is frame and (sx, sy) == (1.0, 1.0)


class _FakeKeypoints:
    def __init__(self, data):
        self.data = data


class _FakeBoxes:
    def __init__(self, conf):
        self.conf = np.asarray(conf)

    def __len__(self):
        return len(self.conf)


class _FakeResult:
    def __init__(self, conf, kpts):
        self.boxes = _FakeBoxes(conf)
        self.keypoints = _FakeKeypoints(np.asarray(kpts))


def test_extract_keypoints_maps_back_to_original_frame_pixels():
    kp = np.zeros((1, 32, 3), dtype=np.float64)
    kp[0, 0] = (480.0, 480.0, 0.9)      # centre of a 960x960 square image
    kp[0, 1] = (0.0, 0.0, 0.9)          # ultralytics' "absent landmark"
    result = _FakeResult([0.9], kp)

    keypoints, confidences, det_conf = extract_keypoints(
        result, 0.5, scale=(1280 / 960, 720 / 960))

    assert keypoints[0] == pytest.approx((640.0, 360.0))
    assert 1 not in keypoints, "(0, 0) means absent and must stay dropped"
    assert det_conf == pytest.approx(0.9)
    assert confidences[0] == pytest.approx(0.9)


# ----------------------------------------------------------------------
# RANSAC consensus accounting
# ----------------------------------------------------------------------

def test_ransac_reports_inliers_and_scores_confidence_on_the_consensus():
    """One gross outlier must not drag the reported confidence, but it must
    still be visible in reprojection_error_m and the inlier count."""
    H_fwd, _ = _broadcast_h()
    inv = np.linalg.inv(H_fwd)
    pitch = np.array([[5.0, 5.0], [100.0, 5.0], [100.0, 63.0], [5.0, 63.0],
                      [52.5, 5.0], [52.5, 63.0], [30.0, 34.0], [75.0, 34.0]])
    pixel = cv2.perspectiveTransform(
        pitch.reshape(-1, 1, 2).astype(np.float64), inv).reshape(-1, 2)
    pixel[0] += 400.0                     # one badly mislocalised landmark

    res = compute_homography(pixel, pitch, method=cv2.RANSAC,
                             ransac_reproj_threshold=2.0)

    assert res.n_inliers is not None and res.n_inliers >= 6
    assert res.inlier_ratio == pytest.approx(res.n_inliers / res.n_points)
    assert len(res.inlier_mask) == res.n_points
    # The outlier is still reported in the all-point error...
    assert res.reprojection_error_m > res.inlier_reprojection_error_m
    # ...while confidence describes the matrix that was actually returned.
    assert res.confidence > 0.9


def test_exact_fit_reports_no_consensus_rather_than_a_perfect_one():
    """method=0 means RANSAC never ran. That must stay distinguishable from
    'RANSAC ran and every point agreed', or the consensus gate would be
    silently satisfied by a 4-point exact fit."""
    H_fwd, pixel = _broadcast_h()
    pitch = np.array([[0.0, 0.0], [PITCH_LENGTH_M, 0.0],
                      [PITCH_LENGTH_M, PITCH_WIDTH_M], [0.0, PITCH_WIDTH_M]])

    res = compute_homography(pixel, pitch, method=0)

    assert res.n_inliers is None
    assert res.inlier_ratio is None
    assert res.confidence > 0.99


def test_consensus_gates_reject_a_too_small_or_too_contradicted_set():
    H, pixel = _broadcast_h()

    too_few = CalibrationState(
        H=H, confidence=0.95, n_points=20,
        n_inliers=HOMOGRAPHY_MIN_INLIERS - 1, inlier_ratio=0.9,
    ).evaluate(keypoints_px=[tuple(p) for p in pixel], frame_size=FRAME_SIZE)
    assert not too_few.valid
    assert "HOMOGRAPHY_MIN_INLIERS" in (too_few.invalid_reason or "")

    contradicted = CalibrationState(
        H=H, confidence=0.95, n_points=100,
        n_inliers=20, inlier_ratio=0.2,
    ).evaluate(keypoints_px=[tuple(p) for p in pixel], frame_size=FRAME_SIZE)
    assert not contradicted.valid
    assert "HOMOGRAPHY_MIN_INLIER_RATIO" in (contradicted.invalid_reason or "")


def test_consensus_gate_is_skipped_when_ransac_did_not_run():
    """n_inliers=None means 'not measured', never 'zero'."""
    H, pixel = _broadcast_h()
    state = CalibrationState(
        H=H, confidence=0.95, n_points=4, n_inliers=None, inlier_ratio=None,
    ).evaluate(keypoints_px=[tuple(p) for p in pixel], frame_size=FRAME_SIZE)
    assert state.valid, f"unexpectedly rejected: {state.invalid_reason}"


# ----------------------------------------------------------------------
# Geometric plausibility
# ----------------------------------------------------------------------

def test_a_plausible_broadcast_homography_has_no_geometry_problems():
    H, pixel = _broadcast_h()
    assert homography_geometry_problems(H, FRAME_SIZE, pixel) == []


def test_mirrored_homography_is_rejected():
    """The most damaging silent failure available: pitch x/y flipped, every
    residual still small, every player on the wrong side of the pitch."""
    H, pixel = _broadcast_h()
    mirrored = np.diag([1.0, -1.0, 1.0]) @ H
    problems = homography_geometry_problems(mirrored, FRAME_SIZE, pixel)
    assert any("mirrored" in p for p in problems), problems


def test_degenerate_and_collapsed_homographies_are_rejected():
    H, pixel = _broadcast_h()

    singular = homography_geometry_problems(np.zeros((3, 3)), FRAME_SIZE, pixel)
    assert any("degenerate" in p for p in singular), singular

    collapsed = homography_geometry_problems(
        np.diag([1e-3, 1e-3, 1.0]) @ H, FRAME_SIZE, pixel)
    assert any("implausible projected scale" in p for p in collapsed), collapsed


def test_geometry_gate_runs_after_the_evidence_gates():
    """A clustered point set must be reported by the spread gate, which names
    the cause, not by the geometry gate, which names its consequence."""
    H_fwd, _ = _broadcast_h()
    inv = np.linalg.inv(H_fwd)
    clustered_pitch = np.array([[10.0, 32.0], [12.0, 32.0],
                                [12.0, 36.0], [10.0, 36.0]])
    clustered_px = cv2.perspectiveTransform(
        clustered_pitch.reshape(-1, 1, 2).astype(np.float64), inv).reshape(-1, 2)
    res = compute_homography(clustered_px, clustered_pitch, method=0)

    state = CalibrationState(
        H=res.H, confidence=res.confidence, n_points=4,
    ).evaluate(keypoints_px=[tuple(p) for p in clustered_px],
               frame_size=FRAME_SIZE)

    assert not state.valid
    assert "spread" in (state.invalid_reason or "").lower(), state.invalid_reason


# ----------------------------------------------------------------------
# Geometry-preserving smoothing
# ----------------------------------------------------------------------

def test_blend_stays_a_valid_projective_transform():
    """The reason this exists instead of averaging the matrices."""
    H_a, _ = _broadcast_h()
    pixel = np.array([[210.0, 205.0], [1690.0, 230.0],
                      [1740.0, 890.0], [160.0, 870.0]])
    pitch = np.array([[0.0, 0.0], [PITCH_LENGTH_M, 0.0],
                      [PITCH_LENGTH_M, PITCH_WIDTH_M], [0.0, PITCH_WIDTH_M]])
    H_b, _ = cv2.findHomography(pixel, pitch, method=0)

    blended = blend_homographies(H_a, H_b, FRAME_SIZE, 0.5)

    assert blended is not None
    assert np.isfinite(blended).all()
    assert abs(np.linalg.det(blended)) > 1e-12
    assert homography_geometry_problems(blended, FRAME_SIZE) == []


def test_blend_endpoints_are_exact():
    H_a, _ = _broadcast_h()
    H_b = np.diag([1.01, 0.99, 1.0]) @ H_a
    assert np.allclose(blend_homographies(H_a, H_b, FRAME_SIZE, 1.0), H_b)
    assert np.allclose(blend_homographies(H_a, H_b, FRAME_SIZE, 0.0), H_a)


def test_blend_weight_is_clamped_not_extrapolated():
    """Extrapolating past either endpoint invents a camera pose that neither
    fit measured."""
    H_a, _ = _broadcast_h()
    H_b = np.diag([1.01, 0.99, 1.0]) @ H_a
    assert np.allclose(blend_homographies(H_a, H_b, FRAME_SIZE, 5.0), H_b)
    assert np.allclose(blend_homographies(H_a, H_b, FRAME_SIZE, -5.0), H_a)


# ----------------------------------------------------------------------
# Bounded temporal fallback
# ----------------------------------------------------------------------

class _StubCalibrator(AutoCalibrator):
    """AutoCalibrator with the model and the motion detector stubbed out, so
    the temporal state machine can be driven deterministically."""

    def __init__(self, **kw):
        H, _ = _broadcast_h()
        self.kpt_conf_min = 0.5
        self.kpt_conf_relaxed = 0.35
        self.imgsz = 960
        self.conf = 0.3
        self.preprocess = "native"
        self.device = None
        self.model = None
        self.max_jump_m = 15.0
        self.max_fallback_frames = kw.get("max_fallback_frames", 3)
        self.fallback_decay = kw.get("fallback_decay", 0.985)
        self.smoothing = 1.0
        self.motion = None
        self.stable = CalibrationState(
            H=H, confidence=0.90, reprojection_error_m=0.2, n_points=10,
            source=CalibrationSource.model, valid=True, solved_on_frame=0,
            n_inliers=10, inlier_ratio=1.0, solved_confidence=0.90,
        )
        self.carried_frames = 0
        self.n_attempted = self.n_accepted = self.n_reused = 0
        self.n_rejected_jump = self.n_rejected_invalid = 0
        self.n_strict = self.n_relaxed = self.n_fallback_expired = 0
        self.n_smoothed = 0


def test_carried_calibration_decays_and_then_expires():
    cal = _StubCalibrator(max_fallback_frames=3)
    camera = CameraState(motion=CameraMotion.static)

    first, _ = cal._carry(camera, 1)
    assert first.valid and first.source is CalibrationSource.carried
    assert first.carried_frames == 1
    assert first.confidence < 0.90, "a carried calibration must not score as fresh"
    assert first.solved_on_frame == 0, "provenance must survive the carry"

    second, _ = cal._carry(camera, 2)
    assert second.valid and second.confidence < first.confidence

    third, _ = cal._carry(camera, 3)
    assert third.valid

    expired, _ = cal._carry(camera, 4)
    assert not expired.valid, "the fallback must expire, not run forever"
    assert "expired" in (expired.invalid_reason or "")
    assert expired.H is None, "an expired fallback must not still serve a matrix"
    assert cal.n_fallback_expired == 1


def test_fallback_expires_early_when_decay_crosses_the_confidence_gate():
    """Weaker evidence must survive fewer frames than strong evidence."""
    cal = _StubCalibrator(max_fallback_frames=1000, fallback_decay=0.5)
    cal.stable.confidence = 0.62          # only just above the 0.6 gate
    camera = CameraState(motion=CameraMotion.static)

    state, _ = cal._carry(camera, 1)

    assert not state.valid
    assert "HOMOGRAPHY_CONFIDENCE_MIN" in (state.invalid_reason or "")
    assert cal.n_fallback_expired == 1


def test_carry_without_a_previous_calibration_is_honest_about_it():
    cal = _StubCalibrator()
    cal.stable = None
    state, _ = cal._carry(CameraState(motion=CameraMotion.static), 1)
    assert not state.valid
    assert state.H is None
    assert "no previous valid calibration" in (state.invalid_reason or "")
