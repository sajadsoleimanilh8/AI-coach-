from types import SimpleNamespace

import numpy as np

from ai.computer_vision.frame_data import FieldRegion
from ai.computer_vision.tactical_analysis.auto_calibration import AutoCalibrationResult
from ai.computer_vision.tactical_analysis.homography import HomographyResult
from backend.api.calibration_debug import render_calibration_debug


class _Field:
    def detect(self, frame):
        h, w = frame.shape[:2]
        return FieldRegion([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)], 0.9, 1.0)


class _Calibrator:
    def __init__(self, confidence, keypoints=None):
        self.confidence = confidence
        self.keypoints = keypoints or [(10, 10), (90, 10), (90, 60), (10, 60)]

    def calibrate_frame(self, frame, field_region=None):
        homography = HomographyResult(np.eye(3), 0.1, 0.2, self.confidence, 4, [0.1] * 4)
        return AutoCalibrationResult(
            homography=homography,
            keypoints_px=list(self.keypoints),
            keypoint_indices=list(range(len(self.keypoints))),
            detection_confidence=0.88,
            n_keypoints_visible=len(self.keypoints),
        )


def _players():
    return [SimpleNamespace(player_id=7, pixel_x=30.0, pixel_y=40.0)]


def test_debug_overlay_projects_players_only_for_valid_calibration():
    png, meta = render_calibration_debug(
        np.zeros((80, 120, 3), dtype=np.uint8), _players(),
        field_detector=_Field(), calibrator=_Calibrator(0.95),
    )
    assert png.startswith(b"\x89PNG")
    assert meta["calibration_valid"] is True
    assert meta["projection_suppressed"] is False
    assert meta["projected_players"][0]["player_id"] == 7
    assert meta["calibration_metric_method"] == "deterministic"
    assert meta["keypoint_detection_method"] == "ml_trained"


def test_debug_overlay_keeps_evidence_but_suppresses_invalid_projection():
    png, meta = render_calibration_debug(
        np.zeros((80, 120, 3), dtype=np.uint8), _players(),
        field_detector=_Field(), calibrator=_Calibrator(0.1),
    )
    assert png.startswith(b"\x89PNG")
    assert meta["calibration_valid"] is False
    assert meta["projection_suppressed"] is True
    assert meta["projected_players"] == []
    assert meta["calibration_metric_confidence"] == "low_upstream_confidence"


def test_debug_overlay_applies_spread_gate_despite_high_confidence():
    """The debug route must reject a clustered fit, like the pipeline does."""
    clustered = [(10, 10), (18, 10), (18, 16), (10, 16)]
    png, meta = render_calibration_debug(
        np.zeros((80, 120, 3), dtype=np.uint8), _players(),
        field_detector=_Field(), calibrator=_Calibrator(0.95, keypoints=clustered),
    )
    assert png.startswith(b"\x89PNG")
    assert meta["calibration_valid"] is False, "clustered points must be rejected"
    assert "spread" in (meta["invalid_reason"] or "").lower()
    assert meta["projection_suppressed"] is True
    assert meta["projected_players"] == []
    assert meta["n_keypoints"] == 4


def test_debug_overlay_spread_gate_does_not_reject_well_spread_points():
    """Guard against the gate over-firing: the default spread must pass."""
    _, meta = render_calibration_debug(
        np.zeros((80, 120, 3), dtype=np.uint8), _players(),
        field_detector=_Field(), calibrator=_Calibrator(0.95),
    )
    assert meta["calibration_valid"] is True
    assert meta["invalid_reason"] is None
