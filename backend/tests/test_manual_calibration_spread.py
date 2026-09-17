"""
The MANUAL calibration override must clear the same geometry gates as the
automatic path.

Before frame_size was threaded through _load_manual_calibration(), the
manual path called CalibrationState.evaluate() without it, so the
point-spread gate (HOMOGRAPHY_MIN_POINT_SPREAD) was silently skipped for
every operator-supplied calibration. A file whose points all sit inside one
penalty box would then have been accepted for a whole run -- passing the
confidence gate (clustered points reproject almost exactly among
themselves) while extrapolating badly everywhere the players actually are.
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from ai.computer_vision.tactical_analysis.constants import (
    HOMOGRAPHY_MIN_POINT_SPREAD,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
)
from backend.pipeline import calibration, runner

FRAME_SIZE = (1920, 1080)


def _broadcast_homography() -> list:
    """A PLAUSIBLE pixel -> pitch-metre matrix for a 1920x1080 frame.

    These tests used np.eye(3) as a stand-in until the geometric
    plausibility gate landed. Identity is not a homography any camera could
    have: it declares one pixel to be one metre, so the frame maps to a
    1600 m x 900 m "pitch" and the gate rejects it -- correctly, and for a
    reason that has nothing to do with what these tests are about.

    So the stub is a real one: the four frame-filling corners an operator
    clicking the pitch outline would produce, mapped to the actual pitch
    corners. Everything these tests assert (the spread gate, the confidence
    gate, and the match_id/video_id lookup rules) is unchanged; only the
    placeholder matrix became physically possible.
    """
    pixel = np.array([[200.0, 200.0], [1700.0, 220.0],
                      [1750.0, 900.0], [150.0, 880.0]])
    pitch = np.array([[0.0, 0.0], [PITCH_LENGTH_M, 0.0],
                      [PITCH_LENGTH_M, PITCH_WIDTH_M], [0.0, PITCH_WIDTH_M]])
    H, _ = cv2.findHomography(pixel, pitch, method=0)
    return H.tolist()


def _write_calibration(tmp_path, pixel_points, confidence=0.95):
    """A manual calibration file in load_calibration()'s format."""
    record = {
        "homography_matrix": _broadcast_homography(),
        "confidence": confidence,
        "reprojection_error_m": 0.1,
        "n_points": len(pixel_points),
        "frame_number": 0,
        "points": [{"pixel": list(p), "name": f"p{i}"}
                   for i, p in enumerate(pixel_points)],
    }
    path = tmp_path / "match-1.json"
    path.write_text(json.dumps(record))
    return path


@pytest.fixture
def calibration_dir(tmp_path, monkeypatch):
    # Patched on the module that owns it: _load_manual_calibration reads
    # CALIBRATION_DIR from backend.pipeline.calibration, not from runner.
    monkeypatch.setattr(calibration, "CALIBRATION_DIR", str(tmp_path))
    return tmp_path


def test_manual_calibration_rejected_when_points_are_clustered(calibration_dir):
    # Four points inside a 120x120 px box of a 1920x1080 frame:
    # hull area 14400 / 2073600 = 0.0069, well under the 0.05 threshold.
    _write_calibration(calibration_dir,
                       [(900, 500), (1020, 500), (1020, 620), (900, 620)])

    state = runner._load_manual_calibration("match-1", "video-1", frame_size=FRAME_SIZE)

    assert state is not None, "the file should still be found and loaded"
    assert state.valid is False, "a clustered manual calibration must be rejected"
    assert "spread" in (state.invalid_reason or "").lower()
    # Honest-null behaviour: rejected, not swapped for a fallback matrix.
    assert state.confidence == pytest.approx(0.95), (
        "the loaded confidence must be reported as-is, not zeroed or faked"
    )


def test_manual_calibration_accepted_when_points_are_well_spread(calibration_dir):
    _write_calibration(calibration_dir,
                       [(200, 200), (1700, 220), (1750, 900), (150, 880)])

    state = runner._load_manual_calibration("match-1", "video-1", frame_size=FRAME_SIZE)

    assert state is not None
    assert state.valid is True, f"unexpectedly rejected: {state.invalid_reason}"
    assert state.invalid_reason is None


def test_manual_calibration_confidence_gate_still_applies(calibration_dir):
    """The spread gate is additional to the confidence gate, not a swap."""
    _write_calibration(calibration_dir,
                       [(200, 200), (1700, 220), (1750, 900), (150, 880)],
                       confidence=0.2)

    state = runner._load_manual_calibration("match-1", "video-1", frame_size=FRAME_SIZE)

    assert state.valid is False
    assert "HOMOGRAPHY_CONFIDENCE_MIN" in (state.invalid_reason or "")


def test_manual_calibration_skips_spread_gate_when_frame_size_unknown(calibration_dir):
    """Documented behaviour: hull area needs a frame area to be a fraction of.

    _video_frame_size() returns None when the video cannot be probed, and in
    that case the spread check is skipped rather than rejecting everything.
    """
    _write_calibration(calibration_dir,
                       [(900, 500), (1020, 500), (1020, 620), (900, 620)])

    state = runner._load_manual_calibration("match-1", "video-1", frame_size=None)

    assert state.valid is True, "with no frame size the spread gate must not fire"


def test_video_frame_size_returns_none_for_unreadable_video():
    assert runner._video_frame_size("does-not-exist.mp4") is None


# ---------------------------------------------------------------------------
# video_id / match_id linkage (re-upload safety)
#
# Re-uploading the same source file is routine -- the live DB has one file
# uploaded 11 times -- and each upload mints a fresh video_id AND a fresh
# match_id (backend/api/main.py::upload_video creates a new Match row every
# time). These tests pin the property that makes that safe: a calibration
# is only ever loaded for the exact ids it was filed under, so one upload's
# calibration can never be silently served to a different upload.
# ---------------------------------------------------------------------------

def test_calibration_is_not_shared_across_re_uploads(calibration_dir):
    """A calibration filed under one upload must not leak to another."""
    record = {
        "homography_matrix": _broadcast_homography(), "confidence": 0.95,
        "n_points": 4,
        "points": [{"pixel": [200, 200]}, {"pixel": [1700, 220]},
                   {"pixel": [1750, 900]}, {"pixel": [150, 880]}],
    }
    (calibration_dir / "video-A.json").write_text(json.dumps(record))

    # The upload it belongs to finds it...
    found = runner._load_manual_calibration("match-A", "video-A", frame_size=FRAME_SIZE)
    assert found is not None and found.valid is True

    # ...a re-upload of the same source video (new video_id, new match_id)
    # does NOT, and degrades to the automatic path instead of inheriting a
    # calibration solved on a different upload.
    other = runner._load_manual_calibration("match-B", "video-B", frame_size=FRAME_SIZE)
    assert other is None, "a different upload must not inherit this calibration"


def test_calibration_lookup_prefers_match_id_over_video_id(calibration_dir):
    """Documents the precedence: {match_id}.json wins over {video_id}.json.

    Safe today only because upload_video() maintains a 1:1 Match<->Video
    relationship (verified against the live DB: 30 videos, 30 matches, no
    match carrying two videos). If a second Video were ever attached to an
    existing Match, both would resolve to the same {match_id}.json -- so
    this ordering is pinned here to make that coupling visible rather than
    incidental.
    """
    by_match = {
        "homography_matrix": _broadcast_homography(), "confidence": 0.95, "n_points": 4,
        "points": [{"pixel": [200, 200]}, {"pixel": [1700, 220]},
                   {"pixel": [1750, 900]}, {"pixel": [150, 880]}],
    }
    by_video = dict(by_match, confidence=0.71)
    (calibration_dir / "match-1.json").write_text(json.dumps(by_match))
    (calibration_dir / "video-1.json").write_text(json.dumps(by_video))

    state = runner._load_manual_calibration("match-1", "video-1", frame_size=FRAME_SIZE)
    assert state.confidence == pytest.approx(0.95), "match_id file must win"


def test_missing_calibration_returns_none_rather_than_a_default(calibration_dir):
    """No file for these ids -> None -> automatic path. Never a fallback H."""
    assert runner._load_manual_calibration("nope", "nope", frame_size=FRAME_SIZE) is None


def test_threshold_is_not_silently_zero():
    """Guard: a zeroed threshold would make every spread test vacuous."""
    assert HOMOGRAPHY_MIN_POINT_SPREAD > 0
