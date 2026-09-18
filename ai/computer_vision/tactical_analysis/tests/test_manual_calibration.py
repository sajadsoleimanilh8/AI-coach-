"""Manual calibration: building, saving and loading a calibration record.

The interactive click-through needs a display and is not tested here; the
record it produces, the loader the pipeline uses, and the headless
(--points-file) CLI path are.
"""

from __future__ import annotations

import json
import sys

import cv2
import numpy as np
import pytest

from ai.computer_vision.tactical_analysis import manual_calibration as mc
from ai.computer_vision.tactical_analysis.constants import MIN_CALIBRATION_POINTS, REFERENCE_POINTS
from ai.computer_vision.tactical_analysis.homography import pixels_to_pitch

POINTS = [
    "corner_bottom_left",
    "corner_top_left",
    "halfway_bottom",
    "halfway_top",
    "left_penalty_area_bottom_near",
    "left_penalty_area_top_near",
]


@pytest.fixture(scope="module")
def clicks() -> dict[str, tuple[float, float]]:
    """Pixel clicks produced by a known broadcast-like perspective camera."""
    pitch = np.array([[0.0, 0.0], [0.0, 68.0], [52.5, 0.0], [52.5, 68.0]])
    pixels = np.array([[200.0, 1000.0], [1720.0, 1000.0], [760.0, 300.0], [1160.0, 300.0]])
    forward, _ = cv2.findHomography(pitch, pixels, method=0)
    pts = np.array([REFERENCE_POINTS[n] for n in POINTS], dtype=np.float64).reshape(-1, 1, 2)
    px = cv2.perspectiveTransform(pts, forward).reshape(-1, 2)
    return {name: (float(x), float(y)) for name, (x, y) in zip(POINTS, px)}


def test_too_few_points_is_refused(clicks):
    few = dict(list(clicks.items())[: MIN_CALIBRATION_POINTS - 1])
    with pytest.raises(ValueError, match="at least"):
        mc.build_calibration_record(few, "clip.mp4", 0)


def test_record_contains_what_the_pipeline_needs(clicks):
    record = mc.build_calibration_record(clicks, "clip.mp4", 12)
    assert record["n_points"] == len(POINTS)
    assert record["frame_number"] == 12
    assert np.array(record["homography_matrix"]).shape == (3, 3)
    assert record["confidence"] > 0.99            # exact synthetic clicks
    assert record["reprojection_error_m"] < 0.01
    assert [p["name"] for p in record["points"]] == POINTS
    assert len(record["per_point_errors_m"]) == len(POINTS)


def test_record_round_trips_through_load_calibration(clicks, tmp_path):
    record = mc.build_calibration_record(clicks, "clip.mp4", 0)
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(record))

    H, loaded = mc.load_calibration(str(path))
    assert loaded == json.loads(json.dumps(record))

    # The loaded matrix maps every clicked pixel back onto its pitch landmark.
    recovered = pixels_to_pitch(np.array(list(clicks.values())), H)
    expected = np.array([REFERENCE_POINTS[n] for n in POINTS])
    assert np.abs(recovered - expected).max() < 0.01


def _run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["manual_calibration", *args])
    mc.main()


def test_headless_cli_writes_a_calibration(clicks, tmp_path, monkeypatch):
    points_file = tmp_path / "clicks.json"
    points_file.write_text(json.dumps({k: list(v) for k, v in clicks.items()}))
    out = tmp_path / "nested" / "match.json"

    _run_cli(monkeypatch, "--points-file", str(points_file), "--out", str(out))

    saved = json.loads(out.read_text())
    assert saved["n_points"] == len(POINTS)
    assert saved["confidence"] > 0.99


def test_headless_cli_skips_unknown_point_names(clicks, tmp_path, monkeypatch, capsys):
    raw = {k: list(v) for k, v in clicks.items()}
    raw["not_a_real_landmark"] = [1.0, 2.0]
    points_file = tmp_path / "clicks.json"
    points_file.write_text(json.dumps(raw))
    out = tmp_path / "match.json"

    _run_cli(monkeypatch, "--points-file", str(points_file), "--out", str(out))

    assert "not_a_real_landmark" in capsys.readouterr().err
    assert json.loads(out.read_text())["n_points"] == len(POINTS)


def test_headless_cli_warns_when_confidence_is_unusable(clicks, tmp_path, monkeypatch, capsys):
    """Badly placed clicks still save, but the operator is told not to use them."""
    rng = np.random.default_rng(0)
    noisy = {k: [x + rng.normal(scale=120), y + rng.normal(scale=120)] for k, (x, y) in clicks.items()}
    points_file = tmp_path / "clicks.json"
    points_file.write_text(json.dumps(noisy))

    _run_cli(monkeypatch, "--points-file", str(points_file), "--out", str(tmp_path / "m.json"))

    assert "WARNING: confidence below" in capsys.readouterr().out


def test_cli_without_input_mode_is_an_error(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _run_cli(monkeypatch, "--out", str(tmp_path / "m.json"))
