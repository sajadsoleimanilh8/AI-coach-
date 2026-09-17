"""Visual calibration diagnostics for one real video frame."""
from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ai.computer_vision.detectors import FieldDetector
from ai.computer_vision.tactical_analysis.auto_calibration import AutoCalibrator, to_calibration_state
from ai.computer_vision.tactical_analysis.constants import PITCH_LENGTH_M, PITCH_WIDTH_M
from ai.computer_vision.tactical_analysis.homography import pixel_to_pitch
from backend.api.schemas import CalibrationDebugResponse
from backend.database.models import Match, MetricConfidence, MetricMethod, PlayerTracking
from backend.database.session import get_db

router = APIRouter(prefix="/api/matches", tags=["calibration_debug"])


@lru_cache(maxsize=1)
def _models() -> tuple[FieldDetector, AutoCalibrator]:
    return FieldDetector(), AutoCalibrator()


def _read_frame(video_path: str, frame_number: int) -> np.ndarray:
    capture = cv2.VideoCapture(video_path)
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise ValueError(f"could not decode frame {frame_number}")
    return frame


def render_calibration_debug(
    frame: np.ndarray,
    players: list,
    *,
    field_detector=None,
    calibrator=None,
) -> tuple[bytes, dict]:
    """Render keypoints, tracked anchors and valid projected pitch geometry.

    Invalid fits still show their detected evidence, but pitch boundaries and
    pitch-space player positions are suppressed. This is diagnostic tooling,
    not an alternate route around ``calibration.valid``.
    """
    if field_detector is None or calibrator is None:
        default_field, default_calibrator = _models()
        field_detector = field_detector or default_field
        calibrator = calibrator or default_calibrator

    canvas = frame.copy()
    field = field_detector.detect(frame)
    attempt = calibrator.calibrate_frame(frame, field_region=field)
    # frame_size enables the point-spread gate (HOMOGRAPHY_MIN_POINT_SPREAD).
    # Without it evaluate() skips that check, and this route would report a
    # clustered fit as VALID while the pipeline rejected the same frame --
    # a debug view that disagrees with production is worse than none.
    frame_h, frame_w = frame.shape[:2]
    state = to_calibration_state(attempt, field_region=field,
                                 frame_size=(frame_w, frame_h))

    if field is not None and len(field.polygon) >= 3:
        polygon = np.asarray(field.polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [polygon], True, (60, 220, 80), 2)

    for idx, (x, y) in zip(attempt.keypoint_indices, attempt.keypoints_px):
        cv2.circle(canvas, (round(x), round(y)), 5, (0, 210, 255), -1)
        cv2.putText(canvas, str(idx), (round(x) + 5, round(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 255), 1, cv2.LINE_AA)

    projected_players = []
    for player in players:
        x, y = float(player.pixel_x), float(player.pixel_y)
        cv2.circle(canvas, (round(x), round(y)), 5, (255, 210, 40), 2)
        if state.valid:
            px, py = pixel_to_pitch(x, y, state.H)
            projected_players.append({"player_id": player.player_id, "pitch_x_m": px, "pitch_y_m": py})
            cv2.putText(canvas, f"#{player.player_id} {px:.1f},{py:.1f}",
                        (round(x) + 7, round(y) + 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (255, 210, 40), 1, cv2.LINE_AA)

    if state.valid:
        pitch_lines = [
            [(0.0, 0.0), (PITCH_LENGTH_M, 0.0), (PITCH_LENGTH_M, PITCH_WIDTH_M),
             (0.0, PITCH_WIDTH_M), (0.0, 0.0)],
            [(PITCH_LENGTH_M / 2.0, 0.0), (PITCH_LENGTH_M / 2.0, PITCH_WIDTH_M)],
        ]
        inverse = np.linalg.inv(state.H)
        for line in pitch_lines:
            points = np.asarray(line, dtype=np.float32).reshape((-1, 1, 2))
            image_points = cv2.perspectiveTransform(points, inverse).reshape((-1, 2))
            cv2.polylines(canvas, [np.rint(image_points).astype(np.int32)], False, (255, 80, 40), 2)

    confidence_enum = (
        MetricConfidence.normal if state.valid else MetricConfidence.low_upstream_confidence
    )
    status_text = (
        f"VALID conf={state.confidence:.3f}" if state.valid
        else f"INVALID: {state.invalid_reason or attempt.reason or 'no valid homography'}"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (15, 15, 15), -1)
    cv2.putText(canvas, status_text[:150], (10, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (90, 230, 90) if state.valid else (80, 120, 255), 1, cv2.LINE_AA)

    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise ValueError("could not encode calibration debug overlay")
    metadata = {
        "calibration_valid": state.valid,
        "calibration_confidence": state.confidence,
        "calibration_metric_method": MetricMethod.deterministic.value,
        "calibration_metric_confidence": confidence_enum.value,
        "invalid_reason": state.invalid_reason or attempt.reason,
        "keypoint_detection_method": MetricMethod.ml_trained.value,
        "keypoint_detection_confidence": attempt.detection_confidence,
        "n_keypoints": attempt.n_keypoints_visible,
        "field_detected": field is not None,
        "field_detection_confidence": field.confidence if field is not None else None,
        "projected_players": projected_players,
        "projection_suppressed": not state.valid,
    }
    return encoded.tobytes(), metadata


@router.get("/{match_id}/calibration-debug/{frame_number}", response_model=CalibrationDebugResponse)
def calibration_debug(match_id: str, frame_number: int, db: Session = Depends(get_db)):
    match = db.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Match not found")
    if frame_number < 0:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="frame_number must be non-negative")
    if not Path(match.video_path).is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Match video is missing")

    players = (
        db.query(PlayerTracking)
        .filter(PlayerTracking.match_id == match_id, PlayerTracking.frame_id == frame_number)
        .all()
    )
    try:
        frame = _read_frame(match.video_path, frame_number)
        png, metadata = render_calibration_debug(frame, players)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return {
        "match_id": match_id,
        "frame_number": frame_number,
        "overlay_png_base64": base64.b64encode(png).decode("ascii"),
        **metadata,
    }
