"""Stage 2: per-frame calibration, pitch geometry and its persistence.

Moved verbatim out of backend/pipeline/runner.py; run_pipeline() there is
the orchestrator that calls these in order.
"""


from __future__ import annotations

import logging
import os
from datetime import datetime

from sqlalchemy.orm import Session

from ai.computer_vision.detectors import DetectorBundle
from ai.computer_vision.frame_data import (
    BallSource,
    CalibrationSource,
    CalibrationState,
    CameraState,
    FrameData,
    interpolate_ball_gaps,
)
from ai.computer_vision.tactical_analysis.constants import (
    PITCH_LENGTH_M,
)
from backend.database.models import (
    CalibrationSourceKind,
    CalibrationStatus,
    Match,
    MetricMethod,
)
from backend.pipeline.results import PipelineAssetError

logger = logging.getLogger(__name__)

# Field region and goal-mouth geometry are sampled every Nth frame rather
# than every frame. Both are near-static compared with the ball: the pitch
# does not move, and a goal enters/leaves shot over tens of frames, not
# between consecutive ones. The last result is carried forward in between.
# The ball is NOT strided -- a missed ball frame is unrecoverable, which is
# the whole reason models.yaml runs it at conf=0.10.
FIELD_DETECT_STRIDE = int(os.getenv("FIELD_DETECT_STRIDE", "25"))


GOALPOST_DETECT_STRIDE = int(os.getenv("GOALPOST_DETECT_STRIDE", "25"))


# Calibration retry policy. THIS ONE CHANGES OUTPUT -- it is not a pure
# performance knob like the strides above, and the note below is the reason
# it is written down rather than tuned quietly.
#
# AutoCalibrator's own policy is "never solved -> attempt on every frame
# until one succeeds" (see its class docstring). On footage the calibration
# model cannot solve at all, that spends one full pose-model inference per
# frame, forever, and returns nothing. Measured on samples/sample_15s.mp4:
# 32.92s of a 132.27s run (25% of total wall-clock) across 375 attempts,
# 0 accepted, 375 rejected_invalid.
#
# After CALIBRATION_FAILURE_BACKOFF_AFTER consecutive failures, the model is
# retried only every CALIBRATION_RETRY_STRIDE-th frame instead of every
# frame. Backing off rather than giving up permanently is deliberate: a clip
# that cannot be calibrated during a tight shot may become solvable after a
# cut to a wide one, and a hard stop would never find that.
#
# WHAT THIS COSTS, stated plainly: a calibration whose first success would
# have landed inside a backoff gap is now found up to
# CALIBRATION_RETRY_STRIDE frames later, and skipped frames carry
# CalibrationState.unavailable() instead of the model's own invalid_reason.
# Set CALIBRATION_FAILURE_BACKOFF_AFTER=0 to restore the previous
# attempt-every-frame behaviour exactly.
CALIBRATION_FAILURE_BACKOFF_AFTER = int(os.getenv("CALIBRATION_FAILURE_BACKOFF_AFTER", "50"))


CALIBRATION_RETRY_STRIDE = int(os.getenv("CALIBRATION_RETRY_STRIDE", "25"))


CALIBRATION_DIR = os.getenv("CALIBRATION_DIR", "calibrations")


def _load_manual_calibration(match_id: str, video_id: str,
                             frame_size: tuple[int, int] | None = None):
    """
    The MANUAL OVERRIDE path. Returns a CalibrationState or None.

    Precedence is deliberate: if an operator has calibrated this match by
    hand, that file wins over the model for every frame. The automatic
    path exists to remove the operator from the critical path, not to
    overrule them -- a human who clicked landmarks on this specific footage
    did so precisely because the model handled it badly.

    Missing calibration degrades gracefully (returns None -> the automatic
    path runs) rather than failing the job, which is the pre-existing
    contract this function inherits.
    """
    from ai.computer_vision.tactical_analysis.auto_calibration import (
        calibration_state_from_manual,
    )
    from ai.computer_vision.tactical_analysis.manual_calibration import load_calibration

    for candidate in (f"{match_id}.json", f"{video_id}.json"):
        path = os.path.join(CALIBRATION_DIR, candidate)
        if not os.path.exists(path):
            continue
        try:
            H, record = load_calibration(path)
        except Exception as exc:  # noqa: BLE001 -- a corrupt file is not fatal
            logger.warning("manual calibration %s unreadable: %s", path, exc)
            continue
        # frame_size enables the point-spread gate. An operator's clicked
        # points are usually well spread, but "usually" is not a guarantee:
        # a calibration clicked entirely inside one penalty box passes the
        # confidence gate and still extrapolates badly across the rest of
        # the pitch. The manual path is an override for the MODEL, not an
        # exemption from the geometry checks.
        state = calibration_state_from_manual(H, record, frame_size=frame_size)
        logger.info("using MANUAL calibration override from %s (valid=%s, conf=%.3f, reason=%s)",
                    path, state.valid, state.confidence, state.invalid_reason)
        return state
    return None


def _calibration_attempt_due(n_consecutive_failures: int, frame_number: int) -> bool:
    """Whether to run the calibration model on this frame.

    True on every frame until CALIBRATION_FAILURE_BACKOFF_AFTER consecutive
    failures have accumulated; after that, only every
    CALIBRATION_RETRY_STRIDE-th frame. A single success resets the counter
    (the caller does that), so normal footage never enters the backoff at
    all and behaves exactly as before.

    Setting CALIBRATION_FAILURE_BACKOFF_AFTER=0 disables the backoff and
    restores the previous attempt-every-frame behaviour.
    """
    if CALIBRATION_FAILURE_BACKOFF_AFTER <= 0:
        return True
    if n_consecutive_failures < CALIBRATION_FAILURE_BACKOFF_AFTER:
        return True
    if CALIBRATION_RETRY_STRIDE <= 1:
        return True
    return frame_number % CALIBRATION_RETRY_STRIDE == 0


def _build_frame_data(
    video_path: str,
    frames: list,
    fps: float,
    manual_override: CalibrationState | None = None,
    max_frames: int | None = None,
    device: str | None = None,
) -> tuple[list[FrameData], dict]:
    """
    Builds one FrameData per video frame: the synchronisation point between
    all five models.

    ONE pass over the decoded video. Every model instance is constructed
    once, before the loop, and reused -- constructing a YOLO() per frame
    reloads weights from disk on every iteration.

    FAILURE POLICY (item 6). No per-frame detector failure ends the run:
      - missing ball        -> ball=None this frame, then short gaps are
                               interpolated and marked `interpolated`;
                               long gaps stay `missing`
      - missing/partial field -> field_region=None; calibration loses its
                               geometric cross-check but keeps the
                               confidence gate
      - missing goalpost    -> goalposts=[] (the normal case for most of a
                               match, not an error)
      - failed calibration  -> CalibrationState with valid=False and a
                               populated invalid_reason
      - blur / occlusion / empty frame -> every detector returns its
                               "absent" sentinel; the frame still exists in
                               the output with players=[] so frame indices
                               stay contiguous
      - unreadable frame    -> loop stops at that frame rather than
                               aborting the job; frames already built are
                               kept
      - whole detector unavailable (checkpoint missing) -> logged once by
                               DetectorBundle, that signal is absent for
                               the whole run, the rest still runs

    Returns (frame_data, stats).
    """
    import cv2

    from ai.computer_vision.tactical_analysis.auto_calibration import AutoCalibrator

    bundle = DetectorBundle(device=device)
    stats: dict = {"detectors_available": bundle.available(),
                   "detector_errors": bundle.errors}

    calibrator = None
    if manual_override is None:
        try:
            calibrator = AutoCalibrator(device=device)
        except Exception as exc:  # noqa: BLE001
            # Same rule as the other detectors: no calibration model means
            # no pitch coordinates, not a failed job. Every metric gated on
            # calibration.valid will honestly report low_upstream_confidence.
            logger.warning("automatic calibration unavailable: %s", exc)
            stats["calibration_error"] = str(exc)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise PipelineAssetError(f"Could not open video for frame analysis: {video_path}")

    frame_data: list[FrameData] = []
    last_field = None
    last_goalposts: list = []
    n_total = len(frames) if max_frames is None else min(len(frames), max_frames)
    # Calibration backoff state -- see CALIBRATION_FAILURE_BACKOFF_AFTER.
    n_consecutive_calibration_failures = 0
    n_calibrations_skipped = 0

    try:
        for frame_number in range(n_total):
            ok, raw = cap.read()
            if not ok or raw is None:
                # Decoder ran out before the tracker's frame list did.
                # Keep what was built rather than discarding the run.
                logger.warning("video decode stopped at frame %d of %d",
                               frame_number, n_total)
                break

            players = [d for d in frames[frame_number]
                       if d.class_name in ("player", "goalkeeper")]

            # -- ball: every frame (see FIELD_DETECT_STRIDE's comment) --
            ball = bundle.ball.detect(raw) if bundle.ball else None

            # -- field / goalposts: strided, carried forward in between --
            if bundle.field and frame_number % FIELD_DETECT_STRIDE == 0:
                detected = bundle.field.detect(raw)
                # Carry the previous region forward on a miss rather than
                # dropping to None: the pitch has not moved, and a
                # momentarily-missed segmentation would otherwise remove
                # calibration's geometric cross-check for 25 frames.
                last_field = detected if detected is not None else last_field
            if bundle.goalpost and frame_number % GOALPOST_DETECT_STRIDE == 0:
                last_goalposts = bundle.goalpost.detect(raw)

            # -- calibration --
            if manual_override is not None:
                calibration = manual_override
                camera = CameraState()
            elif calibrator is not None:
                if _calibration_attempt_due(n_consecutive_calibration_failures,
                                            frame_number):
                    calibration, camera = calibrator.calibrate(
                        raw, frame_number, field_region=last_field
                    )
                    n_consecutive_calibration_failures = (
                        0 if calibration.valid
                        else n_consecutive_calibration_failures + 1
                    )
                else:
                    # Backed off on this frame. The camera-motion estimate is
                    # still updated: it costs ~2.25 ms against ~85.5 ms for the
                    # pose model (measured), it is persisted per calibration
                    # episode, and CameraMotionDetector compares consecutive
                    # frames -- skipping it would corrupt the motion signal on
                    # the frames that ARE attempted.
                    camera = calibrator.motion.update(raw)
                    n_calibrations_skipped += 1
                    calibration = CalibrationState.unavailable(
                        f"not attempted this frame: backed off after "
                        f"{CALIBRATION_FAILURE_BACKOFF_AFTER} consecutive failures, "
                        f"retrying every {CALIBRATION_RETRY_STRIDE} frames"
                    )
            else:
                calibration = CalibrationState.unavailable(
                    "no calibration model and no manual calibration file")
                camera = CameraState()

            frame_data.append(FrameData(
                frame_id=frame_number,
                timestamp=frame_number / fps,
                players=players,
                ball=ball,
                field_region=last_field,
                goalposts=list(last_goalposts),
                calibration=calibration,
                camera=camera,
            ))
    finally:
        cap.release()

    # Short ball gaps become `interpolated`; everything else becomes an
    # explicit `missing` observation with a running age.
    n_interpolated = interpolate_ball_gaps(frame_data)

    # Project ball + goalposts into pitch metres, on the frames where the
    # calibration is actually valid. Uses foot-point semantics for the
    # goalposts (their base is on the z=0 plane) and the ball centre.
    _project_frame_geometry(frame_data)

    n_detected = sum(1 for f in frame_data
                     if f.ball is not None and f.ball.source is BallSource.detected)
    stats.update({
        "frames": len(frame_data),
        "ball_detected": n_detected,
        "ball_interpolated": n_interpolated,
        "field_frames": sum(1 for f in frame_data if f.field_region is not None),
        "goalpost_frames": sum(1 for f in frame_data if f.goalposts),
        "calibration_valid": sum(1 for f in frame_data if f.calibration.valid),
        # How many of the valid frames were SOLVED versus carried forward
        # from an earlier solve. Logged separately because the two are not
        # equivalent evidence, and a rise in calibration_valid driven
        # entirely by reuse would otherwise be indistinguishable from a rise
        # driven by the model actually working.
        "calibration_valid_solved": sum(
            1 for f in frame_data
            if f.calibration.valid and f.calibration.source is not CalibrationSource.carried),
        "calibration_valid_carried": sum(
            1 for f in frame_data
            if f.calibration.valid and f.calibration.source is CalibrationSource.carried),
        "detector_stats": bundle.stats(),
    })
    if calibrator is not None:
        stats["calibration"] = calibrator.stats()
        stats["calibration_skipped_backoff"] = n_calibrations_skipped
        logger.info(
            "calibration: attempted=%d accepted=%d (strict=%d relaxed=%d) "
            "reused=%d smoothed=%d fallback_expired=%d "
            "rejected_invalid=%d rejected_jump=%d skipped_backoff=%d",
            calibrator.n_attempted, calibrator.n_accepted, calibrator.n_strict,
            calibrator.n_relaxed, calibrator.n_reused, calibrator.n_smoothed,
            calibrator.n_fallback_expired, calibrator.n_rejected_invalid,
            calibrator.n_rejected_jump, n_calibrations_skipped,
        )
        logger.info(
            "calibration valid frames: %d solved on-frame, %d carried forward "
            "-- a valid-frame count is not a solve count",
            stats["calibration_valid_solved"], stats["calibration_valid_carried"],
        )
    logger.info("frame synchronisation: %s", stats)
    return frame_data, stats


def _project_frame_geometry(frame_data: list[FrameData]) -> None:
    """
    Fills pitch_x_m/pitch_y_m on each frame's ball and goalposts, in place.

    Gated on `calibration.valid` -- the single flag -- rather than each
    consumer re-testing `confidence >= HOMOGRAPHY_CONFIDENCE_MIN`. Frames
    whose calibration is invalid keep pitch coordinates of None, which is
    the honest "not measured" state.
    """
    from ai.computer_vision.tactical_analysis.homography import pixels_to_pitch

    for f in frame_data:
        if not f.calibration.valid or f.calibration.H is None:
            continue
        H = f.calibration.H

        if f.ball is not None and f.ball.usable:
            try:
                bx, by = pixels_to_pitch([[f.ball.pixel_x, f.ball.pixel_y]], H)[0]
                f.ball.pitch_x_m, f.ball.pitch_y_m = float(bx), float(by)
            except Exception:  # noqa: BLE001 -- a degenerate H is not fatal
                pass

        for gp in f.goalposts:
            try:
                gx, gy = pixels_to_pitch(
                    [[gp.center()[0], gp.goal_line_y()]], H)[0]
                gp.pitch_x_m, gp.pitch_y_m = float(gx), float(gy)
                # Side is assigned from the PITCH x-coordinate, never from
                # where the goal appears in the image -- which half of the
                # frame a goal occupies depends entirely on camera angle.
                gp.side = "left" if gx < PITCH_LENGTH_M / 2.0 else "right"
            except Exception:  # noqa: BLE001
                pass


def _representative_confidence(frame_data: list[FrameData]) -> float:
    """Median confidence across frames whose calibration was VALID; 0.0 if
    none were. Median rather than max so one lucky frame cannot describe a
    run in which calibration mostly failed."""
    vals = sorted(f.calibration.confidence for f in frame_data if f.calibration.valid)
    if not vals:
        return 0.0
    return float(vals[len(vals) // 2])


def _persist_calibration_status(db: Session, match: Match,
                                frame_data: list[FrameData]) -> int:
    """
    Writes one CalibrationStatus row per calibration EPISODE.

    An episode is a maximal run of consecutive frames sharing the same
    validity, source and homography identity. Collapsing them is what keeps
    this table small enough to be useful: a static-camera stretch of 3,000
    frames carrying one homography is one row that says so, not 3,000 rows
    that each say it again.
    """
    if not frame_data:
        return 0

    def key(f: FrameData):
        return (f.calibration.valid, f.calibration.source,
                f.calibration.solved_on_frame,
                round(f.calibration.confidence, 6))

    rows: list[CalibrationStatus] = []
    start = 0
    for i in range(1, len(frame_data) + 1):
        if i < len(frame_data) and key(frame_data[i]) == key(frame_data[start]):
            continue
        f = frame_data[start]
        cal = f.calibration
        H = cal.H
        rows.append(CalibrationStatus(
            match_id=match.match_id,
            frame_start=frame_data[start].frame_id,
            frame_end=frame_data[i - 1].frame_id,
            valid=bool(cal.valid),
            invalid_reason=cal.invalid_reason,
            confidence=float(cal.confidence),
            reprojection_error_m=cal.reprojection_error_m,
            n_points=int(cal.n_points or 0),
            source=CalibrationSourceKind(cal.source.value),
            solved_on_frame=cal.solved_on_frame,
            camera_motion=f.camera.motion.value,
            camera_shift_px=f.camera.shift_px,
            homography_matrix=(H.tolist() if hasattr(H, "tolist") else H),
            # An operator's clicks are a deterministic DLT fit; the pose
            # model's keypoints are ml_trained. Neither is heuristic_proxy:
            # both carry a real, measured reprojection error.
            method=(MetricMethod.deterministic
                    if cal.source is CalibrationSource.manual
                    else MetricMethod.ml_trained),
            computed_at=datetime.utcnow(),
        ))
        start = i

    for r in rows:
        db.add(r)
    db.commit()
    logger.info("wrote %d calibration_status episode rows for match %s",
                len(rows), match.match_id)
    return len(rows)


def _estimate_fps(video_path: str) -> float:
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        return float(fps)
    except Exception:  # noqa: BLE001 - unreadable metadata falls back to 25 fps
        return 25.0


def _video_frame_size(video_path: str) -> tuple[int, int] | None:
    """(width, height) of the video, or None if it cannot be read.

    Needed by the MANUAL calibration path so CalibrationState.evaluate() can
    apply the point-spread gate: hull area is only meaningful as a fraction
    of frame area. Returns None rather than a guessed default on failure --
    evaluate() then skips the spread check, which is the honest outcome when
    the frame size is genuinely unknown, and matches the automatic path's
    behaviour in the same situation.
    """
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return (w, h) if w > 0 and h > 0 else None
    except Exception:  # noqa: BLE001 - unreadable metadata means size unknown
        return None
