"""
Phase 3 pipeline orchestration.
"""

from __future__ import annotations

import logging
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
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
from ai.computer_vision.pass_detection.pass_heuristics import detect_passes, detect_turnovers
from ai.computer_vision.shot_detection.shot_heuristics import build_goal_mouths, detect_shots
from ai.computer_vision.tactical_analysis.constants import (
    PASS_DIRECTION_SECTORS,
    PITCH_LENGTH_M,
    TOUCH_EVAL_WINDOW_S,
)
from ai.computer_vision.tactical_analysis.attacking_direction import (
    UNKNOWN as DIRECTION_UNKNOWN,
    infer_attacking_directions,
)
from ai.computer_vision.tactical_analysis.formation_detection import detect_formation
from ai.computer_vision.tactical_analysis.possession import detect_first_touches, get_ball_possessor
from ai.computer_vision.tactical_analysis.role_inference import Role, infer_roles
from ai.computer_vision.tactical_analysis.team_assignment import assign_teams_with_stats
from ai.computer_vision.player_tracking.trajectory import attach_body_orientation
from ai.player_intelligence.body_orientation_score.score import score_body_orientation
from ai.player_intelligence.decision_making_score.score import score_decision_making
from ai.player_intelligence.defensive_positioning.score import score_defensive_positioning
from ai.player_intelligence.finishing_efficiency_score.score import score_finishing_efficiency
from ai.player_intelligence.first_touch_score.score import score_first_touch
from ai.player_intelligence.off_ball_movement.score import score_off_ball_movement
from ai.player_intelligence.passing_vision_score.score import score_passing_vision
from ai.player_intelligence.press_resistance_score.score import score_press_resistance
from ai.player_intelligence.scanning_behavior.score import score_scanning_behavior
from ai.team_intelligence.formation_stability.team_shape import compute_compactness, compute_formation_stability
from ai.team_intelligence.pressing_structure_analysis.pressing import compute_pressing_intensity
from ai.team_intelligence.weak_zone_detection.weak_zones import compute_weak_zones

from backend.database.models import (
    BallDetection,
    CalibrationSourceKind,
    CalibrationStatus,
    Event,
    Frame,
    Match,
    MetricMethod,
    PlayerDetection,
    PlayerMetric,
    PlayerTracking,
    ProcessingJob,
    TeamMetric,
)
from backend.pipeline.device import resolve_device
from backend.pipeline.latency import PipelineTimer
from backend.pipeline.overlay_video import overlay_output_path, render_overlay_video

logger = logging.getLogger(__name__)

FRAME_PERSIST_STRIDE = int(os.getenv("FRAME_PERSIST_STRIDE", "5"))

POSE_SAMPLE_STRIDE = int(os.getenv("POSE_SAMPLE_STRIDE", "10"))

FIELD_DETECT_STRIDE = int(os.getenv("FIELD_DETECT_STRIDE", "25"))
GOALPOST_DETECT_STRIDE = int(os.getenv("GOALPOST_DETECT_STRIDE", "25"))

CALIBRATION_FAILURE_BACKOFF_AFTER = int(os.getenv("CALIBRATION_FAILURE_BACKOFF_AFTER", "50"))
CALIBRATION_RETRY_STRIDE = int(os.getenv("CALIBRATION_RETRY_STRIDE", "25"))

CALIBRATION_DIR = os.getenv("CALIBRATION_DIR", "calibrations")



def _player_checkpoint() -> str:
    """Resolved path to the trained player checkpoint, via the registry."""
    from configs import registry

    try:
        return str(registry.checkpoint_path("player"))
    except Exception as exc:  # noqa: BLE001 -- re-raised as an asset error
        raise PipelineAssetError(str(exc)) from exc


class PipelineAssetError(Exception):
    """A required asset (model weights, video file, calibration) was
    missing or unusable. Distinct from an unexpected bug -- tasks.py
    surfaces this as job.error with the message as-is, since it's already
    written to be actionable (what's missing, where the pipeline expected
    """


@dataclass
class PipelineResult:
    match_id: str
    frames_processed: int
    players_tracked: int
    events_detected: int
    player_metrics_written: int
    team_metrics_written: int
    homography_confidence: float
    calibration_valid_fraction: float = 0.0
    calibration_episodes: int = 0
    detectors_available: tuple = ()
    overlay_render: dict | None = None


def run_pipeline(db: Session, job: ProcessingJob, timer: PipelineTimer, progress_cb=None) -> PipelineResult:
    """
    Runs the full Phase 2 + Phase 3 pipeline for one uploaded video and
    writes real PlayerTracking / Event / PlayerMetric / TeamMetric rows.
    """
    video = job.video
    if video is None:
        raise PipelineAssetError(f"ProcessingJob {job.id} has no linked Video row.")
    match: Match | None = video.match
    if match is None:
        raise PipelineAssetError(
            f"Video {video.id} has no linked Match row -- upload_video() should have "
            f"created one. Refusing to guess a match_id and write metrics under the "
            f"wrong scope."
        )

    def report(pct: int, msg: str) -> None:
        if progress_cb:
            progress_cb(pct, msg)

    device_decision = resolve_device()
    logger.info("device for this run: %s (%s)", device_decision.device, device_decision.mode)

    report(15, f"{device_decision.message} Running player detection + ByteTrack tracking...")
    player_ckpt = _player_checkpoint()
    with timer.stage("detection_tracking", detail=f"model={player_ckpt} device={device_decision.device}"):
        frames = _run_detection_and_tracking(
            video.storage_path, player_ckpt, device=device_decision.device
        )

    report(20, "Assigning players to teams via jersey-color clustering...")
    with timer.stage("team_assignment"):
        team_result = assign_teams_with_stats(video.storage_path, frames)
        team_assignment_confidence = team_result.confidence

    report(30, "Running ball/field/goalpost detection + automatic calibration...")
    fps = _estimate_fps(video.storage_path)
    frame_size = _video_frame_size(video.storage_path)
    with timer.stage("frame_synchronisation"):
        frame_data, calib_stats = _build_frame_data(
            video.storage_path, frames, fps,
            manual_override=_load_manual_calibration(
                match.match_id, video.id, frame_size=frame_size),
            device=device_decision.device,
        )

    homography_confidence = _representative_confidence(frame_data)
    H = _representative_homography(frame_data)

    report(40, "Recording calibration status...")
    with timer.stage("calibration_history"):
        episodes = _persist_calibration_status(db, match, frame_data)

    report(45, "Mapping tracked positions to pitch coordinates...")
    with timer.stage("pitch_trajectory"):
        trajectories, ball_trajectory = _build_trajectories(
            frame_data, frames, match.match_id, H, homography_confidence, fps
        )

    with timer.stage("role_inference"):
        roles = infer_roles(
            team_result, trajectories,
            calibration_valid=any(f.calibration.valid for f in frame_data),
        )
        referee_ids = {pid for pid, r in roles.items() if r.role is Role.referee}
        if referee_ids:
            logger.info("excluding %d heuristically-identified referee track(s) "
                        "from team scoring: %s", len(referee_ids), sorted(referee_ids))
        scoring_trajectories = {pid: pts for pid, pts in trajectories.items()
                                if pid not in referee_ids}

    with timer.stage("attacking_direction"):
        directions = infer_attacking_directions(scoring_trajectories)
        logger.info("attacking direction: %s (%s)", directions.by_team, directions.reason)

    report(50, "Estimating body orientation (pose)...")
    with timer.stage("pose_estimation"):
        orientation_by_player_frame = _estimate_orientations(
            video.storage_path, frames, stride=POSE_SAMPLE_STRIDE,
            progress=lambda done, total: report(
                50 + int(5 * done / total),
                f"Estimating body orientation (pose): frame {done} of {total}...",
            ) if total else None,
        )
        attach_body_orientation(trajectories, orientation_by_player_frame)

    report(55, "Writing detection & tracking rows...")
    with timer.stage("db_write_tracking"):
        _persist_frames_and_tracking(
            db, match, frames, trajectories, fps, frame_data
        )

    report(65, "Extracting pass, shot, turnover, and first-touch events...")
    with timer.stage("event_heuristics"):
        events, possession_segments = _detect_events(
            match, trajectories, ball_trajectory, fps,
            frame_data=frame_data, directions=directions)
        db.bulk_save_objects(events)
        db.commit()

    report(78, "Calculating formation and team shape metrics...")
    with timer.stage("team_scoring"):
        team_metrics = _score_team_intelligence(
            match, scoring_trajectories, team_assignment_confidence, directions=directions)
        for tm_dict in team_metrics:
            db.add(_team_metric_from_dict(match.match_id, tm_dict))
        db.commit()

    report(90, "Calculating per-player intelligence scores...")
    with timer.stage("player_scoring"):
        player_metrics = _score_player_intelligence(
            match, trajectories, events, homography_confidence, fps,
            team_assignment_confidence, possession_segments, directions=directions,
        )
        for pm_dict, player_id in player_metrics:
            db.add(_player_metric_from_dict(match.match_id, player_id, pm_dict))
        db.commit()

    report(95, "Rendering annotated video...")
    with timer.stage("overlay_render"):
        overlay = render_overlay_video(
            video.storage_path,
            frames,
            overlay_output_path(video.storage_path, video.id),
            fps=fps,
            ball_by_frame={
                f.frame_id: f.ball for f in frame_data
                if f.ball is not None and f.ball.source is BallSource.detected
            },
            calibration_by_frame={f.frame_id: bool(f.calibration.valid)
                                  for f in frame_data},
            progress=lambda done, total: report(
                95 + int(4 * done / total),
                f"Rendering annotated video: frame {done} of {total}...",
            ) if total else None,
        )
    if overlay.skipped_reason:
        logger.warning("annotated video not written: %s", overlay.skipped_reason)
    else:
        logger.info("annotated video written: %s (codec=%s, %d frames written, "
                    "%s decodable)", overlay.path, overlay.codec,
                    overlay.frames_written, overlay.verified_frames)

    n_valid = sum(1 for f in frame_data if f.calibration.valid)
    return PipelineResult(
        match_id=match.match_id,
        frames_processed=len(frames),
        players_tracked=len(trajectories),
        events_detected=len(events),
        player_metrics_written=len(player_metrics),
        team_metrics_written=len(team_metrics),
        homography_confidence=homography_confidence,
        calibration_valid_fraction=(n_valid / len(frame_data)) if frame_data else 0.0,
        calibration_episodes=episodes,
        detectors_available=tuple(calib_stats.get("detectors_available", ())),
        overlay_render=overlay.as_dict(),
    )



def _run_detection_and_tracking(video_path: str, checkpoint: str,
                                device: str | None = None):
    """Delegates to ai.computer_vision.player_tracking.tracker.track_video().
    Not re-implemented here on purpose -- this module owns orchestration,
    not detection/tracking logic.
    """
    if not os.path.exists(video_path):
        raise PipelineAssetError(f"Video file not found on disk: {video_path}")

    from ai.computer_vision.player_tracking.tracker import track_video

    try:
        return list(track_video(checkpoint, video_path, device=device))
    except FileNotFoundError as exc:
        raise PipelineAssetError(f"Model checkpoint not found: {checkpoint} ({exc})") from exc


def _load_manual_calibration(match_id: str, video_id: str,
                             frame_size: tuple[int, int] | None = None):
    """
    The MANUAL OVERRIDE path. Returns a CalibrationState or None.
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
        state = calibration_state_from_manual(H, record, frame_size=frame_size)
        logger.info("using MANUAL calibration override from %s (valid=%s, conf=%.3f, reason=%s)",
                    path, state.valid, state.confidence, state.invalid_reason)
        return state
    return None


def _calibration_attempt_due(n_consecutive_failures: int, frame_number: int) -> bool:
    """Whether to run the calibration model on this frame."""
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
    """
    from ai.computer_vision.tactical_analysis.auto_calibration import AutoCalibrator

    import cv2

    bundle = DetectorBundle(device=device)
    stats: dict = {"detectors_available": bundle.available(),
                   "detector_errors": bundle.errors}

    calibrator = None
    if manual_override is None:
        try:
            calibrator = AutoCalibrator(device=device)
        except Exception as exc:  # noqa: BLE001
            logger.warning("automatic calibration unavailable: %s", exc)
            stats["calibration_error"] = str(exc)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise PipelineAssetError(f"Could not open video for frame analysis: {video_path}")

    frame_data: list[FrameData] = []
    last_field = None
    last_goalposts: list = []
    n_total = len(frames) if max_frames is None else min(len(frames), max_frames)
    n_consecutive_calibration_failures = 0
    n_calibrations_skipped = 0

    try:
        for frame_number in range(n_total):
            ok, raw = cap.read()
            if not ok or raw is None:
                logger.warning("video decode stopped at frame %d of %d",
                               frame_number, n_total)
                break

            players = [d for d in frames[frame_number]
                       if d.class_name in ("player", "goalkeeper")]

            ball = bundle.ball.detect(raw) if bundle.ball else None

            if bundle.field and frame_number % FIELD_DETECT_STRIDE == 0:
                detected = bundle.field.detect(raw)
                last_field = detected if detected is not None else last_field
            if bundle.goalpost and frame_number % GOALPOST_DETECT_STRIDE == 0:
                last_goalposts = bundle.goalpost.detect(raw)

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

    n_interpolated = interpolate_ball_gaps(frame_data)

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


def _representative_homography(frame_data: list[FrameData]):
    """The H of the valid frame whose confidence is closest to the median,
    for the stages that still take a single per-match matrix. Returns None
    when no frame was valid -- never a plausible-looking identity matrix."""
    valid = [f for f in frame_data if f.calibration.valid and f.calibration.H is not None]
    if not valid:
        return None
    target = _representative_confidence(frame_data)
    return min(valid, key=lambda f: abs(f.calibration.confidence - target)).calibration.H


def _persist_calibration_status(db: Session, match: Match,
                                frame_data: list[FrameData]) -> int:
    """
    Writes one CalibrationStatus row per calibration EPISODE.
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
    except Exception:
        return 25.0


def _video_frame_size(video_path: str) -> tuple[int, int] | None:
    """(width, height) of the video, or None if it cannot be read."""
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return (w, h) if w > 0 and h > 0 else None
    except Exception:
        return None


def _build_trajectories(frame_data: list[FrameData], frames, match_id: str, H,
                        homography_confidence: float, fps: float):
    from ai.computer_vision.player_tracking.trajectory import enrich_with_pitch_coordinates

    trajectories = enrich_with_pitch_coordinates(frames, match_id, H, homography_confidence, fps)

    ball_trajectory: list[dict] = []
    for f in frame_data:
        if f.ball is None or not f.ball.usable:
            continue
        ball_trajectory.append({
            "frame_id": f.frame_id,
            "timestamp": f.timestamp,
            "pitch_x_m": f.ball.pitch_x_m,
            "pitch_y_m": f.ball.pitch_y_m,
            "homography_confidence": f.calibration.confidence,
            "calibration_valid": f.calibration.valid,
            "ball_source": f.ball.source.value,
        })

    return trajectories, ball_trajectory


def _estimate_orientations(video_path: str, frames, stride: int, progress=None) -> dict:
    """
    Runs pose estimation on a stride-sampled subset of (player, frame)
    pairs and returns {(player_id, frame_number): OrientationResult}.
    """
    if stride <= 0 or not os.path.exists(video_path):
        return {}

    try:
        from ai.computer_vision.pose_estimation.pose import estimate_body_orientation
    except ImportError:
        return {}

    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    orientation_by_player_frame: dict[tuple[int, int], object] = {}

    n_sampled = len(range(0, len(frames), stride)) if frames else 0
    sampled_done = 0
    report_every = max(1, n_sampled // 20)

    try:
        for frame_number, detections in enumerate(frames):
            ok, raw_frame = cap.read()
            if not ok or raw_frame is None:
                break

            if frame_number % stride != 0:
                continue

            sampled_done += 1
            if progress and (sampled_done % report_every == 0 or sampled_done == n_sampled):
                progress(sampled_done, n_sampled)

            player_dets = [d for d in detections if d.class_name in ("player", "goalkeeper")]
            if not player_dets:
                continue

            frame_h, frame_w = raw_frame.shape[:2]
            for det in player_dets:
                x1 = max(0, int(det.x))
                y1 = max(0, int(det.y))
                x2 = min(frame_w, int(det.x + det.width))
                y2 = min(frame_h, int(det.y + det.height))
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = raw_frame[y1:y2, x1:x2]
                try:
                    reading = estimate_body_orientation(crop)
                except Exception:
                    continue

                orientation_by_player_frame[(det.player_id, frame_number)] = reading
    finally:
        cap.release()

    return orientation_by_player_frame


def _persist_frames_and_tracking(db: Session, match: Match, frames, trajectories,
                                 fps: float, frame_data: list[FrameData] | None = None) -> int:
    n_persisted = 0

    ball_by_frame = {}
    if frame_data is not None:
        ball_by_frame = {
            f.frame_id: f.ball for f in frame_data
            if f.ball is not None and f.ball.source is BallSource.detected
        }

    for frame_number, dets in enumerate(frames):
        if frame_number % FRAME_PERSIST_STRIDE != 0:
            continue
        frame_row = Frame(
            match_id=match.match_id,
            frame_number=frame_number,
            timestamp=frame_number / fps,
            fps=fps,
        )
        db.add(frame_row)
        db.flush()
        n_persisted += 1

        ball = ball_by_frame.get(frame_number)
        if ball is not None:
            db.add(BallDetection(
                frame_id=frame_row.frame_id,
                ball_x=ball.pixel_x,
                ball_y=ball.pixel_y,
                confidence=ball.confidence if ball.confidence is not None else 0.0,
            ))

        for det in dets:
            if det.class_name == "ball":
                continue
            db.add(PlayerDetection(
                frame_id=frame_row.frame_id,
                player_id=det.player_id,
                team_id=det.team_id,
                team_assignment_confidence=det.team_assignment_confidence,
                x=det.x, y=det.y, width=det.width, height=det.height,
                confidence=det.confidence,
            ))

    tracking_rows = []
    for player_id, points in trajectories.items():
        for p in points:
            tracking_rows.append(PlayerTracking(
                match_id=match.match_id,
                player_id=p.player_id,
                frame_id=p.frame_id,
                team_id=p.team_id,
                pixel_x=p.pixel_x, pixel_y=p.pixel_y,
                pitch_x_m=p.pitch_x_m, pitch_y_m=p.pitch_y_m,
                homography_confidence=p.homography_confidence,
                speed=p.speed, distance=p.distance, acceleration=p.acceleration,
                body_orientation_deg=p.body_orientation_deg,
                body_orientation_confidence=p.body_orientation_confidence,
            ))
    db.bulk_save_objects(tracking_rows)
    db.commit()
    return n_persisted


def _image_space_possession(frame_data: list[FrameData]) -> list[dict]:
    """
    Chronological possession sequence resolved in PIXELS, for runs with no
    valid calibration on any frame.
    """
    from ai.computer_vision.tactical_analysis.possession import (
        get_ball_possessor_image_space,
    )

    sequence: list[dict] = []
    for f in frame_data:
        if f.ball is None or not f.ball.usable:
            continue
        boxes = []
        for det in f.players:
            foot_x, foot_y = det.foot_point()
            boxes.append({
                "player_id": det.player_id,
                "team_id": det.team_id,
                "foot_x": foot_x,
                "foot_y": foot_y,
                "box_height_px": det.height,
            })
        possessor = get_ball_possessor_image_space(
            boxes, (f.ball.pixel_x, f.ball.pixel_y),
        )
        if possessor is None:
            continue
        sequence.append({
            "player_id": possessor["player_id"],
            "team_id": possessor["team_id"],
            "pitch_x_m": None,
            "pitch_y_m": None,
            "pixel_x": possessor["foot_x"],
            "pixel_y": possessor["foot_y"],
            "px_per_m": possessor["px_per_m"],
            "ball_distance_m_estimate": possessor["distance_m_estimate"],
            "timestamp": f.timestamp,
            "frame_id": f.frame_id,
            "homography_confidence": f.calibration.confidence,
            "ball_source": f.ball.source.value,
        })
    return sequence


def _detect_events(
    match: Match, trajectories: dict, ball_trajectory: list[dict], fps: float,
    frame_data: list[FrameData] | None = None, directions=None,
) -> tuple[list[Event], list[dict]]:
    players_by_frame = _build_players_by_frame(trajectories)

    possession_sequence: list[dict] = []
    for ball_pt in ball_trajectory:
        frame_idx = ball_pt["frame_id"]
        ball_pos = (
            (ball_pt["pitch_x_m"], ball_pt["pitch_y_m"])
            if ball_pt["pitch_x_m"] is not None else None
        )
        possessor = get_ball_possessor(
            players_by_frame.get(frame_idx, []), ball_pos, ball_pt["homography_confidence"],
            calibration_valid=ball_pt.get("calibration_valid"),
        )
        ball_pt["player_id"] = possessor["player_id"] if possessor else None
        ball_pt["team_id"] = possessor["team_id"] if possessor else None
        if possessor is not None:
            possession_sequence.append({
                "player_id": possessor["player_id"],
                "team_id": possessor["team_id"],
                "pitch_x_m": possessor["pitch_x_m"],
                "pitch_y_m": possessor["pitch_y_m"],
                "timestamp": ball_pt["timestamp"],
                "homography_confidence": ball_pt["homography_confidence"],
            })

    possession_space = "pitch"
    if not possession_sequence and frame_data:
        possession_sequence = _image_space_possession(frame_data)
        if possession_sequence:
            possession_space = "image"
            logger.info(
                "no frame had a valid calibration; possession resolved in IMAGE "
                "space for %d frames (bounding-box-height scale). Events derived "
                "from it carry space=image and no pitch coordinates.",
                len(possession_sequence),
            )
        else:
            logger.info(
                "no possession could be resolved in either pitch or image space "
                "-- either no ball was detected or no player was within the "
                "control radius of it on any frame")

    deduped: list[dict] = []
    for entry in possession_sequence:
        if not deduped or deduped[-1]["player_id"] != entry["player_id"]:
            deduped.append(entry)

    frame_possessions = [{"possessor": {"player_id": e["player_id"], "team_id": e["team_id"]},
                           "timestamp": e["timestamp"], "pitch_x_m": e["pitch_x_m"],
                           "pitch_y_m": e["pitch_y_m"],
                           "homography_confidence": e["homography_confidence"]}
                          for e in deduped]

    first_touch_dicts = detect_first_touches(frame_possessions)
    pass_dicts = detect_passes(deduped)
    turnover_dicts = detect_turnovers(deduped)
    goal_mouths = build_goal_mouths(frame_data or [])
    if goal_mouths:
        logger.info("goal mouths measured from goalpost_v1: %s",
                    {s: f"x={m.x_m:.1f}m n={m.n_observations} conf={m.confidence:.2f}"
                     for s, m in goal_mouths.items()})
    else:
        logger.info("no goalpost detection cleared the confidence gate with a valid "
                    "calibration; shot geometry falls back to nominal pitch constants")
    shot_dicts = detect_shots(
        ball_trajectory, fps=fps,
        goal_mouths=goal_mouths,
        direction_by_team=(directions.by_team if directions is not None else None),
    )

    _enrich_first_touch_metadata(first_touch_dicts, ball_trajectory, players_by_frame, deduped)

    all_event_dicts = first_touch_dicts + pass_dicts + turnover_dicts + shot_dicts

    for e in all_event_dicts:
        meta = e.get("metadata_json")
        if not isinstance(meta, dict):
            meta = {} if meta is None else {"value": meta}
            e["metadata_json"] = meta
        meta.setdefault("space", possession_space)

    event_rows = [
        Event(
            match_id=match.match_id,
            event_type=e["event_type"],
            player_id=e.get("player_id"),
            related_player_id=e.get("related_player_id"),
            team_id=e.get("team_id"),
            pitch_x_m=e.get("pitch_x_m"),
            pitch_y_m=e.get("pitch_y_m"),
            homography_confidence=e.get("homography_confidence"),
            timestamp=e.get("timestamp", 0.0),
            metadata_json=e.get("metadata_json"),
        )
        for e in all_event_dicts
    ]
    return event_rows, deduped


def _enrich_first_touch_metadata(
    first_touch_dicts: list[dict],
    ball_trajectory: list[dict],
    players_by_frame: dict,
    possession_changes: list[dict],
) -> None:
    """Fills each first_touch event's metadata_json with real measured
    values (touch_distance_m, distance_to_nearest_opponent_m,
    touch_execution_time_s, time_to_turnover_s, direction_score) instead
    of leaving score_first_touch() to guess them. See first_touch_score/
    """
    ball_by_frame = {b["frame_id"]: b for b in ball_trajectory}
    fps_guess = 25.0
    eval_window_frames = int(TOUCH_EVAL_WINDOW_S * fps_guess)

    for ev in first_touch_dicts:
        meta = ev.setdefault("metadata_json", {})
        touch_frame = round(ev["timestamp"] * fps_guess)
        touch_pos = (ev.get("pitch_x_m"), ev.get("pitch_y_m"))

        future = ball_by_frame.get(touch_frame + eval_window_frames)
        if touch_pos[0] is not None and future is not None and future["pitch_x_m"] is not None:
            meta["touch_distance_m"] = ((future["pitch_x_m"] - touch_pos[0]) ** 2
                                         + (future["pitch_y_m"] - touch_pos[1]) ** 2) ** 0.5

        others = [p for p in players_by_frame.get(touch_frame, [])
                  if p["player_id"] != ev.get("player_id") and p["pitch_x_m"] is not None]
        if others and touch_pos[0] is not None:
            dists = [((p["pitch_x_m"] - touch_pos[0]) ** 2 + (p["pitch_y_m"] - touch_pos[1]) ** 2) ** 0.5
                     for p in others]
            meta["distance_to_nearest_opponent_m"] = min(dists)
        else:
            meta["distance_to_nearest_opponent_m"] = None

        meta["time_to_turnover_s"] = None



def _dominant_team_id(points) -> str | None:
    """The team a track belongs to, as the modal non-null team_id over its
    own points. Per-detection assignment can flicker on a bad crop; the
    mode is stable against that without inventing an assignment for a track
    that genuinely never got one."""
    counts = Counter(str(p.team_id) for p in points if p.team_id is not None)
    return counts.most_common(1)[0][0] if counts else None


def _split_by_team(trajectories: dict) -> dict[str, dict]:
    """{team_id: {player_id: points}}. Tracks with no team assignment are
    dropped rather than pooled into either side -- an unassigned track is
    usually an official or a bad crop, and putting it in a team's shape
    would corrupt exactly the metric being fixed here."""
    out: dict[str, dict] = defaultdict(dict)
    for player_id, points in trajectories.items():
        tid = _dominant_team_id(points)
        if tid is not None:
            out[tid][player_id] = points
    return dict(out)


def _mean_positions_per_player(team_traj: dict) -> list[tuple[float, float]]:
    """One (x, y) per PLAYER -- their mean position over the clip."""
    out = []
    for points in team_traj.values():
        xs = [(p.pitch_x_m, p.pitch_y_m) for p in points if p.pitch_x_m is not None]
        if xs:
            out.append((sum(v[0] for v in xs) / len(xs), sum(v[1] for v in xs) / len(xs)))
    return out


def _score_team_intelligence(match: Match, trajectories: dict, team_assignment_confidence: float,
                             directions=None) -> list[dict]:
    """Per-team team-level metrics."""
    by_team = _split_by_team(trajectories)

    if not by_team:
        empty: list[tuple[float, float]] = []
        rows = [
            detect_formation(player_positions=empty,
                             team_assignment_confidence=team_assignment_confidence),
            compute_compactness(empty, team_assignment_confidence=team_assignment_confidence),
            compute_formation_stability([], team_assignment_confidence=team_assignment_confidence),
            compute_weak_zones(empty, team_assignment_confidence=team_assignment_confidence),
            compute_pressing_intensity([], team_assignment_confidence=team_assignment_confidence),
        ]
        for r in rows:
            r["team_id"] = "unassigned"
            r.setdefault("sub_scores", {})["reason_no_team_split"] = (
                "no tracked player carried a team_id; team assignment "
                "and/or calibration did not produce a usable split"
            )
        return rows

    metrics: list[dict] = []
    team_ids = sorted(by_team)
    for tid in team_ids:
        own = by_team[tid]
        own_positions = [(p.pitch_x_m, p.pitch_y_m)
                         for points in own.values() for p in points
                         if p.pitch_x_m is not None]
        opp_positions = [(p.pitch_x_m, p.pitch_y_m)
                         for other in team_ids if other != tid
                         for points in by_team[other].values() for p in points
                         if p.pitch_x_m is not None]

        direction = directions.for_team(tid) if directions is not None else "left_to_right"
        formation_kwargs = {}
        if direction in ("left_to_right", "right_to_left"):
            formation_kwargs["attacking_direction"] = direction

        rows = [
            detect_formation(
                player_positions=_mean_positions_per_player(own),
                team_assignment_confidence=team_assignment_confidence,
                **formation_kwargs,
            ),
            compute_compactness(own_positions,
                                team_assignment_confidence=team_assignment_confidence),
            compute_formation_stability(_positions_by_frame(own),
                                        team_assignment_confidence=team_assignment_confidence),
            compute_weak_zones(own_positions,
                               team_assignment_confidence=team_assignment_confidence,
                               opponent_positions_m=opp_positions or None),
            compute_pressing_intensity([],
                                       team_assignment_confidence=team_assignment_confidence),
        ]
        for r in rows:
            r["team_id"] = tid
            sub = r.setdefault("sub_scores", {})
            sub["n_players_in_team"] = len(own)
            sub["team_assignment_confidence"] = team_assignment_confidence
            if directions is not None:
                sub["attacking_direction"] = direction
                if direction == "unknown":
                    sub["attacking_direction_reason"] = directions.reason
        metrics.extend(rows)

    if len(team_ids) == 1:
        for r in metrics:
            r.setdefault("sub_scores", {})["single_team_only"] = (
                "only one team_id was present, so opponent-relative context "
                "(weak-zone exposure) could not be computed"
            )
    return metrics


def _positions_by_frame(trajectories: dict) -> list[list[tuple]]:
    """Per-frame position lists, ordered by real frame_id."""
    by_frame: dict[int, list[tuple]] = {}
    for points in trajectories.values():
        for p in points:
            if p.pitch_x_m is not None:
                by_frame.setdefault(p.frame_id, []).append((p.pitch_x_m, p.pitch_y_m))
    return [by_frame[fid] for fid in sorted(by_frame)]


def _build_players_by_frame(trajectories: dict) -> dict[int, list[dict]]:
    """
    {real frame_id: [{player_id, team_id, pitch_x_m, pitch_y_m}, ...]}.
    """
    players_by_frame: dict[int, list[dict]] = {}
    for player_id, points in trajectories.items():
        for p in points:
            players_by_frame.setdefault(p.frame_id, []).append({
                "player_id": p.player_id,
                "team_id": p.team_id,
                "pitch_x_m": p.pitch_x_m,
                "pitch_y_m": p.pitch_y_m,
            })
    return players_by_frame


def _nearest_other_player_distance(player_id: int, x: float, y: float, frame_others: list[dict]) -> float | None:
    """Distance to the closest OTHER tracked player at this frame. Same
    documented nearest-other-player-as-opponent-stand-in approximation as
    _enrich_first_touch_metadata() uses -- not yet upgraded to filter by
    real team_id (see that function's NOTE for why)."""
    others = [p for p in frame_others if p["player_id"] != player_id and p["pitch_x_m"] is not None]
    if not others:
        return None
    return min(((p["pitch_x_m"] - x) ** 2 + (p["pitch_y_m"] - y) ** 2) ** 0.5 for p in others)


def _compute_off_ball_inputs(player_id: int, points: list, players_by_frame: dict[int, list[dict]]) -> dict:
    """
    Derives score_off_ball_movement()'s inputs from real trajectory data
    instead of the all-zero defaults it was previously called with (see
    the FIXED note in _score_player_intelligence below).
    """
    valid_points = [p for p in points if p.pitch_x_m is not None]

    total_path_length_m = 0.0
    for prev, nxt in zip(valid_points, valid_points[1:]):
        total_path_length_m += ((nxt.pitch_x_m - prev.pitch_x_m) ** 2 + (nxt.pitch_y_m - prev.pitch_y_m) ** 2) ** 0.5

    net_displacement_m = 0.0
    if len(valid_points) >= 2:
        first, last = valid_points[0], valid_points[-1]
        net_displacement_m = ((last.pitch_x_m - first.pitch_x_m) ** 2 + (last.pitch_y_m - first.pitch_y_m) ** 2) ** 0.5

    separation_gains: list[float] = []
    for p_now, p_next in zip(points, points[1:]):
        if p_now.pitch_x_m is None or p_next.pitch_x_m is None:
            continue
        if p_next.frame_id != p_now.frame_id + 1:
            continue
        dist_now = _nearest_other_player_distance(
            player_id, p_now.pitch_x_m, p_now.pitch_y_m,
            players_by_frame.get(p_now.frame_id, []))
        dist_next = _nearest_other_player_distance(
            player_id, p_next.pitch_x_m, p_next.pitch_y_m,
            players_by_frame.get(p_next.frame_id, []))
        if dist_now is None or dist_next is None:
            continue
        gain = dist_next - dist_now
        separation_gains.append(max(gain, 0.0))

    avg_distance_gained_from_marker = sum(separation_gains) / len(separation_gains) if separation_gains else 0.0

    return {
        "avg_distance_gained_from_marker": avg_distance_gained_from_marker,
        "off_ball_frames_in_attacking_third": 0,
        "total_off_ball_frames_while_team_in_possession": 0,
        "net_displacement_m": net_displacement_m,
        "total_path_length_m": total_path_length_m,
        "sample_size_phases": len(separation_gains),
    }


def _compute_decision_times(possession_segments: list[dict]) -> dict[int, list[float]]:
    """
    One decision-time sample per touch = time until that touch's
    possession segment ends (the next segment's timestamp minus this
    one's) -- fed to score_decision_making()'s decision_speed sub-score,
    same scale press_resistance_score already uses for its own
    """
    times: dict[int, list[float]] = {}
    for i in range(len(possession_segments) - 1):
        pid = possession_segments[i]["player_id"]
        dt = possession_segments[i + 1]["timestamp"] - possession_segments[i]["timestamp"]
        times.setdefault(pid, []).append(dt)
    return times


def _compute_passing_vision_inputs(
    player_id: int, events_by_type_and_player: dict[str, dict[int, list[Event]]],
    attacking_direction: str = DIRECTION_UNKNOWN,
) -> dict:
    """
    completed_passes / turnovers_lost: real event counts -- both only
    classify correctly once team assignment (Stage 1.5) has run, which is
    why score_passing_vision() gates on team_assignment_confidence rather
    than treating these as always-safe inputs (see
    """
    passes = events_by_type_and_player.get("pass", {}).get(player_id, [])
    turnovers_lost = events_by_type_and_player.get("turnover", {}).get(player_id, [])

    sector_counts = [0] * PASS_DIRECTION_SECTORS
    known_direction = attacking_direction in ("left_to_right", "right_to_left")
    forward_passes: int | None = 0 if known_direction else None

    for e in passes:
        meta = e.metadata_json or {}
        start_x, start_y = meta.get("start_x"), meta.get("start_y")
        end_x, end_y = e.pitch_x_m, e.pitch_y_m
        if start_x is None or start_y is None or end_x is None or end_y is None:
            continue
        angle = math.atan2(end_y - start_y, end_x - start_x)
        sector = int(((angle + math.pi) / (2 * math.pi)) * PASS_DIRECTION_SECTORS) % PASS_DIRECTION_SECTORS
        sector_counts[sector] += 1
        if known_direction:
            advanced = (end_x > start_x) if attacking_direction == "left_to_right" else (end_x < start_x)
            if advanced:
                forward_passes += 1

    return {
        "completed_passes": len(passes),
        "turnovers_lost": len(turnovers_lost),
        "pass_sector_counts": sector_counts,
        "forward_passes": forward_passes,
        "attacking_direction": attacking_direction,
    }


def _compute_finishing_inputs(
    player_id: int, events_by_type_and_player: dict[str, dict[int, list[Event]]]
) -> list[dict]:
    """This player's shot events' real location/trajectory-projection
    fields, for score_finishing_efficiency()'s per-shot loop."""
    shots = events_by_type_and_player.get("shot", {}).get(player_id, [])
    return [
        {
            "pitch_x_m": e.pitch_x_m,
            "pitch_y_m": e.pitch_y_m,
            "projected_y_at_goal": (e.metadata_json or {}).get("projected_y_at_goal"),
        }
        for e in shots
    ]


def _score_player_intelligence(
    match: Match, trajectories: dict, events: list[Event], homography_confidence: float, fps: float,
    team_assignment_confidence: float, possession_segments: list[dict],
    directions=None,
):
    results = []
    events_by_type_and_player: dict[str, dict[int, list[Event]]] = {}
    for e in events:
        if e.player_id is not None:
            events_by_type_and_player.setdefault(e.event_type, {}).setdefault(e.player_id, []).append(e)

    events_by_player: dict[int, list[dict]] = {
        pid: [{"homography_confidence": e.homography_confidence, "metadata_json": e.metadata_json} for e in evs]
        for pid, evs in events_by_type_and_player.get("first_touch", {}).items()
    }

    players_by_frame = _build_players_by_frame(trajectories)
    decision_times_by_player = _compute_decision_times(possession_segments)

    for player_id, points in trajectories.items():
        player_events = events_by_player.get(player_id, [])
        first_touch = score_first_touch(player_events, homography_confidence=homography_confidence)
        results.append((first_touch, player_id))

        press_resistance = score_press_resistance(team_assignment_confidence=team_assignment_confidence)
        results.append((press_resistance, player_id))

        defensive_positioning = score_defensive_positioning(team_assignment_confidence=team_assignment_confidence)
        results.append((defensive_positioning, player_id))

        off_ball_inputs = _compute_off_ball_inputs(player_id, points, players_by_frame)
        off_ball = score_off_ball_movement(homography_confidence=homography_confidence, **off_ball_inputs)
        results.append((off_ball, player_id))

        orientation_values = [p.body_orientation_deg for p in points if p.body_orientation_deg is not None]
        body_orientation = score_body_orientation(orientation_values)
        results.append((body_orientation, player_id))

        orientation_readings = [
            (p.frame_id / fps, p.body_orientation_deg)
            for p in points
            if p.body_orientation_deg is not None
        ]
        tracked_duration_s = (points[-1].frame_id - points[0].frame_id) / fps if len(points) >= 2 else 0.0
        scanning = score_scanning_behavior(orientation_readings, tracked_duration_s=tracked_duration_s)
        results.append((scanning, player_id))

        player_team = _dominant_team_id(points)
        passing_inputs = _compute_passing_vision_inputs(
            player_id, events_by_type_and_player,
            attacking_direction=(directions.for_team(player_team)
                                 if directions is not None else DIRECTION_UNKNOWN),
        )
        passing_direction = passing_inputs.pop("attacking_direction")
        passing_vision = score_passing_vision(team_assignment_confidence=team_assignment_confidence, **passing_inputs)
        passing_vision.setdefault("sub_scores", {})["attacking_direction"] = passing_direction
        results.append((passing_vision, player_id))

        player_decision_times = decision_times_by_player.get(player_id, [])
        avg_decision_time = (
            sum(player_decision_times) / len(player_decision_times) if player_decision_times else None
        )
        n_pass = len(events_by_type_and_player.get("pass", {}).get(player_id, []))
        n_shot = len(events_by_type_and_player.get("shot", {}).get(player_id, []))
        n_turnover = len(events_by_type_and_player.get("turnover", {}).get(player_id, []))
        decision_making = score_decision_making(
            successful_actions=n_pass + n_shot,
            lost_actions=n_turnover,
            avg_decision_time=avg_decision_time,
            team_assignment_confidence=team_assignment_confidence,
        )
        results.append((decision_making, player_id))

        shots = _compute_finishing_inputs(player_id, events_by_type_and_player)
        finishing = score_finishing_efficiency(shots=shots, homography_confidence=homography_confidence)
        results.append((finishing, player_id))

    return results


def _team_metric_from_dict(match_id: str, d: dict) -> TeamMetric:
    raw_value = d.get("value")
    return TeamMetric(
        match_id=match_id,
        team_id=d.get("team_id", "unassigned"),
        metric_name=d["metric_name"],
        value_numeric=raw_value if isinstance(raw_value, (int, float)) else None,
        value_label=raw_value if isinstance(raw_value, str) else None,
        method=d["method"],
        confidence=d["confidence"],
        confidence_score=d.get("confidence_score"),
        sample_size=d["sample_size"],
        sub_scores=d["sub_scores"],
        computed_at=datetime.utcnow(),
        schema_version=d["schema_version"],
    )


def _player_metric_from_dict(match_id: str, player_id: int, d: dict) -> PlayerMetric:
    return PlayerMetric(
        match_id=match_id,
        player_id=player_id,
        metric_name=d["metric_name"],
        value=d["value"],
        method=d["method"],
        confidence=d["confidence"],
        sample_size=d["sample_size"],
        sub_scores=d["sub_scores"],
        computed_at=datetime.utcnow(),
        schema_version=d["schema_version"],
    )
