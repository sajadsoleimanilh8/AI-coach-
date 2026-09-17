"""
Phase 3 pipeline orchestration.

The real orchestration layer. backend/tasks.py only wraps `run_pipeline()` in
Celery bookkeeping (job status/progress).

Kept independent of Celery on purpose so it can be unit-tested with
synthetic data (see backend/pipeline/tests/test_runner.py) the same way
ai/computer_vision/player_tracking/tests/test_tracking_pipeline.py tests
tracker.py without a real video.

Honesty rules this module follows (do not weaken these when extending it):
  - If a required asset (trained model, calibration, video file) is
    missing, raise PipelineAssetError with a specific, actionable message.
    Never substitute a placeholder and continue as if it succeeded.
  - Every score/metric gate (homography_confidence, team_assignment_confidence)
    is passed through honestly from what was actually measured this run --
    never hardcoded to a "safe" value to make a module compute instead of
    reporting low_upstream_confidence.
  - team_assignment_confidence is now a REAL, per-run measurement -- see
    Stage 1.5 (_assign_teams_stage / ai/computer_vision/tactical_analysis/
    team_assignment.py's assign_teams()), which clusters players into two
    teams by jersey color and returns a match-level cluster-separation
    confidence. Formation/Press Resistance/Defensive Positioning/team-shape
    splits still correctly report "low_upstream_confidence" whenever that
    measured value is genuinely low (short clip, similar-colored kits,
    heavy occlusion) -- the gate itself is unchanged, only the value
    feeding it is no longer hardcoded.
"""

from __future__ import annotations

import logging
import os

from sqlalchemy.orm import Session

from ai.computer_vision.frame_data import (
    BallSource,
)
from ai.computer_vision.player_tracking.trajectory import attach_body_orientation
from ai.computer_vision.tactical_analysis.attacking_direction import (
    infer_attacking_directions,
)
from ai.computer_vision.tactical_analysis.role_inference import Role, infer_roles
from ai.computer_vision.tactical_analysis.team_assignment import assign_teams_with_stats
from backend.database.models import (
    Match,
    ProcessingJob,
)
from backend.pipeline.calibration import (  # noqa: F401 - re-exported for existing callers
    CALIBRATION_DIR,
    CALIBRATION_FAILURE_BACKOFF_AFTER,
    CALIBRATION_RETRY_STRIDE,
    FIELD_DETECT_STRIDE,
    GOALPOST_DETECT_STRIDE,
    _build_frame_data,
    _calibration_attempt_due,
    _estimate_fps,
    _load_manual_calibration,
    _persist_calibration_status,
    _project_frame_geometry,
    _representative_confidence,
    _video_frame_size,
)
from backend.pipeline.detection import (  # noqa: F401 - re-exported for existing callers
    REID_MERGE_ENABLED,
    _merge_reidentified_tracks,
    _pitch_coord_coverage,
    _player_checkpoint,
    _run_detection_and_tracking,
)
from backend.pipeline.device import resolve_device
from backend.pipeline.events import (  # noqa: F401 - re-exported for existing callers
    _detect_events,
    _enrich_first_touch_metadata,
    _image_space_possession,
)
from backend.pipeline.latency import PipelineTimer
from backend.pipeline.overlay_video import overlay_output_path, render_overlay_video
from backend.pipeline.persistence import (  # noqa: F401 - re-exported for existing callers
    FRAME_PERSIST_STRIDE,
    _persist_frames_and_tracking,
    _player_metric_from_dict,
    _team_metric_from_dict,
)
from backend.pipeline.player_scoring import (  # noqa: F401 - re-exported for existing callers
    _build_players_by_frame,
    _compute_decision_times,
    _compute_finishing_inputs,
    _compute_off_ball_inputs,
    _compute_passing_vision_inputs,
    _nearest_other_player_distance,
    _score_player_intelligence,
)
from backend.pipeline.results import PipelineAssetError, PipelineResult  # noqa: F401 - re-exported for existing callers
from backend.pipeline.team_scoring import (  # noqa: F401 - re-exported for existing callers
    _dominant_team_id,
    _mean_positions_per_player,
    _positions_by_frame,
    _score_team_intelligence,
    _split_by_team,
)
from backend.pipeline.trajectories import (  # noqa: F401 - re-exported for existing callers
    _build_trajectories,
    _estimate_orientations,
)

logger = logging.getLogger(__name__)

# Pose estimation is run on a stride-sampled subset of frames per
# player, not every frame -- MediaPipe per-crop is far more expensive than
# the homography math it feeds alongside, and orientation is a slowly-
# changing signal relative to position/speed, so sparse sampling loses
# little. See ai/computer_vision/pose_estimation/pose.py's docstring.
POSE_SAMPLE_STRIDE = int(os.getenv("POSE_SAMPLE_STRIDE", "10"))


def run_pipeline(db: Session, job: ProcessingJob, timer: PipelineTimer, progress_cb=None) -> PipelineResult:
    """
    Runs the full Phase 2 + Phase 3 pipeline for one uploaded video and
    writes real PlayerTracking / Event / PlayerMetric / TeamMetric rows.

    Args:
        db: active SQLAlchemy session (same one the caller will commit).
        job: ProcessingJob with `.video` and `.video.match` already loaded
            (see backend/api/main.py::upload_video, which creates the
            Match row and links it at upload time -- this function raises
            immediately if that link is missing, rather than guessing).
        timer: PipelineTimer -- every stage below is wrapped in
            `with timer.stage(...)` so the resulting PipelineLatencyReport
            is a real measurement of this exact run, not a hand-typed
            estimate (see backend/pipeline/latency.py's docstring for why
            that distinction matters here).
        progress_cb: optional `(percent: int, message: str) -> None`,
            called between stages so the caller (tasks.py) can update
            ProcessingJob.progress/message for the dashboard's progress bar.

    Returns:
        PipelineResult summarizing what was written, for logging/tests.

    Raises:
        PipelineAssetError: video/model/match-link missing.
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

    # ---- Device selection: ONCE per run, before any model loads --------
    # torch.cuda.is_available() alone is not enough -- a GPU whose compute
    # capability this torch build was not compiled for reports True and then
    # fails on the first real kernel. resolve_device() proves the GPU with a
    # real op and falls back to CPU if it cannot. See device.py's docstring.
    # The result is passed explicitly to every model below rather than left
    # to per-call auto-selection, so one bad probe means one clean CPU run.
    device_decision = resolve_device()
    logger.info("device for this run: %s (%s)", device_decision.device, device_decision.mode)

    # ---- Stage 1: detection + tracking -------------------------------
    report(15, f"{device_decision.message} Running player detection + ByteTrack tracking...")
    player_ckpt = _player_checkpoint()
    fps = _estimate_fps(video.storage_path)
    with timer.stage("detection_tracking", detail=f"model={player_ckpt} device={device_decision.device}"):
        frames = _run_detection_and_tracking(
            video.storage_path, player_ckpt, device=device_decision.device
        )

    report(18, "Re-associating tracks that ByteTrack dropped...")
    with timer.stage("reid_merge"):
        merge_report = _merge_reidentified_tracks(video.storage_path, frames, fps)

    # ---- Stage 1.5: jersey-color team assignment ------------------------
    # Must run before Stage 3 (_build_trajectories): enrich_with_pitch_coordinates()
    # copies det.team_id onto TrackingPoint verbatim, so team_id needs to
    # already be final by then. Mutates `frames` in place -- see
    # assign_teams()'s own docstring for the full algorithm and honesty
    # rules (per-track majority vote, goalkeepers always left unassigned).
    report(20, "Assigning players to teams via jersey-color clustering...")
    with timer.stage("team_assignment"):
        team_result = assign_teams_with_stats(video.storage_path, frames)
        team_assignment_confidence = team_result.confidence

    # ---- Stage 2: per-frame synchronisation (FrameData) ----------------
    # This is the join point for all five models. Ball, field and goalpost
    # detection plus automatic calibration all run here, over ONE pass of
    # the decoded video, and are assembled into one FrameData per frame
    # alongside the players tracked in Stage 1. Downstream stages read
    # FrameData, not loose per-detector lists.
    report(30, "Running ball/field/goalpost detection + automatic calibration...")
    frame_size = _video_frame_size(video.storage_path)
    manual_override = _load_manual_calibration(
        match.match_id, video.id, frame_size=frame_size)
    with timer.stage("frame_synchronisation"):
        frame_data, calib_stats = _build_frame_data(
            video.storage_path, frames, fps,
            manual_override=manual_override,
            device=device_decision.device,
        )

    # Representative scalar for the run, kept for the existing
    # PipelineResult/PlayerTracking contract. It is the MEDIAN confidence
    # over frames whose calibration was actually valid, not the best one
    # found -- a single lucky frame must not describe the whole run.
    homography_confidence = _representative_confidence(frame_data)

    # ---- Stage 2.5: persist calibration history -----------------------
    report(40, "Recording calibration status...")
    with timer.stage("calibration_history"):
        episodes = _persist_calibration_status(db, match, frame_data)

    # ---- Stage 3: pitch-coordinate trajectories -----------------------
    report(45, "Mapping tracked positions to pitch coordinates...")
    with timer.stage("pitch_trajectory"):
        trajectories, ball_trajectory = _build_trajectories(
            frame_data, frames, match.match_id, fps,
            manual_override=manual_override,
        )

    # ---- Stage 3.5: goalkeeper / referee role inference ----------------
    # HEURISTIC, not detection: the player model is single-class, so these
    # two roles are recovered from kit-colour outlier status plus pitch
    # position. Every result carries method=heuristic_proxy. Referees are
    # excluded from the trajectories fed to team-level scoring below, which
    # was the carried risk documented in docs/pipeline_architecture.md
    # ("until implemented, referees will be clustered as ordinary players
    # and pollute team assignment").
    with timer.stage("role_inference"):
        roles = infer_roles(
            team_result, trajectories,
            # Goalkeeper inference needs real pitch coordinates; referee
            # inference does not. Passing the honest flag means goalkeepers
            # come back as `unknown` rather than guessed when calibration
            # failed -- see role_inference.py's docstring.
            calibration_valid=any(f.calibration.valid for f in frame_data),
        )
        referee_ids = {pid for pid, r in roles.items() if r.role is Role.referee}
        if referee_ids:
            logger.info("excluding %d heuristically-identified referee track(s) "
                        "from team scoring: %s", len(referee_ids), sorted(referee_ids))
        scoring_trajectories = {pid: pts for pid, pts in trajectories.items()
                                if pid not in referee_ids}

    # ---- Stage 3.6: attacking direction per team -----------------------
    # REPLACES the hardcoded left_to_right global that shot detection and
    # forward-pass counting both inherited. Derived from where each team's
    # players actually are; returns `unknown` for both teams when there is
    # no valid calibration to measure that from, and the consumers then
    # decline to score direction-dependent values rather than guessing.
    with timer.stage("attacking_direction"):
        directions = infer_attacking_directions(scoring_trajectories)
        logger.info("attacking direction: %s (%s)", directions.by_team, directions.reason)

    # ---- Stage 4: pose / body orientation ----------------------------
    # By far the longest stage in the run (58.47s of 70.37s measured, 83.1%),
    # so it reports progress through the 50-55 band as it walks the sampled
    # frames. A progress bar that sits on 50% for a minute is
    # indistinguishable from a hung job.
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

    # ---- Stage 5: persist raw detections + tracking (sampled) ---------
    report(55, "Writing detection & tracking rows...")
    with timer.stage("db_write_tracking"):
        _persist_frames_and_tracking(
            db, match, frames, trajectories, fps, frame_data
        )

    # ---- Stage 6: event heuristics (pass/shot/turnover/first-touch) ---
    report(65, "Extracting pass, shot, turnover, and first-touch events...")
    with timer.stage("event_heuristics"):
        events, possession_segments = _detect_events(
            match, trajectories, ball_trajectory, fps,
            frame_data=frame_data, directions=directions)
        db.bulk_save_objects(events)
        db.commit()

    # ---- Stage 7: formation + team shape -------------------------------
    report(78, "Calculating formation and team shape metrics...")
    with timer.stage("team_scoring"):
        # Referee tracks removed -- see Stage 3.5.
        team_metrics = _score_team_intelligence(
            match, scoring_trajectories, team_assignment_confidence, directions=directions)
        for tm_dict in team_metrics:
            db.add(_team_metric_from_dict(match.match_id, tm_dict))
        db.commit()

    # ---- Stage 8: per-player scores ------------------------------------
    report(90, "Calculating per-player intelligence scores...")
    with timer.stage("player_scoring"):
        player_metrics = _score_player_intelligence(
            match, trajectories, events, homography_confidence, fps,
            team_assignment_confidence, possession_segments, directions=directions,
        )
        for pm_dict, player_id in player_metrics:
            db.add(_player_metric_from_dict(match.match_id, player_id, pm_dict))
        db.commit()

    # ---- Stage 9: burn the tracking overlay into a playable video ------
    # Presentation, not measurement. Deliberately LAST and deliberately
    # unable to fail the run: every metric above is already computed and
    # committed by this point, so an encoder problem costs a video file and
    # nothing else (see overlay_video.render_overlay_video's docstring).
    #
    # It is REPORTED rather than swallowed, though: the outcome (including
    # the codec used and the frame count the written file actually decodes
    # back to) is returned on PipelineResult, persisted by tasks.py into the
    # job's AnalysisResult, and served to the dashboard -- so "there is no
    # processed video" always comes with the reason there is none.
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
            calibration_by_frame={f.frame_id: f.calibration for f in frame_data},
            field_by_frame={f.frame_id: f.field_region for f in frame_data
                            if f.field_region is not None},
            goalposts_by_frame={f.frame_id: f.goalposts for f in frame_data
                                if f.goalposts},
            homography_by_frame={f.frame_id: f.calibration.H for f in frame_data
                                 if f.calibration.valid and f.calibration.H is not None},
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
        reid_merge=merge_report,
        pitch_coord_coverage=_pitch_coord_coverage(trajectories),
    )
