"""Stage 3-4: pixel-to-pitch trajectories and pose / body orientation.

Moved verbatim out of backend/pipeline/runner.py; run_pipeline() there is
the orchestrator that calls these in order.
"""


from __future__ import annotations

import logging
import os

from ai.computer_vision.frame_data import (
    CalibrationState,
    FrameData,
)

logger = logging.getLogger(__name__)

def _build_trajectories(frame_data: list[FrameData], frames, match_id: str,
                        fps: float,
                        manual_override: CalibrationState | None = None):
    """
    Projects tracked pixel positions into pitch metres using EACH FRAME'S OWN
    homography, and assembles the ball's pitch trajectory alongside.

    Automatic calibration is re-solved per frame against a panning broadcast
    camera, so one match-representative matrix applied to the whole clip
    would make a stationary player's pitch position drift with the camera --
    and every time-resolved team metric would then read camera motion as
    tactical movement. A MANUAL calibration is a single fixed matrix by
    construction, so that path keeps the single-H behaviour it has always
    had.
    """
    from ai.computer_vision.player_tracking.trajectory import enrich_with_pitch_coordinates

    # Players still go through enrich_with_pitch_coordinates(), which
    # anchors on det.foot_point() -- bottom-centre of the bbox, the point
    # that actually lies on the z=0 pitch plane. Verified, not assumed:
    # see trajectory.py:130 and TrackedDetection.foot_point().
    if manual_override is not None:
        # An invalid manual calibration contributes a confidence of 0.0, not
        # its raw solve confidence -- the validity gate is the one that saw
        # the geometry checks, and it must not be bypassed here.
        manual_confidence = manual_override.confidence if manual_override.valid else 0.0
        trajectories = enrich_with_pitch_coordinates(
            frames, match_id, manual_override.H, manual_confidence, fps)
    else:
        trajectories = enrich_with_pitch_coordinates(
            frames, match_id, None, 0.0, fps,
            calibration_by_frame={f.frame_id: f.calibration for f in frame_data},
        )

    # The ball's pitch track now comes from FrameData, where it was already
    # projected per frame under that frame's OWN calibration (see
    # _project_frame_geometry) -- rather than being re-derived here against
    # one match-level H. Interpolated points are carried through with their
    # provenance so a consumer can refuse to treat an inferred position as
    # evidence of a pass or shot.
    ball_trajectory: list[dict] = []
    for f in frame_data:
        if f.ball is None or not f.ball.usable:
            continue
        ball_trajectory.append({
            "frame_id": f.frame_id,
            "timestamp": f.timestamp,
            "pitch_x_m": f.ball.pitch_x_m,
            "pitch_y_m": f.ball.pitch_y_m,
            # Per-frame confidence, from that frame's own calibration
            # episode -- not one global constant repeated N times.
            "homography_confidence": f.calibration.confidence,
            "calibration_valid": f.calibration.valid,
            "ball_source": f.ball.source.value,
        })

    return trajectories, ball_trajectory


def _estimate_orientations(video_path: str, frames, stride: int, progress=None) -> dict:
    """
    Runs pose estimation on a stride-sampled subset of (player, frame)
    pairs and returns {(player_id, frame_number): OrientationResult}.

    Deliberately tolerant of pose_estimation failures on individual
    frames/players: a crop that's too small, a corrupt read, or a
    mediapipe error on one (player, frame) pair should not fail the whole
    pipeline run the way a missing trained checkpoint does (see
    PipelineAssetError) -- pose is an enrichment on top of tracking that
    already succeeded, not a required asset. Frames/players that fail
    simply have no entry in the returned dict, which
    attach_body_orientation() already treats as "not measured" (see its
    docstring) -- the honest default, not a crash.

    If mediapipe's legacy Solutions API isn't available at all on this
    machine (see pose.py's _load_legacy_pose_solution() docstring for the
    known-gotcha explanation), this degrades to returning an empty dict --
    every PlayerTracking row's body_orientation_deg stays None, and
    body_orientation_score/scanning_behavior_score correctly report
    low_sample rather than the run failing outright. A missing optional
    enrichment should not block the CV/tracking/scoring core that Day
    4-6 already delivers.
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

    # Sampled frames, not total frames: this is the count of frames that
    # actually cost pose inference, so the fraction it reports tracks the
    # time it is really spending rather than the video's length.
    n_sampled = len(range(0, len(frames), stride)) if frames else 0
    sampled_done = 0
    # One DB write per progress report, so report at most ~20 times rather
    # than once per sampled frame.
    report_every = max(1, n_sampled // 20)

    try:
        for frame_number, detections in enumerate(frames):
            # cap.read() always advances to the NEXT sequential frame --
            # it cannot be skipped forward without reading (or an
            # imprecise CAP_PROP_POS_FRAMES seek, which is unreliable on
            # many codecs). Read every frame to stay in sync with
            # `frame_number`, but only spend time on the expensive
            # crop+MediaPipe work when this frame is actually sampled.
            ok, raw_frame = cap.read()
            if not ok or raw_frame is None:
                break  # end of video, or a decode failure we can't recover from

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
                    continue  # degenerate bbox -- estimate_body_orientation()
                                # handles this too, but skip the crop entirely here

                crop = raw_frame[y1:y2, x1:x2]
                try:
                    reading = estimate_body_orientation(crop)
                    # One bad crop/model call must not take down the whole
                    # stage -- see this function's docstring.
                except Exception:  # noqa: BLE001 - one bad crop must not abort orientation for the clip
                    continue

                orientation_by_player_frame[(det.player_id, frame_number)] = reading
    finally:
        cap.release()

    return orientation_by_player_frame
