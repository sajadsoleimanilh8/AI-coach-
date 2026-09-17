"""Stage 1: player detection, ByteTrack tracking and re-identification merge.

Moved verbatim out of backend/pipeline/runner.py; run_pipeline() there is
the orchestrator that calls these in order.
"""


from __future__ import annotations

import logging
import os

from backend.pipeline.results import PipelineAssetError

logger = logging.getLogger(__name__)

#: Appearance re-association post-pass (player_tracking/reid_merge.py). On by
#: default: ByteTrack alone hands the same player a new id every time they are
#: occluded or missed for longer than its buffer, and every per-player metric
#: keys on player_id, so without this a single player is scored as several
#: half-players. Set SSC_REID_MERGE=0 to measure the pipeline without it.
REID_MERGE_ENABLED = os.getenv("SSC_REID_MERGE", "1").strip().lower() not in (
    "0", "false", "no", "off")


def _player_checkpoint() -> str:
    """Resolved path to the trained player checkpoint, via the registry.

    Raises PipelineAssetError (not the registry's own exception type) so
    tasks.py's existing handler surfaces it as job.error unchanged -- the
    registry's message is already written to say what is missing and which
    trainer produces it, so it is passed through verbatim.
    """
    from configs import registry

    try:
        return str(registry.checkpoint_path("player"))
    except Exception as exc:  # noqa: BLE001 -- re-raised as an asset error
        raise PipelineAssetError(str(exc)) from exc


def _run_detection_and_tracking(video_path: str, checkpoint: str,
                                device: str | None = None):
    """Delegates to ai.computer_vision.player_tracking.tracker.track_video().
    Not re-implemented here on purpose -- this module owns orchestration,
    not detection/tracking logic.

    Player detection is the ONE detector whose absence is fatal: with no
    tracked players there is nothing for any downstream stage to measure.
    Ball/field/goalpost/calibration all degrade instead (see
    _build_frame_data)."""
    if not os.path.exists(video_path):
        raise PipelineAssetError(f"Video file not found on disk: {video_path}")

    from ai.computer_vision.player_tracking.tracker import track_video

    try:
        return list(track_video(checkpoint, video_path, device=device))
    except FileNotFoundError as exc:
        raise PipelineAssetError(f"Model checkpoint not found: {checkpoint} ({exc})") from exc


def _pitch_coord_coverage(trajectories: dict) -> dict:
    """How much of the tracking actually carries a pitch position, and over
    how many distinct frames. With per-frame calibration the frame count is
    the number of frames that solved, rather than the single representative
    frame the old match-wide matrix effectively stood for."""
    total = 0
    with_coords = 0
    frames_with_coords: set[int] = set()
    frames_seen: set[int] = set()
    for points in trajectories.values():
        for p in points:
            total += 1
            frames_seen.add(p.frame_id)
            if p.pitch_x_m is not None:
                with_coords += 1
                frames_with_coords.add(p.frame_id)
    return {
        "tracking_points_total": total,
        "tracking_points_with_pitch_coords": with_coords,
        "point_coverage": (with_coords / total) if total else 0.0,
        "frames_with_tracked_players": len(frames_seen),
        "frames_contributing_pitch_coords": len(frames_with_coords),
        "frame_coverage": (len(frames_with_coords) / len(frames_seen)) if frames_seen else 0.0,
    }


def _merge_reidentified_tracks(video_path: str, frames, fps: float) -> dict | None:
    """
    Runs the appearance re-association post-pass over the MATERIALISED frame
    list, rewriting det.player_id in place.

    This runs before assign_teams_with_stats() on purpose. reid_merge does
    its own per-track team clustering for its "same team" gate, so it does
    not need team_id set; and merging first means team assignment sees whole
    tracks rather than fragments, and no merged chain can end up holding two
    different team_ids from having been labelled in two pieces.

    A failure here is not fatal -- unmerged ids are the status quo ante, not
    a wrong answer -- so it is logged and the run continues.
    """
    if not REID_MERGE_ENABLED:
        logger.info("reid_merge post-pass DISABLED via SSC_REID_MERGE; "
                    "player ids are raw ByteTrack output")
        return None
    if not frames:
        return None

    try:
        from ai.computer_vision.player_tracking.reid_merge import merge_reidentified_tracks

        result = merge_reidentified_tracks(video_path, frames, fps=fps)
    except Exception as exc:  # noqa: BLE001 -- degrade to unmerged ids
        logger.warning("reid_merge post-pass failed, continuing with raw "
                       "ByteTrack ids: %s", exc)
        return None

    report = result.as_dict()
    logger.info("reid_merge: %d tracks -> %d (%d merges in %d chains); "
                "rejected %d cross-team, %d across a hard cut; "
                "%d tracks had no resolvable team and were never merged; "
                "suppressed %d ghost tracks (%d detections)",
                result.tracks_before, result.tracks_after, result.merges,
                result.chains, result.rejected_cross_team,
                result.rejected_across_cut, result.unresolved_team_tracks,
                result.ghosts_suppressed, result.ghost_detections_dropped)
    return report
