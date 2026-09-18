"""
Raw tracking + heatmap API router.

Serves what backend/pipeline/runner.py::_persist_frames_and_tracking()
actually wrote to PlayerTracking/BallDetection -- nothing here is
aggregated the way tactical.py/player_intelligence.py are, this is the
per-frame data those endpoints don't expose. Two endpoints:

  - GET /api/matches/{match_id}/tracking: a frame-range window of raw
    pixel positions, for the video overlay. Windowed on purpose -- a real
    match is tens of thousands of frames, so returning "the whole match"
    in one payload isn't a viable API shape.
  - GET /api/matches/{match_id}/heatmap/{player_id}: server-aggregated
    pitch-coordinate density grid, gated on homography_confidence the same
    honesty pattern as TeamMetric/PlayerMetric (see the team-selection
    comment in backend/api/tactical.py and TabTeamIntelligence.jsx's PitchVisual) --
    never presents a heatmap as trustworthy when the upstream pitch
    coordinates aren't.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from ai.computer_vision.tactical_analysis.constants import (
    HOMOGRAPHY_CONFIDENCE_MIN,
    MIN_SAMPLE_EVENTS,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
)
from backend.api.schemas import (
    EventsResponse,
    HeatmapResponse,
    MatchListItem,
    MatchSummaryResponse,
    TrackingWindowResponse,
)
from backend.database.models import (
    AnalysisResult,
    BallDetection,
    CalibrationStatus,
    Event,
    Frame,
    Match,
    PlayerTracking,
    ProcessingJob,
    Video,
)
from backend.database.session import get_db
from backend.pipeline.overlay_video import find_overlay_output

router = APIRouter(prefix="/api/matches", tags=["tracking"])


def _valid_calibration_ranges(db: Session, match_id: str) -> list[tuple[int, int]] | None:
    """
    The frame ranges where this match's calibration was VALID, from
    calibration_status.

    This is the single implementation of the calibration validity gate.
    Validity is the `homography_confidence >= HOMOGRAPHY_CONFIDENCE_MIN` gate
    AND a geometric cross-check against the field model's detected pitch
    region, so a homography can clear 0.6 and still have been matched to the
    wrong part of the image. Re-deriving the gate from confidence alone
    disagrees with the pipeline on exactly those frames.

    Returns None when the match has NO calibration_status rows at all --
    i.e. it was processed before this table existed. Callers fall back to
    the legacy per-row confidence comparison in that case, so historical
    matches keep working instead of every heatmap silently emptying.
    """
    rows = (
        db.query(CalibrationStatus.frame_start, CalibrationStatus.frame_end,
                 CalibrationStatus.valid)
        .filter(CalibrationStatus.match_id == match_id)
        .all()
    )
    if not rows:
        return None
    return [(r.frame_start, r.frame_end) for r in rows if r.valid]

# A real match clip is tens of thousands of frames; PlayerTracking holds
# one row per player per frame (see runner.py's comment on why it isn't
# stride-sampled like Frame/PlayerDetection/BallDetection are). Shipping
# an unbounded frame range would mean a single request could pull the
# entire match's tracking table. DEFAULT_WINDOW_FRAMES is ~6s at a typical
# 25fps clip -- enough for the frontend's rolling playback cache (see
# TabMatchAnalysis.jsx) without either endpoint needing to guess an fps
# ahead of time.
DEFAULT_WINDOW_FRAMES = 150
MAX_WINDOW_FRAMES = 500

# 20x13 bins over the standard pitch (105m x 68m) = ~5.25m x ~5.23m per
# cell -- coarse enough that a few dozen tracked points per player produce
# a legible density map, without shipping every raw pitch coordinate to
# the browser (see this module's docstring).
HEATMAP_GRID_COLS = 20
HEATMAP_GRID_ROWS = 13


def _get_match_or_404(db: Session, match_id: str) -> Match:
    match = db.get(Match, match_id)
    if not match:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No match found for match_id={match_id}.",
        )
    return match


@router.get("", response_model=list[MatchListItem])
def list_matches(
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Every match in the database, newest first.

    Added because there was no way to DISCOVER a match id. The dashboard's
    match-scoped tabs (Player, Team, Calibration, Simulation) are gated on
    one, and the only two ways to obtain it were to upload a clip in this
    session or to already know a UUID by heart -- so a refresh, a deep link,
    or a new browser left four of the eight tabs permanently unusable with no
    route out of that state except retyping an id from a terminal.

    Read-only, and deliberately declared before /{match_id}: an empty path
    under this router's prefix cannot be confused with a match id.

    tracking_rows comes from one grouped COUNT rather than a per-match query
    in a loop -- the number of matches is unbounded and N+1 here would get
    slower exactly as the list got more useful.
    """
    matches = (
        db.query(Match)
        .order_by(Match.created_at.desc().nullslast(), Match.match_id)
        .limit(limit)
        .all()
    )
    if not matches:
        return []

    match_ids = [m.match_id for m in matches]

    row_counts = dict(
        db.query(PlayerTracking.match_id, func.count())
        .filter(PlayerTracking.match_id.in_(match_ids))
        .group_by(PlayerTracking.match_id)
        .all()
    )

    videos = {
        v.match_id: v
        for v in db.query(Video).filter(Video.match_id.in_(match_ids)).all()
    }

    video_ids = [v.id for v in videos.values()]
    jobs: dict[str, ProcessingJob] = {}
    if video_ids:
        # Newest last so the dict ends up holding the most recent job per
        # video -- the same "latest job wins" rule get_match_summary uses.
        for job in (
            db.query(ProcessingJob)
            .filter(ProcessingJob.video_id.in_(video_ids))
            .order_by(ProcessingJob.created_at.asc())
            .all()
        ):
            jobs[job.video_id] = job

    items: list[MatchListItem] = []
    for match in matches:
        video = videos.get(match.match_id)
        job = jobs.get(video.id) if video else None
        items.append(MatchListItem(
            match_id=match.match_id,
            home_team=match.home_team,
            away_team=match.away_team,
            created_at=match.created_at,
            video_filename=video.original_filename if video else None,
            # Same filesystem check as get_match_summary: a stored path is
            # not evidence the file is still there.
            video_file_exists=bool(
                video and video.storage_path and Path(video.storage_path).exists()
            ),
            job_id=job.id if job else None,
            job_status=job.status.value if job else None,
            tracking_rows=int(row_counts.get(match.match_id, 0)),
        ))
    return items


@router.get("/{match_id}", response_model=MatchSummaryResponse)
def get_match_summary(match_id: str, db: Session = Depends(get_db)):
    """Resolves a match to the ids its other views are keyed by.

    Declared before the longer /{match_id}/... routes only for readability --
    FastAPI matches on the full path shape, so this cannot shadow them.

    The most recent ProcessingJob is chosen when several exist: re-processing
    a video creates a new job, and the latest one is the only one whose
    latency report describes the data currently in the database.
    """
    match = _get_match_or_404(db, match_id)

    video = db.query(Video).filter(Video.match_id == match_id).first()
    job = None
    if video:
        job = (
            db.query(ProcessingJob)
            .filter(ProcessingJob.video_id == video.id)
            .order_by(ProcessingJob.created_at.desc())
            .first()
        )

    # Reported from the filesystem, not inferred from the row existing. A
    # Video row whose file has since been deleted would otherwise advertise
    # a playable clip and hand the UI a URL that 404s.
    file_exists = False
    processed_exists = False
    rendered = None
    if video and video.storage_path:
        file_exists = Path(video.storage_path).exists()
        rendered = find_overlay_output(video.storage_path, video.id)
        processed_exists = rendered is not None

    # When there is no render but a run finished, say WHY. The pipeline
    # records the render outcome on the job's AnalysisResult precisely so
    # this is answerable without reading worker logs.
    processed_error = None
    processed_codec = None
    if job:
        result = (
            db.query(AnalysisResult)
            .filter(AnalysisResult.job_id == job.id)
            .first()
        )
        overlay = (result.result_json or {}).get("overlay_render") if result else None
        if isinstance(overlay, dict):
            processed_codec = overlay.get("codec")
            if not processed_exists:
                processed_error = overlay.get("skipped_reason")

    return MatchSummaryResponse(
        match_id=match.match_id,
        home_team=match.home_team,
        away_team=match.away_team,
        duration=match.duration,
        video_id=video.id if video else None,
        video_filename=video.original_filename if video else None,
        video_file_exists=file_exists,
        processed_video_exists=processed_exists,
        processed_video_error=processed_error,
        processed_video_codec=processed_codec,
        **_frame_size_fields(db, match_id),
        job_id=job.id if job else None,
        job_status=job.status.value if job else None,
    )


def _frame_size_fields(db: Session, match_id: str) -> dict:
    """{'frame_width_px': .., 'frame_height_px': ..} for this match's SOURCE clip.

    Both None when the file is gone or unreadable -- consumers then fall back
    to the played video's own intrinsic size, which is correct for the raw
    clip and merely imprecise for the downscaled render, rather than being
    handed a guessed extent.
    """
    dimensions = _frame_dimensions(db, match_id)
    if dimensions is None:
        return {"frame_width_px": None, "frame_height_px": None}
    return {"frame_width_px": dimensions[0], "frame_height_px": dimensions[1]}


@router.get("/{match_id}/tracking", response_model=TrackingWindowResponse)
def get_tracking_window(
    match_id: str,
    start_frame: int = Query(default=0, ge=0),
    end_frame: int | None = Query(default=None, ge=0),
    db: Session = Depends(get_db),
):
    _get_match_or_404(db, match_id)

    if end_frame is None:
        end_frame = start_frame + DEFAULT_WINDOW_FRAMES
    if end_frame < start_frame:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="end_frame must be >= start_frame.",
        )
    if end_frame - start_frame > MAX_WINDOW_FRAMES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Requested window ({end_frame - start_frame} frames) exceeds the "
                   f"{MAX_WINDOW_FRAMES}-frame max per request. Page through with "
                   f"start_frame/end_frame instead of requesting the whole match.",
        )

    # fps is stored per-Frame (sampled rows only, see FRAME_PERSIST_STRIDE
    # in backend/pipeline/runner.py) -- any row for this match carries the
    # same fps the pipeline measured, so the first one is enough to convert
    # PlayerTracking.frame_id (a real frame *number*, not a Frame FK -- see
    # models.py's PlayerTracking.frame_id comment) into a timestamp.
    frame_row = db.query(Frame).filter(Frame.match_id == match_id).first()
    fps = frame_row.fps if frame_row else 25.0

    tracking_rows = (
        db.query(PlayerTracking)
        .filter(
            PlayerTracking.match_id == match_id,
            PlayerTracking.frame_id >= start_frame,
            PlayerTracking.frame_id <= end_frame,
        )
        .order_by(PlayerTracking.frame_id)
        .all()
    )

    # BallDetection only exists for the stride-sampled Frame rows (unlike
    # PlayerTracking, it isn't backfilled for every frame), so it's joined
    # through Frame.frame_number rather than assumed to line up 1:1 with
    # every frame_number in the requested window.
    ball_rows = (
        db.query(Frame.frame_number, BallDetection)
        .join(BallDetection, BallDetection.frame_id == Frame.frame_id)
        .filter(
            Frame.match_id == match_id,
            Frame.frame_number >= start_frame,
            Frame.frame_number <= end_frame,
        )
        .all()
    )
    ball_by_frame = {frame_number: ball for frame_number, ball in ball_rows}

    players_by_frame: dict[int, list] = {}
    for t in tracking_rows:
        players_by_frame.setdefault(t.frame_id, []).append({
            "player_id": t.player_id,
            "team_id": t.team_id,
            "pixel_x": t.pixel_x,
            "pixel_y": t.pixel_y,
            "pitch_x_m": t.pitch_x_m,
            "pitch_y_m": t.pitch_y_m,
        })

    frame_numbers = sorted(set(players_by_frame) | set(ball_by_frame))
    frames = []
    for fn in frame_numbers:
        ball = ball_by_frame.get(fn)
        frames.append({
            "frame_number": fn,
            "timestamp": fn / fps,
            "players": players_by_frame.get(fn, []),
            "ball": {"pixel_x": ball.ball_x, "pixel_y": ball.ball_y} if ball else None,
        })

    # Carry calibration validity alongside `pitch_x_m`/`pitch_y_m`, so a
    # consumer can tell "projected under a trusted homography" from "None
    # because calibration failed" without inspecting every coordinate. On real
    # broadcast footage every coordinate is currently None
    # (calibration_valid_fraction = 0.0, see docs/pipeline_architecture.md
    # 6.3).
    valid_ranges = _valid_calibration_ranges(db, match_id)
    positioned = sum(1 for t in tracking_rows if t.pitch_x_m is not None)

    # Highest tracked frame for this match. One aggregate, not a scan: the
    # player uses it to page its own window forward as playback advances,
    # which is what stopped the overlay from going dead six seconds in.
    max_frame = (
        db.query(func.max(PlayerTracking.frame_id))
        .filter(PlayerTracking.match_id == match_id)
        .scalar()
    )
    dimensions = _frame_dimensions(db, match_id)

    return {
        "match_id": match_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "fps": fps,
        "frames": frames,
        "frame_width_px": dimensions[0] if dimensions else None,
        "frame_height_px": dimensions[1] if dimensions else None,
        "max_frame": int(max_frame) if max_frame is not None else None,
        "calibration": {
            "has_status_rows": valid_ranges is not None,
            "valid_frame_ranges": valid_ranges or [],
            "valid_in_window": any(
                not (hi < start_frame or lo > end_frame) for lo, hi in (valid_ranges or [])
            ),
            "tracking_rows_in_window": len(tracking_rows),
            "rows_with_pitch_coordinates": positioned,
            "note": (
                "pitch_x_m/pitch_y_m are None wherever calibration was invalid; "
                "they are never imputed. Pixel positions remain valid regardless."
            ),
        },
    }


@router.get("/{match_id}/events", response_model=EventsResponse)
def get_match_events(
    match_id: str,
    event_type: str | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
    db: Session = Depends(get_db),
):
    """Every detected event for a match, chronological.

    Reads back the Event rows -- passes, shots, turnovers, first touches --
    that the pipeline writes on every job.

    Ordered by timestamp so the caller gets a timeline it can lay against the
    video without sorting. `space` reports how possession was resolved for
    this match (see EventItem.space) -- one answer for the match, because a
    run either had valid calibration or it did not.
    """
    _get_match_or_404(db, match_id)

    base = db.query(Event).filter(Event.match_id == match_id)

    # Counted over ALL of this match's events, before the type filter and
    # before the limit -- a caller filtering to shots still needs to know
    # how many passes exist, and a truncated list must not report its own
    # length as the total.
    by_type = dict(
        db.query(Event.event_type, func.count())
        .filter(Event.match_id == match_id)
        .group_by(Event.event_type)
        .all()
    )
    total = sum(by_type.values())

    if event_type:
        base = base.filter(Event.event_type == event_type)

    rows = base.order_by(Event.timestamp).limit(limit).all()

    items = []
    n_pitch = 0
    for row in rows:
        meta = row.metadata_json or {}
        # An event is in pitch space when it actually carries metres. This
        # is read off the row rather than assumed from the match's overall
        # calibration, so a run that solved calibration for part of a clip
        # reports each event honestly.
        space = "pitch" if row.pitch_x_m is not None else "image"
        if space == "pitch":
            n_pitch += 1
        items.append({
            "event_id": row.event_id,
            "event_type": row.event_type,
            "timestamp": row.timestamp,
            "player_id": row.player_id,
            "related_player_id": row.related_player_id,
            "team_id": row.team_id,
            "pitch_x_m": row.pitch_x_m,
            "pitch_y_m": row.pitch_y_m,
            "homography_confidence": row.homography_confidence,
            "space": space,
            "metadata": meta if isinstance(meta, dict) else {"value": meta},
        })

    if not items:
        space = "none"
    elif n_pitch == len(items):
        space = "pitch"
    elif n_pitch == 0:
        space = "image"
    else:
        # Genuinely mixed. Reported as image, the weaker of the two, so no
        # consumer treats the whole set as metric on the strength of a
        # calibrated minority.
        space = "image"

    return {
        "match_id": match_id,
        "total": total,
        "returned": len(items),
        "by_type": {str(k): int(v) for k, v in by_type.items()},
        "events": items,
        "space": space,
    }


def _frame_dimensions(db: Session, match_id: str) -> tuple[int, int] | None:
    """(width, height) of this match's video, read from the file itself.

    Not stored anywhere: Video.metadata_json is null on every row and Frame
    carries only timing. Reading the two capture properties does not decode a
    single frame, so this is a header read, not a pass over the video.
    """
    video = db.query(Video).filter(Video.match_id == match_id).first()
    if not video or not video.storage_path or not Path(video.storage_path).exists():
        return None
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(video.storage_path)
    try:
        if not cap.isOpened():
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return (width, height) if width > 0 and height > 0 else None


@router.get("/{match_id}/heatmap/{player_id}", response_model=HeatmapResponse)
def get_player_heatmap(
    match_id: str,
    player_id: int,
    space: str = Query(default="pitch", pattern="^(pitch|image)$"),
    db: Session = Depends(get_db),
):
    """Density grid for one player, in pitch metres or in image pixels.

    `space=image` was added because on all current footage the pitch grid is
    permanently empty: calibration never validates, so no PlayerTracking row
    carries pitch_x_m/pitch_y_m and every pitch heatmap honestly returns zero
    cells. The pixel positions those same rows DO carry are a real
    measurement, and binning them answers "where in frame was this player"
    without inventing a single coordinate. See HeatmapResponse.space for what
    that reading is and is not.
    """
    _get_match_or_404(db, match_id)

    rows = (
        db.query(PlayerTracking)
        .filter(PlayerTracking.match_id == match_id, PlayerTracking.player_id == player_id)
        .all()
    )
    sample_size = len(rows)

    if space == "image":
        return _image_space_heatmap(db, match_id, player_id, rows, sample_size)

    # Single source of truth: the calibration_status episodes written by
    # the pipeline. Falls back to the legacy per-row confidence comparison
    # only for matches processed before that table existed (ranges is None).
    ranges = _valid_calibration_ranges(db, match_id)
    if ranges is None:
        def _calibrated(r) -> bool:
            return (r.homography_confidence is not None
                    and r.homography_confidence >= HOMOGRAPHY_CONFIDENCE_MIN)
    else:
        def _calibrated(r) -> bool:
            return any(lo <= r.frame_id <= hi for lo, hi in ranges)

    usable = [
        r for r in rows
        if r.pitch_x_m is not None
        and r.pitch_y_m is not None
        and _calibrated(r)
    ]
    usable_sample_size = len(usable)

    # Same honesty gate as PlayerMetric/TeamMetric (backend/database/models.py's
    # MetricConfidence): "not computed" isn't representable by this enum, so
    # zero-row and zero-usable-row cases both honestly report
    # low_upstream_confidence with an empty grid -- never a fabricated
    # density map. sample_size/usable_sample_size tell the caller which of
    # the two actually happened.
    if usable_sample_size == 0:
        confidence = "low_upstream_confidence"
    elif usable_sample_size < MIN_SAMPLE_EVENTS:
        confidence = "low_sample"
    else:
        confidence = "normal"

    counts: dict[tuple[int, int], int] = {}
    if confidence == "normal" or confidence == "low_sample":
        for r in usable:
            gx = min(HEATMAP_GRID_COLS - 1, max(0, int((r.pitch_x_m / PITCH_LENGTH_M) * HEATMAP_GRID_COLS)))
            gy = min(HEATMAP_GRID_ROWS - 1, max(0, int((r.pitch_y_m / PITCH_WIDTH_M) * HEATMAP_GRID_ROWS)))
            counts[(gx, gy)] = counts.get((gx, gy), 0) + 1

    max_count = max(counts.values()) if counts else 0
    cells = [
        {
            "grid_x": gx,
            "grid_y": gy,
            "count": count,
            "density": (count / max_count) if max_count else 0.0,
        }
        for (gx, gy), count in sorted(counts.items())
    ]

    return {
        "match_id": match_id,
        "player_id": player_id,
        "grid_cols": HEATMAP_GRID_COLS,
        "grid_rows": HEATMAP_GRID_ROWS,
        "pitch_length_m": PITCH_LENGTH_M,
        "pitch_width_m": PITCH_WIDTH_M,
        "cells": cells,
        "confidence": confidence,
        "sample_size": sample_size,
        "usable_sample_size": usable_sample_size,
        "space": "pitch",
        "frame_width_px": None,
        "frame_height_px": None,
    }


def _image_space_heatmap(db: Session, match_id: str, player_id: int, rows, sample_size: int):
    """Bin this player's PIXEL positions over the video frame.

    No calibration gate, because none applies: a pixel position is not a
    projection of anything, it is where the tracker's box was. The honesty
    burden moves instead onto labelling -- the response says space="image"
    and reports the frame it was binned over, so no consumer can mistake
    these cells for pitch coordinates.

    The foot point (bottom-centre of the box) is not used here: rows store
    the box centre, and re-deriving a foot point would need a box height that
    PlayerTracking does not carry. Centre is what was measured, so centre is
    what is binned.
    """
    dimensions = _frame_dimensions(db, match_id)
    if dimensions is None:
        # Without the frame size there is no extent to bin over, and guessing
        # one from the observed pixel spread would make the grid's meaning
        # depend on where the player happened to run.
        return {
            "match_id": match_id, "player_id": player_id,
            "grid_cols": HEATMAP_GRID_COLS, "grid_rows": HEATMAP_GRID_ROWS,
            "pitch_length_m": PITCH_LENGTH_M, "pitch_width_m": PITCH_WIDTH_M,
            "cells": [], "confidence": "low_upstream_confidence",
            "sample_size": sample_size, "usable_sample_size": 0,
            "space": "image", "frame_width_px": None, "frame_height_px": None,
        }

    width, height = dimensions
    usable = [r for r in rows if r.pixel_x is not None and r.pixel_y is not None]
    usable_sample_size = len(usable)

    if usable_sample_size == 0:
        confidence = "low_upstream_confidence"
    elif usable_sample_size < MIN_SAMPLE_EVENTS:
        confidence = "low_sample"
    else:
        confidence = "normal"

    counts: dict[tuple[int, int], int] = {}
    for r in usable:
        gx = min(HEATMAP_GRID_COLS - 1, max(0, int((r.pixel_x / width) * HEATMAP_GRID_COLS)))
        gy = min(HEATMAP_GRID_ROWS - 1, max(0, int((r.pixel_y / height) * HEATMAP_GRID_ROWS)))
        counts[(gx, gy)] = counts.get((gx, gy), 0) + 1

    max_count = max(counts.values()) if counts else 0
    cells = [
        {"grid_x": gx, "grid_y": gy, "count": count,
         "density": (count / max_count) if max_count else 0.0}
        for (gx, gy), count in sorted(counts.items())
    ]

    return {
        "match_id": match_id, "player_id": player_id,
        "grid_cols": HEATMAP_GRID_COLS, "grid_rows": HEATMAP_GRID_ROWS,
        "pitch_length_m": PITCH_LENGTH_M, "pitch_width_m": PITCH_WIDTH_M,
        "cells": cells, "confidence": confidence,
        "sample_size": sample_size, "usable_sample_size": usable_sample_size,
        "space": "image", "frame_width_px": width, "frame_height_px": height,
    }
