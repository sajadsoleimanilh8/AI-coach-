"""
Raw tracking + heatmap API router.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from ai.computer_vision.tactical_analysis.constants import (
    HOMOGRAPHY_CONFIDENCE_MIN,
    MIN_SAMPLE_EVENTS,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
)
from pathlib import Path

from sqlalchemy import func

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

DEFAULT_WINDOW_FRAMES = 150
MAX_WINDOW_FRAMES = 500

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
    """Every match in the database, newest first."""
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
    """Resolves a match to the ids its other views are keyed by."""
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

    file_exists = False
    processed_exists = False
    rendered = None
    if video and video.storage_path:
        file_exists = Path(video.storage_path).exists()
        rendered = find_overlay_output(video.storage_path, video.id)
        processed_exists = rendered is not None

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
    """{'frame_width_px': .., 'frame_height_px': ..} for this match's SOURCE clip."""
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

    valid_ranges = _valid_calibration_ranges(db, match_id)
    positioned = sum(1 for t in tracking_rows if t.pitch_x_m is not None)

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
    """Every detected event for a match, chronological."""
    _get_match_or_404(db, match_id)

    base = db.query(Event).filter(Event.match_id == match_id)

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
    """(width, height) of this match's video, read from the file itself."""
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
    """Density grid for one player, in pitch metres or in image pixels."""
    _get_match_or_404(db, match_id)

    rows = (
        db.query(PlayerTracking)
        .filter(PlayerTracking.match_id == match_id, PlayerTracking.player_id == player_id)
        .all()
    )
    sample_size = len(rows)

    if space == "image":
        return _image_space_heatmap(db, match_id, player_id, rows, sample_size)

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
    """Bin this player's PIXEL positions over the video frame."""
    dimensions = _frame_dimensions(db, match_id)
    if dimensions is None:
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
