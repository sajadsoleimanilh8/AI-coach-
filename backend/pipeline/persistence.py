"""Writing frames, detections, tracking points and metric rows.

Moved verbatim out of backend/pipeline/runner.py; run_pipeline() there is
the orchestrator that calls these in order.
"""


from __future__ import annotations

import logging
import os
from datetime import datetime

from sqlalchemy.orm import Session

from ai.computer_vision.frame_data import (
    BallSource,
    FrameData,
)
from backend.database.models import (
    BallDetection,
    Frame,
    Match,
    PlayerDetection,
    PlayerMetric,
    PlayerTracking,
    TeamMetric,
)

logger = logging.getLogger(__name__)

# Every Nth frame gets a persisted Frame/PlayerDetection/BallDetection row.
# PlayerTracking (used for scoring) is still built from every frame -- this
# only thins out the raw-detection audit trail, which is far higher volume
# and mostly useful for debugging/replay, not scoring itself.
FRAME_PERSIST_STRIDE = int(os.getenv("FRAME_PERSIST_STRIDE", "5"))


def _persist_frames_and_tracking(db: Session, match: Match, frames, trajectories,
                                 fps: float, frame_data: list[FrameData] | None = None) -> int:
    n_persisted = 0

    # The ball comes from FrameData, where the dedicated small-object ball
    # model put it -- not from the tracker. The player model is single-class
    # (`nc: 1, names: [player]`), so no tracker detection is ever a ball.
    ball_by_frame = {}
    if frame_data is not None:
        ball_by_frame = {
            f.frame_id: f.ball for f in frame_data
            # Only REAL detections are persisted as BallDetection rows.
            # Interpolated positions are inferences, and this table is the
            # raw-detection audit trail -- writing an invented position here
            # with a fabricated confidence would make it indistinguishable
            # from a measurement. They remain available, marked as
            # interpolated, on FrameData/ball_trajectory.
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
        db.flush()  # need frame_row.frame_id for the FK below
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
                continue  # cannot occur with the single-class player model
            db.add(PlayerDetection(
                frame_id=frame_row.frame_id,
                player_id=det.player_id,
                team_id=det.team_id,
                team_assignment_confidence=det.team_assignment_confidence,
                x=det.x, y=det.y, width=det.width, height=det.height,
                confidence=det.confidence,
            ))

    # PlayerTracking: every frame, every player -- this is what scoring
    # and the heatmap/dashboard consume, so it isn't sampled down like the
    # raw detection audit trail above.
    tracking_rows = []
    for _player_id, points in trajectories.items():
        for p in points:
            tracking_rows.append(PlayerTracking(
                match_id=match.match_id,
                player_id=p.player_id,
                frame_id=p.frame_id,  # real video frame number, not an FK (see models.py PlayerTracking.frame_id)
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


def _team_metric_from_dict(match_id: str, d: dict) -> TeamMetric:
    raw_value = d.get("value")
    return TeamMetric(
        match_id=match_id,
        team_id=d.get("team_id", "unassigned"),
        metric_name=d["metric_name"],
        # See TeamMetric.value_numeric / value_label in models.py: the
        # Standard Output Contract's single `value` field is categorical
        # for formation ("4-3-3") and numeric for everything else. Route
        # to the correct column instead of losing the formation label the
        # way a naive Float-only write would have.
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
