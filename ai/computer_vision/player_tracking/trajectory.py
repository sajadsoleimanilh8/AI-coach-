"""
Builds per-player trajectories from tracked detections and computes
speed / distance / acceleration in pitch meters.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai.computer_vision.pose_estimation.pose import OrientationResult

_AI_COMPUTER_VISION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _AI_COMPUTER_VISION_DIR not in sys.path:
    sys.path.insert(0, _AI_COMPUTER_VISION_DIR)

_PLAYER_TRACKING_DIR = os.path.dirname(os.path.abspath(__file__))
if _PLAYER_TRACKING_DIR not in sys.path:
    sys.path.insert(0, _PLAYER_TRACKING_DIR)

from tactical_analysis.constants import HOMOGRAPHY_CONFIDENCE_MIN
from tactical_analysis.homography import pixel_to_pitch

from tracker import TrackedDetection


@dataclass
class TrackingPoint:
    """One row of docs/database_schema.md's PlayerTracking table."""

    match_id: str
    player_id: int
    frame_id: int
    team_id: str | None
    pixel_x: float
    pixel_y: float
    pitch_x_m: float | None
    pitch_y_m: float | None
    homography_confidence: float | None
    speed: float | None
    distance: float | None
    acceleration: float | None
    body_orientation_deg: float | None = None
    body_orientation_confidence: float | None = None


def enrich_with_pitch_coordinates(
    frames: list[list["TrackedDetection"]],
    match_id: str,
    H,
    homography_confidence: float,
    fps: float,
) -> dict[int, list[TrackingPoint]]:
    """
    Convert tracked pixel detections into pitch-meter TrackingPoints, then
    compute speed/distance/acceleration per player along the way.
    """
    dt = 1.0 / fps
    usable = homography_confidence >= HOMOGRAPHY_CONFIDENCE_MIN

    trajectories: dict[int, list[TrackingPoint]] = {}
    last_point_by_player: dict[int, TrackingPoint] = {}
    last_speed_by_player: dict[int, float] = {}

    for frame_number, detections in enumerate(frames):
        for det in detections:
            if det.class_name not in ("player", "goalkeeper"):
                continue

            px, py = det.foot_point()

            if not usable:
                point = TrackingPoint(
                    match_id=match_id, player_id=det.player_id, frame_id=frame_number,
                    team_id=det.team_id, pixel_x=px, pixel_y=py,
                    pitch_x_m=None, pitch_y_m=None,
                    homography_confidence=homography_confidence,
                    speed=None, distance=None, acceleration=None,
                )
                trajectories.setdefault(det.player_id, []).append(point)
                continue

            pitch_x, pitch_y = pixel_to_pitch(px, py, H)

            prev = last_point_by_player.get(det.player_id)
            distance = speed = acceleration = None
            if prev is not None and prev.pitch_x_m is not None:
                distance = math.hypot(pitch_x - prev.pitch_x_m, pitch_y - prev.pitch_y_m)
                speed = distance / dt
                prev_speed = last_speed_by_player.get(det.player_id)
                if prev_speed is not None:
                    acceleration = (speed - prev_speed) / dt

            point = TrackingPoint(
                match_id=match_id, player_id=det.player_id, frame_id=frame_number,
                team_id=det.team_id, pixel_x=px, pixel_y=py,
                pitch_x_m=pitch_x, pitch_y_m=pitch_y,
                homography_confidence=homography_confidence,
                speed=speed, distance=distance, acceleration=acceleration,
            )
            trajectories.setdefault(det.player_id, []).append(point)
            last_point_by_player[det.player_id] = point
            if speed is not None:
                last_speed_by_player[det.player_id] = speed

    return trajectories


def attach_body_orientation(
    trajectories: dict[int, list[TrackingPoint]],
    orientation_by_player_frame: dict[tuple[int, int], "OrientationResult"],
) -> None:
    """
    Mutates `trajectories` in place, filling in body_orientation_deg/
    body_orientation_confidence on the TrackingPoints that have a pose
    reading.
    """
    for player_id, points in trajectories.items():
        for point in points:
            key = (player_id, point.frame_id)
            reading = orientation_by_player_frame.get(key)
            if reading is None:
                continue
            point.body_orientation_deg = reading.orientation_deg
            point.body_orientation_confidence = reading.confidence


def total_distance_covered(trajectory: list[TrackingPoint]) -> float:
    """Sum of per-frame distance deltas, ignoring frames with no distance
    (first frame of the trajectory, or frames dropped for low homography
    confidence)."""
    return sum(p.distance for p in trajectory if p.distance is not None)


def sprint_count(trajectory: list[TrackingPoint], sprint_threshold_ms: float = 7.0) -> int:
    """
    Number of frames where instantaneous speed exceeds sprint_threshold_ms
    (7.0 m/s ~= 25.2 km/h is a commonly used sprint-speed threshold in
    sports-science literature). Used as `sprint_load` input for the MVP
    Injury Risk fallback in docs/data_analysis.md §4.
    """
    return sum(1 for p in trajectory if p.speed is not None and p.speed > sprint_threshold_ms)
