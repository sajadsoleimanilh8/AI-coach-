"""
Possession and First Touch tracking helper.
Implementation Spec §2.
"""

from __future__ import annotations
import math

from ai.computer_vision.tactical_analysis.constants import (
    BALL_CONTROL_RADIUS_M,
    HOMOGRAPHY_CONFIDENCE_MIN,
    MIN_SCALE_BOX_HEIGHT_PX,
    PLAYER_HEIGHT_M,
)


def image_scale_px_per_m(box_height_px: float | None) -> float | None:
    """Pixels per metre at one player's depth, from their own bounding box."""
    if box_height_px is None or box_height_px < MIN_SCALE_BOX_HEIGHT_PX:
        return None
    return float(box_height_px) / PLAYER_HEIGHT_M


def get_ball_possessor_image_space(
    player_boxes: list[dict],
    ball_pixel: tuple[float, float] | None,
    control_radius: float = BALL_CONTROL_RADIUS_M,
) -> dict | None:
    """
    Possession resolved in PIXELS, for footage with no valid calibration.
    """
    if ball_pixel is None:
        return None

    bx, by = ball_pixel
    closest = None
    min_distance = float("inf")

    for player in player_boxes:
        fx, fy = player.get("foot_x"), player.get("foot_y")
        if fx is None or fy is None:
            continue
        px_per_m = image_scale_px_per_m(player.get("box_height_px"))
        if px_per_m is None:
            continue

        distance_m = math.hypot(fx - bx, fy - by) / px_per_m
        if distance_m <= control_radius and distance_m < min_distance:
            min_distance = distance_m
            closest = {**player,
                       "distance_m_estimate": distance_m,
                       "px_per_m": px_per_m}

    return closest


def get_ball_possessor(
    player_positions: list[dict],
    ball_position: tuple[float, float] | None,
    homography_confidence: float = 1.0,
    control_radius: float = BALL_CONTROL_RADIUS_M,
    calibration_valid: bool | None = None,
) -> dict | None:
    """
    Finds the tracked player whose foot point is within control_radius of ball pitch position.
    """
    usable = (calibration_valid if calibration_valid is not None
              else homography_confidence >= HOMOGRAPHY_CONFIDENCE_MIN)
    if not usable or ball_position is None:
        return None

    bx, by = ball_position
    closest_player = None
    min_dist = float("inf")

    for p in player_positions:
        px = p.get("pitch_x_m")
        py = p.get("pitch_y_m")
        if px is None or py is None:
            continue

        dist = math.hypot(px - bx, py - by)
        if dist <= control_radius and dist < min_dist:
            min_dist = dist
            closest_player = p

    return closest_player


def detect_first_touches(
    frame_possessions: list[dict],
) -> list[dict]:
    """
    Emits a 'first_touch' event on the frame a player gains possession after a frame with no controller or different controller.
    """
    events = []
    prev_possessor_id = None

    for entry in frame_possessions:
        curr = entry.get("possessor")
        curr_id = curr.get("player_id") if curr else None

        if curr_id is not None and curr_id != prev_possessor_id:
            events.append({
                "event_type": "first_touch",
                "player_id": curr_id,
                "team_id": curr.get("team_id"),
                "timestamp": entry.get("timestamp", 0.0),
                "pitch_x_m": entry.get("pitch_x_m"),
                "pitch_y_m": entry.get("pitch_y_m"),
                "homography_confidence": entry.get("homography_confidence", 1.0),
                "metadata_json": {"distance_to_ball": entry.get("distance_to_ball", 0.0)},
            })

        prev_possessor_id = curr_id

    return events
