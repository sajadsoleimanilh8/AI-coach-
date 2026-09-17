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
    """Pixels per metre at one player's depth, from their own bounding box.

    None when the box is missing or too small to be a usable ruler -- see
    MIN_SCALE_BOX_HEIGHT_PX and the PLAYER_HEIGHT_M note in constants.py for
    what this estimate is and is not.
    """
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

    WHY THIS EXISTS. get_ball_possessor() below requires pitch metres, and on
    every clip this project's calibration model cannot solve there are none.
    That is not a small gap: possession is the input to first-touch, pass and
    turnover detection, so a match with no calibration produced ZERO events of
    any kind -- verified, 0 rows in the events table across 30 processed
    matches -- while the ball detector and the tracker had both worked.

    WHAT IT MEASURES. The same question, with a weaker instrument: which
    tracked player's feet are within `control_radius` METRES of the ball,
    where metres are converted from pixels using that player's OWN bounding
    box height as the local scale (see image_scale_px_per_m). Every input is
    a real measurement -- the tracker's box, the ball model's detection --
    and the only assumption is one stated constant for how tall a footballer
    is.

    WHAT IT REFUSES TO DO. It does not produce a position. The caller gets
    the possessor and the pixel positions that were compared, and pitch
    coordinates remain None all the way to the Event row. An event derived
    this way is labelled space="image" end to end, through the API, to the
    UI. This is a fallback that keeps possession-derived analysis alive on
    uncalibrated footage; it is not a substitute for calibration, and it must
    never be presented as one.

    Args:
        player_boxes: dicts with player_id, team_id, foot_x, foot_y (pixels,
            bottom-centre of the box) and box_height_px.
        ball_pixel: (x, y) ball centre in pixels, or None.
        control_radius: metres, same threshold the pitch-space path uses.

    Returns:
        The closest qualifying player's dict, augmented with the measured
        `distance_m_estimate` and the `px_per_m` used, or None.
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
            # No usable ruler for this player. Declining is the honest
            # outcome: a default scale would silently apply one player's
            # depth to another's.
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

    Args:
        player_positions: list of dicts with keys: player_id, team_id, pitch_x_m, pitch_y_m
        ball_position: (x, y) pitch meter position of ball
        homography_confidence: confidence score for this frame's homography.
            LEGACY fallback only -- see calibration_valid.
        control_radius: max distance in meters to declare control (default 1.5m)
        calibration_valid: the single source of truth for "are pitch
            coordinates usable in this frame"
            (frame_data.CalibrationState.valid). PREFERRED over
            homography_confidence: validity is the confidence gate AND a
            geometric cross-check against the field model's detected pitch
            region, so a homography can clear 0.6 and still be matched to
            the wrong part of the image. This function used to re-derive
            `confidence >= HOMOGRAPHY_CONFIDENCE_MIN` itself, which was one
            of four independent copies of that gate across the codebase
            (here, runner.py, ai/computer_vision/pipeline.py, and
            backend/api/tracking.py) -- they could and did disagree.

            None means "caller has no validity flag", and the old
            confidence comparison is used, so existing callers and tests
            keep working unchanged.

    Returns:
        dict of closest player within control_radius, or None if no player
        in radius or the calibration is unusable.
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

    Args:
        frame_possessions: list of dicts with keys: frame_id, timestamp, possessor (dict or None), homography_confidence, pitch_x_m, pitch_y_m

    Returns:
        list of Event dicts for first touches.
    """
    events = []
    prev_possessor_id = None

    for entry in frame_possessions:
        curr = entry.get("possessor")
        curr_id = curr.get("player_id") if curr else None

        if curr_id is not None and curr_id != prev_possessor_id:
            # First touch event!
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
