"""
The image-space possession fallback.

WHY THIS EXISTS. Possession is the input to first-touch, pass and turnover
detection, and get_ball_possessor() requires pitch metres. On footage this
project's calibration model cannot solve there are none, so that path returned
None on every frame and every match produced ZERO events -- measured: 0 rows
in `events` across 30 processed matches, on runs where the tracker and the
ball model had both worked.

These tests pin what the fallback does and, just as importantly, what it
refuses to do: it never produces a pitch coordinate, and it declines rather
than guesses whenever its one measured input -- the player's own bounding-box
height, used as the local pixels-per-metre ruler -- is unusable.
"""

from __future__ import annotations

import pytest

from ai.computer_vision.pass_detection.pass_heuristics import detect_passes
from ai.computer_vision.tactical_analysis.constants import (
    MIN_SCALE_BOX_HEIGHT_PX,
    PLAYER_HEIGHT_M,
)
from ai.computer_vision.tactical_analysis.possession import (
    get_ball_possessor_image_space,
    image_scale_px_per_m,
)


def _player(pid, foot_x, foot_y, box_height_px=90.0, team_id="team-home"):
    return {
        "player_id": pid,
        "team_id": team_id,
        "foot_x": foot_x,
        "foot_y": foot_y,
        "box_height_px": box_height_px,
    }


# --- the ruler -------------------------------------------------------------

def test_scale_is_box_height_over_player_height():
    # 90px tall box on a 1.80m player = 50 px per metre.
    assert image_scale_px_per_m(90.0) == pytest.approx(90.0 / PLAYER_HEIGHT_M)


def test_a_box_too_small_to_be_a_ruler_returns_no_scale():
    # Declining is the point: a default scale would silently apply one
    # player's depth to another's.
    assert image_scale_px_per_m(MIN_SCALE_BOX_HEIGHT_PX - 1) is None
    assert image_scale_px_per_m(None) is None


# --- possession ------------------------------------------------------------

def test_the_nearest_player_inside_the_control_radius_has_possession():
    # 90px box -> 50 px/m, so 1.5m of control radius is 75px.
    near = _player(7, foot_x=100.0, foot_y=200.0)
    far = _player(8, foot_x=400.0, foot_y=200.0)

    possessor = get_ball_possessor_image_space([near, far], (140.0, 200.0))

    assert possessor is not None
    assert possessor["player_id"] == 7
    # 40px / 50 px-per-m = 0.8m
    assert possessor["distance_m_estimate"] == pytest.approx(0.8)


def test_nobody_inside_the_radius_means_nobody_has_possession():
    # 300px away at 50 px/m is 6m -- well outside the 1.5m control radius.
    lone = _player(7, foot_x=100.0, foot_y=200.0)

    assert get_ball_possessor_image_space([lone], (400.0, 200.0)) is None


def test_no_ball_means_no_possessor():
    assert get_ball_possessor_image_space([_player(7, 100.0, 200.0)], None) is None


def test_a_player_whose_box_is_unusable_is_skipped_not_estimated():
    """A tiny box is usually a partially-occluded or badly-cropped player.
    Estimating from it would inflate or deflate the distance with no way to
    tell -- the fallback declines instead."""
    unusable = _player(7, foot_x=100.0, foot_y=200.0,
                       box_height_px=MIN_SCALE_BOX_HEIGHT_PX - 1)

    assert get_ball_possessor_image_space([unusable], (101.0, 200.0)) is None


def test_perspective_is_handled_per_player_not_globally():
    """A player far from camera has a shorter box, so the SAME pixel distance
    is a LARGER real distance for them. A single global scale cannot express
    that; this is the reason the ruler is per-player."""
    near_camera = _player(1, foot_x=100.0, foot_y=200.0, box_height_px=180.0)  # 100 px/m
    far_camera = _player(2, foot_x=100.0, foot_y=400.0, box_height_px=36.0)    # 20 px/m

    # 50px from each. For the near player that is 0.5m (inside the radius);
    # for the far player it is 2.5m (outside it).
    assert get_ball_possessor_image_space([near_camera], (150.0, 200.0)) is not None
    assert get_ball_possessor_image_space([far_camera], (150.0, 400.0)) is None


# --- what it must never do -------------------------------------------------

def test_the_fallback_never_produces_a_pitch_coordinate():
    possessor = get_ball_possessor_image_space(
        [_player(7, foot_x=100.0, foot_y=200.0)], (110.0, 200.0),
    )

    assert possessor is not None
    # The whole honesty contract in one assertion: this path measures a
    # distance, it does not locate anything on a pitch.
    assert "pitch_x_m" not in possessor
    assert "pitch_y_m" not in possessor


# --- passes derived from it ------------------------------------------------

def test_a_pass_can_be_detected_from_image_space_possession():
    """The distance GATE is evaluated on an estimate; the pass's position is
    still reported as unknown."""
    # 50 px/m on both ends. 250px apart -> 5m, over the 3m minimum.
    sequence = [
        {"player_id": 1, "team_id": "team-home", "pitch_x_m": None, "pitch_y_m": None,
         "pixel_x": 100.0, "pixel_y": 200.0, "px_per_m": 50.0, "timestamp": 1.0},
        {"player_id": 2, "team_id": "team-home", "pitch_x_m": None, "pitch_y_m": None,
         "pixel_x": 350.0, "pixel_y": 200.0, "px_per_m": 50.0, "timestamp": 1.6},
    ]

    passes = detect_passes(sequence)

    assert len(passes) == 1
    event = passes[0]
    assert event["player_id"] == 1 and event["related_player_id"] == 2
    assert event["metadata_json"]["space"] == "image"
    assert event["metadata_json"]["pass_distance_m"] == pytest.approx(5.0)
    assert event["metadata_json"]["pass_distance_m_is_estimate"] is True
    # Never back-filled, no matter how confidently the distance was measured.
    assert event["pitch_x_m"] is None and event["pitch_y_m"] is None


def test_a_short_image_space_transfer_is_not_a_pass():
    # 50px at 50 px/m = 1m, under the 3m minimum.
    sequence = [
        {"player_id": 1, "team_id": "team-home", "pitch_x_m": None, "pitch_y_m": None,
         "pixel_x": 100.0, "pixel_y": 200.0, "px_per_m": 50.0, "timestamp": 1.0},
        {"player_id": 2, "team_id": "team-home", "pitch_x_m": None, "pitch_y_m": None,
         "pixel_x": 150.0, "pixel_y": 200.0, "px_per_m": 50.0, "timestamp": 1.2},
    ]

    assert detect_passes(sequence) == []


def test_image_space_pass_detection_declines_without_a_scale():
    """No ruler, no distance, no pass. It does not fall back to comparing raw
    pixels against a metre threshold."""
    sequence = [
        {"player_id": 1, "team_id": "team-home", "pitch_x_m": None, "pitch_y_m": None,
         "pixel_x": 100.0, "pixel_y": 200.0, "timestamp": 1.0},
        {"player_id": 2, "team_id": "team-home", "pitch_x_m": None, "pitch_y_m": None,
         "pixel_x": 900.0, "pixel_y": 200.0, "timestamp": 1.6},
    ]

    assert detect_passes(sequence) == []


def test_pitch_space_passes_are_unchanged_by_the_fallback():
    """The calibrated path must behave exactly as before -- the fallback is
    only consulted when pitch coordinates are absent."""
    sequence = [
        {"player_id": 1, "team_id": "team-home", "pitch_x_m": 10.0, "pitch_y_m": 30.0,
         "timestamp": 1.0, "homography_confidence": 0.9},
        {"player_id": 2, "team_id": "team-home", "pitch_x_m": 20.0, "pitch_y_m": 30.0,
         "timestamp": 1.6, "homography_confidence": 0.9},
    ]

    passes = detect_passes(sequence)

    assert len(passes) == 1
    assert passes[0]["metadata_json"]["space"] == "pitch"
    assert "pass_distance_m_is_estimate" not in passes[0]["metadata_json"]
    assert passes[0]["pitch_x_m"] == 20.0
