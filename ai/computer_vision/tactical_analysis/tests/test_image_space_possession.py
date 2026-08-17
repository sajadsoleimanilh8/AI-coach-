"""
The image-space possession fallback.
"""

from __future__ import annotations

import pytest

from ai.computer_vision.tactical_analysis.constants import (
    MIN_SCALE_BOX_HEIGHT_PX,
    PLAYER_HEIGHT_M,
)
from ai.computer_vision.tactical_analysis.possession import (
    get_ball_possessor_image_space,
    image_scale_px_per_m,
)
from ai.computer_vision.pass_detection.pass_heuristics import detect_passes


def _player(pid, foot_x, foot_y, box_height_px=90.0, team_id="team-home"):
    return {
        "player_id": pid,
        "team_id": team_id,
        "foot_x": foot_x,
        "foot_y": foot_y,
        "box_height_px": box_height_px,
    }



def test_scale_is_box_height_over_player_height():
    assert image_scale_px_per_m(90.0) == pytest.approx(90.0 / PLAYER_HEIGHT_M)


def test_a_box_too_small_to_be_a_ruler_returns_no_scale():
    assert image_scale_px_per_m(MIN_SCALE_BOX_HEIGHT_PX - 1) is None
    assert image_scale_px_per_m(None) is None



def test_the_nearest_player_inside_the_control_radius_has_possession():
    near = _player(7, foot_x=100.0, foot_y=200.0)
    far = _player(8, foot_x=400.0, foot_y=200.0)

    possessor = get_ball_possessor_image_space([near, far], (140.0, 200.0))

    assert possessor is not None
    assert possessor["player_id"] == 7
    assert possessor["distance_m_estimate"] == pytest.approx(0.8)


def test_nobody_inside_the_radius_means_nobody_has_possession():
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
    near_camera = _player(1, foot_x=100.0, foot_y=200.0, box_height_px=180.0)
    far_camera = _player(2, foot_x=100.0, foot_y=400.0, box_height_px=36.0)

    assert get_ball_possessor_image_space([near_camera], (150.0, 200.0)) is not None
    assert get_ball_possessor_image_space([far_camera], (150.0, 400.0)) is None



def test_the_fallback_never_produces_a_pitch_coordinate():
    possessor = get_ball_possessor_image_space(
        [_player(7, foot_x=100.0, foot_y=200.0)], (110.0, 200.0),
    )

    assert possessor is not None
    assert "pitch_x_m" not in possessor
    assert "pitch_y_m" not in possessor



def test_a_pass_can_be_detected_from_image_space_possession():
    """The distance GATE is evaluated on an estimate; the pass's position is
    still reported as unknown."""
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
    assert event["pitch_x_m"] is None and event["pitch_y_m"] is None


def test_a_short_image_space_transfer_is_not_a_pass():
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
