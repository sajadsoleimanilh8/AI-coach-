"""
Unit tests for Event Heuristics (Pass, Shot, First Touch, Turnover).
"""



from ai.computer_vision.pass_detection.pass_heuristics import detect_passes
from ai.computer_vision.shot_detection.shot_heuristics import detect_shots
from ai.computer_vision.tactical_analysis.possession import get_ball_possessor


def test_get_ball_possessor_and_first_touch():
    players = [
        {"player_id": 10, "team_id": "team-A", "pitch_x_m": 20.0, "pitch_y_m": 30.0},
        {"player_id": 7, "team_id": "team-B", "pitch_x_m": 50.0, "pitch_y_m": 30.0},
    ]

    # Ball near player 10
    ball = (20.5, 30.2)
    possessor = get_ball_possessor(players, ball)
    assert possessor is not None
    assert possessor["player_id"] == 10

    # Ball far from everyone
    ball_far = (80.0, 10.0)
    assert get_ball_possessor(players, ball_far) is None


def test_pass_detection_synthetic():
    possessions = [
        {"player_id": 10, "team_id": "team-A", "pitch_x_m": 20.0, "pitch_y_m": 30.0, "timestamp": 1.0},
        {"player_id": 8, "team_id": "team-A", "pitch_x_m": 30.0, "pitch_y_m": 32.0, "timestamp": 3.0},
    ]

    passes = detect_passes(possessions)
    assert len(passes) == 1
    assert passes[0]["event_type"] == "pass"
    assert passes[0]["player_id"] == 10
    assert passes[0]["related_player_id"] == 8


def test_shot_detection_synthetic():
    # Ball traveling fast towards goal at x=105.0, y=34.0
    ball_positions = [
        {"frame_id": 1, "timestamp": 0.0, "pitch_x_m": 80.0, "pitch_y_m": 34.0, "player_id": 9, "team_id": "team-A"},
        {"frame_id": 2, "timestamp": 0.04, "pitch_x_m": 82.0, "pitch_y_m": 34.0, "player_id": 9, "team_id": "team-A"},
    ]  # dx = 2m in 0.04s -> 50 m/s

    # detect_shots() does not assume left_to_right.
    # The direction must be supplied, because it is only knowable per team
    # and it flips at half time -- see
    # ai/computer_vision/tactical_analysis/attacking_direction.py.
    shots = detect_shots(ball_positions, fps=25.0,
                         direction_by_team={"team-A": "left_to_right"})
    assert len(shots) == 1
    assert shots[0]["event_type"] == "shot"
    assert shots[0]["player_id"] == 9
    # Nominal geometry here (no goalpost detections supplied) -- and the
    # event says so rather than presenting it as measured.
    assert shots[0]["metadata_json"]["goal_geometry_source"] == "nominal"


def test_shot_detection_without_direction_is_uncertain_not_a_shot():
    """The old code defaulted to left_to_right, so this same ball was
    reported as a shot on the right-hand goal regardless of which way the
    team was actually playing. Half of all matches were scored backwards."""
    ball_positions = [
        {"frame_id": 1, "timestamp": 0.0, "pitch_x_m": 80.0, "pitch_y_m": 34.0, "player_id": 9, "team_id": "team-A"},
        {"frame_id": 2, "timestamp": 0.04, "pitch_x_m": 82.0, "pitch_y_m": 34.0, "player_id": 9, "team_id": "team-A"},
    ]
    events = detect_shots(ball_positions, fps=25.0)
    assert [e["event_type"] for e in events] == ["uncertain"]
    assert events[0]["metadata_json"]["classification_reason"] == "attacking_direction_unknown"


def test_shot_detection_uses_detected_goal_mouth_over_nominal():
    """A measured mouth displaced in y changes the verdict for a ball that
    would have been 'on target' against the nominal centred goal."""
    from ai.computer_vision.shot_detection.shot_heuristics import GoalMouth

    ball_positions = [
        {"frame_id": 1, "timestamp": 0.0, "pitch_x_m": 95.0, "pitch_y_m": 34.0, "player_id": 9, "team_id": "team-A"},
        {"frame_id": 2, "timestamp": 0.04, "pitch_x_m": 97.0, "pitch_y_m": 34.0, "player_id": 9, "team_id": "team-A"},
    ]
    measured = {"right": GoalMouth(side="right", x_m=105.0, y_min_m=10.0, y_max_m=17.32,
                                   source="detected", confidence=0.9, n_observations=40)}
    events = detect_shots(ball_positions, fps=25.0,
                          direction_by_team={"team-A": "left_to_right"},
                          goal_mouths=measured)
    assert events[0]["event_type"] != "shot"          # 34 m is nowhere near a mouth at 10-17 m
    assert events[0]["metadata_json"]["goal_geometry_source"] == "detected"
    assert events[0]["metadata_json"]["goal_mouth_observations"] == 40


def test_wide_delivery_outside_the_posts_is_a_cross_not_a_shot():
    ball_positions = [
        {"frame_id": 1, "timestamp": 0.0, "pitch_x_m": 95.0, "pitch_y_m": 4.0, "player_id": 7, "team_id": "team-A"},
        {"frame_id": 2, "timestamp": 0.04, "pitch_x_m": 96.5, "pitch_y_m": 6.0, "player_id": 7, "team_id": "team-A"},
    ]
    events = detect_shots(ball_positions, fps=25.0,
                          direction_by_team={"team-A": "left_to_right"})
    assert events[0]["event_type"] == "cross"


def test_fast_ball_out_of_own_box_is_a_clearance_not_a_shot():
    """team-A attacks right_to_left, so its OWN goal is the right one
    (x=105). A ball struck hard from inside that box is travelling towards
    the goal team-A attacks, and on a 1-D x axis that is indistinguishable
    from a shot by direction alone -- the origin is what separates them."""
    ball_positions = [
        {"frame_id": 1, "timestamp": 0.0, "pitch_x_m": 98.0, "pitch_y_m": 34.0, "player_id": 4, "team_id": "team-A"},
        {"frame_id": 2, "timestamp": 0.04, "pitch_x_m": 95.5, "pitch_y_m": 34.0, "player_id": 4, "team_id": "team-A"},
    ]
    events = detect_shots(ball_positions, fps=25.0,
                          direction_by_team={"team-A": "right_to_left"})
    assert events[0]["event_type"] == "clearance"
    assert events[0]["metadata_json"]["classification_reason"] == \
        "fast_ball_struck_from_inside_own_penalty_area"


if __name__ == "__main__":
    test_get_ball_possessor_and_first_touch()
    test_pass_detection_synthetic()
    test_shot_detection_synthetic()
    print("ALL EVENT HEURISTICS TESTS PASSED!")
