"""
Unit tests for Player Intelligence scoring modules.
"""


from ai.player_intelligence.decision_making_score.score import score_decision_making
from ai.player_intelligence.defensive_positioning.score import score_defensive_positioning
from ai.player_intelligence.finishing_efficiency_score.score import score_finishing_efficiency
from ai.player_intelligence.off_ball_movement.score import score_off_ball_movement
from ai.player_intelligence.passing_vision_score.score import score_passing_vision
from ai.player_intelligence.press_resistance_score.score import score_press_resistance


def test_press_resistance_never_pressured():
    result = score_press_resistance(
        possessions_under_pressure=0, pressure_events=0, team_assignment_confidence=0.9
    )
    assert result["value"] is None
    assert result["confidence"] == "low_sample"


def test_press_resistance_low_team_assignment_confidence():
    result = score_press_resistance(
        possessions_under_pressure=10, pressure_events=10, team_assignment_confidence=0.0
    )
    assert result["value"] is None
    assert result["confidence"] == "low_upstream_confidence"


def test_off_ball_movement_space_creation():
    result = score_off_ball_movement(
        avg_distance_gained_from_marker=5.0,  # Max gain (5.0m)
        off_ball_frames_in_attacking_third=100,
        total_off_ball_frames_while_team_in_possession=200,
        net_displacement_m=50.0,
        total_path_length_m=50.0,
        sample_size_phases=10,
    )
    assert result["value"] is not None
    assert result["sub_scores"]["space_creation"] == 100.0


def test_defensive_positioning_drifted_player():
    # Player drifted 15m out of line (max acceptable = 8m)
    result_drifted = score_defensive_positioning(
        avg_deviation_from_defensive_line_m=15.0,
        nearest_teammate_distance_m=10.0,
        closing_speed_toward_threat_ms=2.0,
        sample_size_defensive_phases=10,
        team_assignment_confidence=0.9,
    )
    assert result_drifted["sub_scores"]["line_discipline"] == 0.0

    # Player in line (0m deviation)
    result_in_line = score_defensive_positioning(
        avg_deviation_from_defensive_line_m=0.0,
        nearest_teammate_distance_m=10.0,
        closing_speed_toward_threat_ms=2.0,
        sample_size_defensive_phases=10,
        team_assignment_confidence=0.9,
    )
    assert result_in_line["sub_scores"]["line_discipline"] == 100.0


def test_passing_vision_low_team_assignment_confidence():
    result = score_passing_vision(
        completed_passes=10, turnovers_lost=0, team_assignment_confidence=0.0
    )
    assert result["value"] is None
    assert result["confidence"] == "low_upstream_confidence"


def test_passing_vision_perfect_completion():
    result = score_passing_vision(
        completed_passes=10, turnovers_lost=0,
        pass_sector_counts=[5, 5, 0, 0, 0, 0, 0, 0], forward_passes=10,
        team_assignment_confidence=0.9,
    )
    assert result["value"] is not None
    assert result["confidence"] == "normal"
    assert result["sub_scores"]["completion_rate"] == 100.0
    assert result["sub_scores"]["forward_pass_ratio"] == 100.0


def test_passing_vision_no_sector_data_drops_diversity_subscore():
    result = score_passing_vision(
        completed_passes=6, turnovers_lost=0, pass_sector_counts=None,
        forward_passes=3, team_assignment_confidence=0.9,
    )
    assert result["sub_scores"]["angle_diversity"] is None
    assert result["value"] is not None  # redistributed from the other two components


def test_decision_making_low_team_assignment_confidence():
    result = score_decision_making(
        successful_actions=10, lost_actions=0, team_assignment_confidence=0.0
    )
    assert result["value"] is None
    assert result["confidence"] == "low_upstream_confidence"


def test_decision_making_fast_reliable_player():
    result = score_decision_making(
        successful_actions=9, lost_actions=1, avg_decision_time=0.3, team_assignment_confidence=0.9,
    )
    assert result["value"] is not None
    assert result["confidence"] == "normal"
    assert result["sub_scores"]["decision_speed"] == 100.0  # at DECISION_TIME_MIN_S, clips to max


def test_decision_making_no_decision_time_redistributes():
    result = score_decision_making(
        successful_actions=8, lost_actions=2, avg_decision_time=None, team_assignment_confidence=0.9,
    )
    assert result["sub_scores"]["decision_speed"] is None
    assert result["value"] is not None  # retention alone, weight redistributed


def test_finishing_efficiency_low_homography_confidence():
    result = score_finishing_efficiency(
        shots=[{"pitch_x_m": 100.0, "pitch_y_m": 34.0, "projected_y_at_goal": 34.0}],
        homography_confidence=0.0,
    )
    assert result["value"] is None
    assert result["confidence"] == "low_upstream_confidence"


def test_finishing_efficiency_empty_shots_is_low_sample():
    result = score_finishing_efficiency(shots=[], homography_confidence=0.9)
    assert result["value"] is None
    assert result["confidence"] == "low_sample"
    assert result["sample_size"] == 0


def test_finishing_efficiency_dead_center_close_shot():
    shots = [{"pitch_x_m": 104.5, "pitch_y_m": 34.0, "projected_y_at_goal": 34.0}] * 6
    result = score_finishing_efficiency(shots=shots, homography_confidence=0.9)
    assert result["confidence"] == "normal"
    assert result["sub_scores"]["placement"] == 100.0  # dead center of the goal mouth


if __name__ == "__main__":
    test_press_resistance_never_pressured()
    test_press_resistance_low_team_assignment_confidence()
    test_off_ball_movement_space_creation()
    test_defensive_positioning_drifted_player()
    test_passing_vision_low_team_assignment_confidence()
    test_passing_vision_perfect_completion()
    test_passing_vision_no_sector_data_drops_diversity_subscore()
    test_decision_making_low_team_assignment_confidence()
    test_decision_making_fast_reliable_player()
    test_decision_making_no_decision_time_redistributes()
    test_finishing_efficiency_low_homography_confidence()
    test_finishing_efficiency_empty_shots_is_low_sample()
    test_finishing_efficiency_dead_center_close_shot()
    print("ALL PLAYER INTELLIGENCE SCORE TESTS PASSED!")
