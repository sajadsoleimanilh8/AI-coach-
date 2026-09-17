"""
Unit tests for formation_detection.py
"""



from ai.computer_vision.tactical_analysis.constants import PITCH_LENGTH_M, PITCH_WIDTH_M
from ai.computer_vision.tactical_analysis.formation_detection import detect_formation
from ai.computer_vision.tactical_analysis.formation_templates import FORMATION_TEMPLATES


def test_known_433_template_matches_high_confidence():
    slots_433 = FORMATION_TEMPLATES["4-3-3"]
    positions = [(x_frac * PITCH_LENGTH_M, y_frac * PITCH_WIDTH_M) for x_frac, y_frac in slots_433]

    result = detect_formation(positions, team_assignment_confidence=0.9)
    assert result["value"] == "4-3-3"
    assert result["confidence"] == "normal"
    assert result["sub_scores"]["matching_confidence_pct"] > 90.0


def test_scrambled_positions_yields_lower_confidence():
    import numpy as np

    rng = np.random.default_rng(42)
    positions = [(float(x), float(y)) for x, y in zip(rng.uniform(0, 105, 10), rng.uniform(0, 68, 10))]

    result = detect_formation(positions, team_assignment_confidence=0.9)
    assert result["value"] is not None
    assert result["sub_scores"]["matching_confidence_pct"] < 90.0


def test_fewer_than_8_players_flags_low_sample():
    positions = [(20.0, 30.0), (30.0, 40.0), (40.0, 50.0)]  # 3 players
    result = detect_formation(positions, team_assignment_confidence=0.9)

    assert result["value"] is None
    assert result["confidence"] == "low_sample"


def test_low_team_assignment_confidence_flags_low_upstream():
    slots_433 = FORMATION_TEMPLATES["4-3-3"]
    positions = [(x_frac * PITCH_LENGTH_M, y_frac * PITCH_WIDTH_M) for x_frac, y_frac in slots_433]

    result = detect_formation(positions, team_assignment_confidence=0.0)
    assert result["value"] is None
    assert result["confidence"] == "low_upstream_confidence"


if __name__ == "__main__":
    test_known_433_template_matches_high_confidence()
    test_scrambled_positions_yields_lower_confidence()
    test_fewer_than_8_players_flags_low_sample()
    test_low_team_assignment_confidence_flags_low_upstream()
    print("ALL FORMATION DETECTION TESTS PASSED!")
