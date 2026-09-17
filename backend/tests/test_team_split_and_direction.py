"""Per-team team metrics, attacking direction, and frame_id keying.

These are validated on SYNTHETIC trajectories on purpose. All three fixes
operate on pitch coordinates, and pitch coordinates only exist when
`calibration.valid` is True -- which on the real broadcast footage in
storage/uploads is never (docs/pipeline_architecture.md 6.3,
`calibration_valid_fraction = 0.0`). Synthetic input is therefore the only
way to exercise the arithmetic at all; it is not a substitute for
end-to-end verification and is not presented as one.
"""
from __future__ import annotations

from ai.computer_vision.player_tracking.trajectory import TrackingPoint
from ai.computer_vision.tactical_analysis.attacking_direction import (
    LEFT_TO_RIGHT,
    RIGHT_TO_LEFT,
    UNKNOWN,
    infer_attacking_directions,
)
from backend.pipeline.runner import (
    _build_players_by_frame,
    _positions_by_frame,
    _score_team_intelligence,
    _split_by_team,
)


def _pt(player_id, frame_id, x, y, team_id):
    return TrackingPoint(
        match_id="m", player_id=player_id, frame_id=frame_id, team_id=team_id,
        pixel_x=0.0, pixel_y=0.0, pitch_x_m=x, pitch_y_m=y,
        homography_confidence=0.9, speed=None, distance=None, acceleration=None,
    )


def _two_team_trajectories(n_frames=60):
    """team-A camped in the left third, team-B in the right third: an
    unambiguous, deliberately unrealistic separation so the direction
    inference has a clear right answer to find."""
    traj = {}
    for i in range(11):
        traj[100 + i] = [_pt(100 + i, f, 15.0 + (i % 4), 10.0 + 5 * (i % 5), "team-A")
                         for f in range(n_frames)]
    for i in range(11):
        traj[200 + i] = [_pt(200 + i, f, 90.0 + (i % 4), 10.0 + 5 * (i % 5), "team-B")
                         for f in range(n_frames)]
    return traj


# ---------------------------------------------------------------- frame_id

def test_players_by_frame_keys_by_real_frame_id_across_a_tracking_gap():
    """The bug: player 2 is occluded for frames 1-3 and re-acquired at 4.
    Keyed by list index, player 2's index 1 (frame 4) collided with player
    1's index 1 (frame 1) -- two players 'in the same frame' who were three
    frames apart."""
    traj = {
        1: [_pt(1, f, 10.0, 10.0, "A") for f in range(6)],
        2: [_pt(2, 0, 20.0, 20.0, "B"), _pt(2, 4, 21.0, 20.0, "B"), _pt(2, 5, 22.0, 20.0, "B")],
    }
    by_frame = _build_players_by_frame(traj)

    assert sorted(by_frame) == [0, 1, 2, 3, 4, 5]
    assert {p["player_id"] for p in by_frame[0]} == {1, 2}
    # Frames 1-3: player 2 is genuinely absent, and says so.
    for f in (1, 2, 3):
        assert {p["player_id"] for p in by_frame[f]} == {1}
    assert {p["player_id"] for p in by_frame[4]} == {1, 2}


def test_positions_by_frame_orders_by_frame_id_and_does_not_pad():
    traj = {
        1: [_pt(1, 0, 1.0, 1.0, "A"), _pt(1, 9, 2.0, 2.0, "A")],
        2: [_pt(2, 9, 3.0, 3.0, "B")],
    }
    frames = _positions_by_frame(traj)
    assert len(frames) == 2                  # frames 0 and 9, not 10 slots
    assert frames[0] == [(1.0, 1.0)]
    assert sorted(frames[1]) == [(2.0, 2.0), (3.0, 3.0)]


# -------------------------------------------------------------- direction

def test_direction_inferred_from_which_end_each_team_occupies():
    d = infer_attacking_directions(_two_team_trajectories())
    assert d.by_team["team-A"] == LEFT_TO_RIGHT     # nearer x=0, defends left
    assert d.by_team["team-B"] == RIGHT_TO_LEFT
    assert d.resolved
    assert d.separation_m > 50.0
    assert d.method == "heuristic_proxy"


def test_direction_unknown_when_no_pitch_coordinates():
    """The real-footage case: calibration invalid, so every pitch_x_m is
    None. Must NOT fall back to left_to_right."""
    traj = {1: [TrackingPoint("m", 1, f, "team-A", 0.0, 0.0, None, None, None, None, None, None)
                for f in range(60)]}
    d = infer_attacking_directions(traj)
    assert d.for_team("team-A") == UNKNOWN
    assert not d.resolved
    assert d.confidence == "low_upstream_confidence"
    assert "calibration invalid" in d.reason


def test_direction_unknown_when_teams_are_not_separated():
    """Both teams' mean x within a few metres -- ordering is noise."""
    traj = {}
    for i in range(11):
        traj[100 + i] = [_pt(100 + i, f, 52.0, 30.0, "team-A") for f in range(60)]
        traj[200 + i] = [_pt(200 + i, f, 53.0, 30.0, "team-B") for f in range(60)]
    d = infer_attacking_directions(traj)
    assert d.for_team("team-A") == UNKNOWN
    assert "not distinguishable from noise" in d.reason


def test_unknown_team_id_never_gets_a_default_direction():
    d = infer_attacking_directions(_two_team_trajectories())
    assert d.for_team(None) == UNKNOWN
    assert d.for_team("team-that-does-not-exist") == UNKNOWN


# -------------------------------------------------------------- per-team

def test_split_by_team_uses_modal_team_id_and_drops_unassigned():
    traj = {
        1: [_pt(1, 0, 1.0, 1.0, "A"), _pt(1, 1, 1.0, 1.0, "A"), _pt(1, 2, 1.0, 1.0, "B")],
        2: [_pt(2, 0, 1.0, 1.0, None), _pt(2, 1, 1.0, 1.0, None)],
    }
    split = _split_by_team(traj)
    assert set(split) == {"A"}          # player 1 is modal-A; player 2 has no team
    assert list(split["A"]) == [1]


def test_team_metrics_are_computed_per_team_not_pooled():
    traj = _two_team_trajectories()
    directions = infer_attacking_directions(traj)
    rows = _score_team_intelligence(None, traj, team_assignment_confidence=0.9,
                                    directions=directions)

    teams = {r["team_id"] for r in rows}
    assert teams == {"team-A", "team-B"}
    # Five metrics per team, not five for a pooled population.
    assert len(rows) == 10
    assert all(r["team_id"] != "unassigned" for r in rows)

    compact = {r["team_id"]: r for r in rows if r["metric_name"] == "compactness_score"}
    assert set(compact) == {"team-A", "team-B"}
    # Each team is tightly grouped; the POOLED population spans ~75 m of
    # pitch. A per-team compactness must reflect the team, not the union.
    for r in compact.values():
        assert r["sub_scores"]["n_players_in_team"] == 11

    for r in rows:
        assert r["sub_scores"]["attacking_direction"] in (LEFT_TO_RIGHT, RIGHT_TO_LEFT)


def test_weak_zones_use_opponent_context_when_both_teams_present():
    traj = _two_team_trajectories()
    rows = _score_team_intelligence(None, traj, team_assignment_confidence=0.9,
                                    directions=infer_attacking_directions(traj))
    wz = [r for r in rows if r["metric_name"] == "weak_zone_map"]
    assert len(wz) == 2
    for r in wz:
        assert r["sub_scores"]["basis"] == "own_density_and_opponent_presence"
        assert r["sub_scores"]["opponent_sample_size"] > 0
        assert "exposed_zone_count" in r["sub_scores"]


def test_no_team_split_still_emits_honest_gated_rows():
    """Calibration invalid / assignment failed: the API must still get
    rows, carrying low confidence -- not an empty list that renders as a
    missing panel."""
    traj = {1: [TrackingPoint("m", 1, f, None, 0.0, 0.0, None, None, None, None, None, None)
                for f in range(30)]}
    rows = _score_team_intelligence(None, traj, team_assignment_confidence=0.0)
    assert len(rows) == 5
    assert all(r["team_id"] == "unassigned" for r in rows)
    assert all(r["value"] is None for r in rows)
    assert all(r["confidence"] in ("low_upstream_confidence", "low_sample") for r in rows)
    assert all("reason_no_team_split" in r["sub_scores"] for r in rows)


def test_formation_receives_one_point_per_player_not_the_first_ten_samples():
    """detect_formation matches N points against 10 slots. Feeding it the
    first ten TRAJECTORY POINTS (the old behaviour) fed it ten consecutive
    samples of one or two players."""
    traj = _two_team_trajectories()
    rows = _score_team_intelligence(None, traj, team_assignment_confidence=0.9,
                                    directions=infer_attacking_directions(traj))
    formations = [r for r in rows if r["metric_name"] == "formation"]
    assert len(formations) == 2
    # 11 players -> 11 mean positions, one per player.
    assert all(r["sample_size"] == 11 for r in formations)
