"""What-if engine: recomputation, provenance, and refusal to invent."""
from __future__ import annotations

import pytest

from ai.computer_vision.player_tracking.trajectory import TrackingPoint
from ai.simulation_ai.what_if_analysis.engine import (
    Intervention,
    SimulationError,
    simulate,
)


def _pt(pid, f, x, y, team="A", speed=3.0):
    return TrackingPoint("m", pid, f, team, 0.0, 0.0, x, y, 0.9, speed, None, None)


def _teams(n_frames=40):
    """team A spread around (30, 34); team B around (75, 34)."""
    a, b = {}, {}
    for i in range(10):
        ax = 30.0 + (i % 5) * 6.0
        ay = 20.0 + (i // 5) * 18.0
        a[100 + i] = [_pt(100 + i, f, ax + 0.10 * f, ay, "A") for f in range(n_frames)]
        bx = 75.0 + (i % 5) * 4.0
        by = 22.0 + (i // 5) * 16.0
        b[200 + i] = [_pt(200 + i, f, bx, by, "B") for f in range(n_frames)]
    return {"A": a, "B": b}


def test_increasing_compactness_shrinks_the_hull_and_raises_the_score():
    res = simulate(_teams(), [Intervention("compactness", team_id="A", pct=25.0)],
                   team_assignment_confidence=0.9)
    comp = next(m for m in res.metrics if m.metric_name == "compactness_score")
    assert comp.baseline_value is not None
    assert comp.simulated_value > comp.baseline_value
    assert comp.delta > 0
    assert comp.simulated_sub_scores["hull_area_m2"] < comp.baseline_sub_scores["hull_area_m2"]


def test_every_metric_names_its_real_input_and_the_parameter_changed():
    res = simulate(_teams(), [Intervention("compactness", team_id="A", pct=-10.0)],
                   team_assignment_confidence=0.9)
    assert res.metrics
    for m in res.metrics:
        assert "real tracked pitch positions" in m.derived_from
        assert "positioned samples" in m.derived_from
        assert m.parameter_changed == "team A compactness -10.0%"
        assert m.recomputed_by.startswith("ai.")
        assert m.method == "heuristic_proxy"


def test_persisted_metric_provenance_is_carried_to_simulated_metric():
    provenance = {"A": {"compactness_score": {
        "metric_id": "metric-123", "method": "deterministic", "confidence": "normal"
    }}}
    res = simulate(_teams(), [Intervention("compactness", team_id="A", pct=5.0)],
                   team_assignment_confidence=0.9, baseline_provenance=provenance)
    metric = next(m for m in res.metrics if m.metric_name == "compactness_score")
    payload = metric.as_dict()
    assert payload["source_metric_id"] == "metric-123"
    assert payload["source_method"] == "deterministic"
    assert payload["source_confidence"] == "normal"


def test_result_declares_it_is_not_reinforcement_learning():
    res = simulate(_teams(), [Intervention("compactness", team_id="A", pct=5.0)],
                   team_assignment_confidence=0.9)
    d = res.as_dict()
    assert d["is_reinforcement_learning"] is False
    assert any("Not reinforcement learning" in c for c in d["caveats"])
    assert any("does not mean the outcome would change" in c for c in d["caveats"])


def test_no_pitch_coordinates_yields_no_simulated_values():
    """The real-footage case: calibration invalid, so the baseline metrics
    are already gated to None. The simulation must not fill that in."""
    traj = {"A": {1: [TrackingPoint("m", 1, f, "A", 0.0, 0.0, None, None, None, None, None, None)
                      for f in range(40)]}}
    res = simulate(traj, [Intervention("compactness", team_id="A", pct=20.0)],
                   team_assignment_confidence=0.9)
    assert all(m.simulated_value is None for m in res.metrics)
    assert all(m.delta is None for m in res.metrics)
    assert all(m.confidence == "low_upstream_confidence" for m in res.metrics)
    assert res.unavailable


def test_low_team_confidence_propagates_rather_than_being_bypassed():
    res = simulate(_teams(), [Intervention("compactness", team_id="A", pct=20.0)],
                   team_assignment_confidence=0.0)
    assert all(m.baseline_value is None for m in res.metrics)
    assert all(m.simulated_value is None for m in res.metrics)


def test_empty_team_split_reports_unavailable_not_an_empty_success():
    res = simulate({}, [Intervention("compactness", team_id="A", pct=10.0)],
                   team_assignment_confidence=0.9)
    assert res.metrics == []
    assert any("no measured baseline" in u or "no team-split" in u for u in res.unavailable)


def test_removing_a_player_changes_the_shape_metrics():
    res = simulate(_teams(), [Intervention("remove_player", team_id="A", player_id=104)],
                   team_assignment_confidence=0.9)
    formation = next(m for m in res.metrics if m.metric_name == "formation")
    assert formation.baseline_sub_scores  # 10 players
    assert "remove player 104" in formation.parameter_changed


def test_swap_player_replays_the_donors_real_positions():
    teams = _teams()
    res = simulate(teams, [Intervention("swap_player", team_id="A",
                                        player_id=100, other_player_id=109)],
                   team_assignment_confidence=0.9)
    assert res.metrics
    assert any("takes player 109's tracked movement" in m.parameter_changed for m in res.metrics)


def test_transition_speed_moves_stability_not_formation_label():
    res = simulate(_teams(), [Intervention("transition_speed", team_id="A", pct=50.0)],
                   team_assignment_confidence=0.9)
    names = {m.metric_name for m in res.metrics}
    assert names == {"formation_stability_score", "compactness_score"}
    assert any("does not model fatigue" in c for c in res.caveats)


def test_unknown_intervention_is_rejected_loudly():
    with pytest.raises(SimulationError, match="unknown intervention kind"):
        simulate(_teams(), [Intervention("make_them_better", team_id="A", pct=10.0)],
                 team_assignment_confidence=0.9)


def test_intervention_targeting_an_untracked_team_is_rejected():
    with pytest.raises(SimulationError, match="no tracked players"):
        simulate(_teams(), [Intervention("compactness", team_id="Z", pct=10.0)],
                 team_assignment_confidence=0.9)


def test_removing_an_untracked_player_is_rejected():
    with pytest.raises(SimulationError, match="not tracked"):
        simulate(_teams(), [Intervention("remove_player", team_id="A", player_id=999)],
                 team_assignment_confidence=0.9)


def test_missing_pct_is_rejected():
    with pytest.raises(SimulationError, match="requires `pct`"):
        simulate(_teams(), [Intervention("compactness", team_id="A")],
                 team_assignment_confidence=0.9)


def test_baseline_trajectories_are_not_mutated_by_a_simulation():
    teams = _teams()
    before = teams["A"][100][0].pitch_x_m
    simulate(teams, [Intervention("compactness", team_id="A", pct=90.0)],
             team_assignment_confidence=0.9)
    assert teams["A"][100][0].pitch_x_m == before
