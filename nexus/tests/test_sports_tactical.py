from __future__ import annotations

from nexus.sports.adapter import MatchAnalysis, SportsMetric
from nexus.sports.tactical import derive_findings


def _metric(name: str, value, confidence: str = "normal", sample_size: int = 10, player_id=None) -> SportsMetric:
    return SportsMetric(
        metric_name=name, value=value, method="deterministic", confidence=confidence,
        sample_size=sample_size, sub_scores={}, player_id=player_id,
    )


def _analysis(team_metrics=None, player_metrics=None) -> MatchAnalysis:
    return MatchAnalysis(
        match_id="m1", team_metrics=team_metrics or [], player_metrics=player_metrics or [],
        unavailable=[], coverage=1.0,
    )


def test_no_finding_derived_from_an_unavailable_team_metric() -> None:
    metric = _metric("compactness_score", None, confidence="low_upstream_confidence")
    findings = derive_findings(_analysis(team_metrics=[metric]))

    assert findings == []


def test_no_finding_derived_from_an_unavailable_player_metric() -> None:
    metric = _metric("decision_making_score", None, confidence="low_upstream_confidence", player_id=1)
    findings = derive_findings(_analysis(player_metrics=[metric]))

    assert findings == []


def test_available_team_metric_produces_a_finding() -> None:
    metric = _metric("compactness_score", 80.0)
    findings = derive_findings(_analysis(team_metrics=[metric]))

    assert len(findings) == 1
    assert findings[0].area == "defensive_compactness"
    assert findings[0].assessment == "strength"
    assert findings[0].supporting_metrics == ["compactness_score"]


def test_thresholds_strength_neutral_weakness() -> None:
    strong = derive_findings(_analysis(team_metrics=[_metric("compactness_score", 70.0)]))[0]
    neutral = derive_findings(_analysis(team_metrics=[_metric("compactness_score", 50.0)]))[0]
    weak = derive_findings(_analysis(team_metrics=[_metric("compactness_score", 20.0)]))[0]

    assert strong.assessment == "strength"
    assert neutral.assessment == "neutral"
    assert weak.assessment == "weakness"


def test_unmapped_metric_name_produces_no_finding() -> None:
    metric = _metric("weak_zone_map", 42.0)
    findings = derive_findings(_analysis(team_metrics=[metric]))

    assert findings == []


def test_formation_string_value_does_not_crash_and_produces_no_finding() -> None:
    metric = _metric("formation", "4-3-3")
    findings = derive_findings(_analysis(team_metrics=[metric]))

    assert findings == []


def test_player_metrics_are_averaged_team_wide() -> None:
    metrics = [
        _metric("decision_making_score", 80.0, player_id=1),
        _metric("decision_making_score", 60.0, player_id=2),
    ]
    findings = derive_findings(_analysis(player_metrics=metrics))

    assert len(findings) == 1
    finding = findings[0]
    assert finding.area == "decision_making"
    assert "70.0" in finding.explanation  # (80+60)/2
    assert finding.assessment == "strength"


def test_unavailable_players_are_excluded_from_the_average() -> None:
    metrics = [
        _metric("decision_making_score", 80.0, player_id=1),
        _metric("decision_making_score", None, confidence="low_upstream_confidence", player_id=2),
    ]
    findings = derive_findings(_analysis(player_metrics=metrics))

    assert len(findings) == 1
    assert "80.0" in findings[0].explanation


def test_player_finding_confidence_reflects_low_sample_contributors() -> None:
    metrics = [
        _metric("decision_making_score", 80.0, confidence="normal", player_id=1),
        _metric("decision_making_score", 60.0, confidence="low_sample", player_id=2),
    ]
    findings = derive_findings(_analysis(player_metrics=metrics))

    assert findings[0].confidence == "low_sample"


def test_mixed_team_and_player_findings_both_present() -> None:
    team = _metric("pressing_intensity_score", 75.0)
    player = _metric("off_ball_movement_score", 30.0, player_id=1)
    findings = derive_findings(_analysis(team_metrics=[team], player_metrics=[player]))

    areas = {f.area for f in findings}
    assert areas == {"pressing_intensity", "off_ball_movement"}
