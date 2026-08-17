from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from nexus.sports.adapter import MatchAnalysis, SportsMetric

_STRENGTH_THRESHOLD = 65.0
_WEAKNESS_THRESHOLD = 40.0

_TEAM_METRIC_AREAS: dict[str, str] = {
    "compactness_score": "defensive_compactness",
    "formation_stability_score": "shape_stability",
    "pressing_intensity_score": "pressing_intensity",
}

_PLAYER_METRIC_AREAS: dict[str, str] = {
    "decision_making_score": "decision_making",
    "passing_vision_score": "passing_vision",
    "press_resistance_score": "press_resistance",
    "first_touch_score": "first_touch",
    "body_orientation_score": "body_orientation",
    "defensive_positioning_score": "defensive_positioning",
    "finishing_efficiency_score": "finishing_efficiency",
    "off_ball_movement_score": "off_ball_movement",
    "scanning_behavior_score": "scanning_behavior",
}


@dataclass
class TacticalFinding:
    area: str
    assessment: Literal["strength", "neutral", "weakness"]
    supporting_metrics: list[str]
    confidence: str
    explanation: str


def _assess(value: float) -> Literal["strength", "neutral", "weakness"]:
    if value >= _STRENGTH_THRESHOLD:
        return "strength"
    if value < _WEAKNESS_THRESHOLD:
        return "weakness"
    return "neutral"


def _numeric_value(metric: SportsMetric) -> float | None:
    return metric.value if isinstance(metric.value, (int, float)) else None


def _team_findings(team_metrics: list[SportsMetric]) -> list[TacticalFinding]:
    findings: list[TacticalFinding] = []
    for metric in team_metrics:
        if not metric.is_available:
            continue
        area = _TEAM_METRIC_AREAS.get(metric.metric_name)
        value = _numeric_value(metric)
        if area is None or value is None:
            continue
        assessment = _assess(value)
        findings.append(
            TacticalFinding(
                area=area,
                assessment=assessment,
                supporting_metrics=[metric.metric_name],
                confidence=metric.confidence,
                explanation=(
                    f"{area.replace('_', ' ')}: {metric.metric_name}={value:.1f} "
                    f"({assessment}), confidence={metric.confidence}, "
                    f"sample_size={metric.sample_size}."
                ),
            )
        )
    return findings


def _player_findings(player_metrics: list[SportsMetric]) -> list[TacticalFinding]:
    by_metric_name: dict[str, list[SportsMetric]] = {}
    for metric in player_metrics:
        if not metric.is_available or _numeric_value(metric) is None:
            continue
        by_metric_name.setdefault(metric.metric_name, []).append(metric)

    findings: list[TacticalFinding] = []
    for metric_name, metrics in by_metric_name.items():
        area = _PLAYER_METRIC_AREAS.get(metric_name)
        if area is None:
            continue
        values = [_numeric_value(m) for m in metrics]
        avg_value = sum(values) / len(values)
        assessment = _assess(avg_value)
        confidence = "normal" if all(m.confidence == "normal" for m in metrics) else "low_sample"
        findings.append(
            TacticalFinding(
                area=area,
                assessment=assessment,
                supporting_metrics=[metric_name],
                confidence=confidence,
                explanation=(
                    f"{area.replace('_', ' ')}: team-wide average {metric_name}="
                    f"{avg_value:.1f} across {len(metrics)} player(s) ({assessment})."
                ),
            )
        )
    return findings


def derive_findings(analysis: MatchAnalysis) -> list[TacticalFinding]:
    """Deterministic thresholding over AVAILABLE metrics only (principle
    1/3) — no LLM call anywhere in this function."""
    return _team_findings(analysis.team_metrics) + _player_findings(analysis.player_metrics)
