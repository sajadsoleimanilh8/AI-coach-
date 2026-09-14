"""Opponent weakness map.

Turns measured team/player metrics into a ranked list of named weaknesses
(and strengths). Pure arithmetic over values that were already computed
upstream: this module never invents a number, and every entry it emits
carries the metric names it was derived from so a consumer can trace the
claim back to its source.

Anything that could not be measured is NOT a weakness -- it is an absence,
and callers are expected to report it as such rather than letting a missing
metric read as a clean bill of health.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

# Shared with nexus.sports.tactical so a "weakness" means the same thing in
# the findings list and in the weakness map.
STRENGTH_THRESHOLD = 65.0
WEAKNESS_THRESHOLD = 40.0

Severity = Literal["minor", "moderate", "major"]

# Metrics where a HIGH score is bad. Everything else is "higher is better".
_INVERTED_METRICS: frozenset[str] = frozenset()

# Human-facing labels, so the map does not leak raw column names into prose.
_METRIC_LABELS: dict[str, str] = {
    "compactness_score": "defensive compactness",
    "formation_stability_score": "shape stability",
    "pressing_intensity_score": "pressing intensity",
    "decision_making_score": "decision making",
    "passing_vision_score": "passing vision",
    "press_resistance_score": "press resistance",
    "first_touch_score": "first touch",
    "body_orientation_score": "body orientation",
    "defensive_positioning_score": "defensive positioning",
    "finishing_efficiency_score": "finishing efficiency",
    "off_ball_movement_score": "off-ball movement",
    "scanning_behavior_score": "scanning behaviour",
    "pressing_success_rate": "pressing success rate",
    "transition_speed": "transition speed",
    "territory_share": "territorial dominance",
}


def metric_label(metric_name: str) -> str:
    return _METRIC_LABELS.get(metric_name, metric_name.replace("_", " "))


@dataclass(frozen=True)
class Weakness:
    """One measured weakness, with the evidence that produced it."""

    key: str
    label: str
    severity: Severity
    value: float
    supporting_metrics: list[str]
    confidence: str
    sample_size: int
    evidence: str
    zone: str | None = None

    def as_line(self) -> str:
        zone = f" [{self.zone}]" if self.zone else ""
        return (
            f"{self.label}{zone} ({self.severity}): {self.evidence} "
            f"[confidence={self.confidence}, sample_size={self.sample_size}]"
        )


@dataclass(frozen=True)
class Strength:
    key: str
    label: str
    value: float
    supporting_metrics: list[str]
    confidence: str
    sample_size: int
    evidence: str

    def as_line(self) -> str:
        return (
            f"{self.label}: {self.evidence} "
            f"[confidence={self.confidence}, sample_size={self.sample_size}]"
        )


@dataclass
class WeaknessMap:
    """Ranked weaknesses/strengths plus the metrics that could not be read."""

    weaknesses: list[Weakness] = field(default_factory=list)
    strengths: list[Strength] = field(default_factory=list)
    unmeasured: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.weaknesses and not self.strengths


def _severity(value: float) -> Severity:
    """Distance below the weakness threshold, bucketed."""
    if value < WEAKNESS_THRESHOLD * 0.5:
        return "major"
    if value < WEAKNESS_THRESHOLD * 0.8:
        return "moderate"
    return "minor"


def _oriented(metric_name: str, value: float) -> float:
    """Normalize so that, after this, higher always means better."""
    return 100.0 - value if metric_name in _INVERTED_METRICS else value


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    return float(value) if isinstance(value, (int, float)) else None


@dataclass(frozen=True)
class MetricInput:
    """The subset of a metric row this module needs.

    Deliberately not the backend's ORM row nor NEXUS's SportsMetric: keeping
    this a plain local structure is what lets both call the same scorer.
    """

    metric_name: str
    value: Any
    confidence: str
    sample_size: int
    sub_scores: dict[str, Any] = field(default_factory=dict)
    available: bool = True


def _zone_weaknesses(metric: MetricInput) -> list[Weakness]:
    """Under-occupied pitch zones from a weak_zone_map's sub_scores.

    The zone densities are shares of the team's own tracked positions. When
    `basis` says opponent context was unavailable, that caveat is carried
    into the evidence string rather than dropped -- an under-occupied zone
    measured against your own density alone is a weaker claim than one
    measured against where the opponent actually played.
    """
    sub = metric.sub_scores or {}
    zone_items = [
        (name, _numeric(share))
        for name, share in sub.items()
        if name.startswith("zone_") and _numeric(share) is not None
    ]
    if not zone_items:
        return []

    basis = str(sub.get("basis", "unknown"))
    caveat = (
        " (measured against own player density only; opponent-relative "
        "exposure was not available)"
        if basis == "own_density_only"
        else ""
    )

    empty = [name for name, share in zone_items if share == 0.0]
    if not empty:
        return []

    count = _numeric(sub.get("under_occupied_zone_count"))
    occupied = [(n, s) for n, s in zone_items if s > 0.0]
    occupied_text = ", ".join(
        f"{name}={share:.3f}" for name, share in sorted(occupied, key=lambda kv: -kv[1])
    )
    return [
        Weakness(
            key="under_occupied_zones",
            label="under-occupied pitch zones",
            severity="major" if len(empty) >= len(zone_items) * 0.75 else "moderate",
            value=float(len(empty)),
            supporting_metrics=["weak_zone_map"],
            confidence=metric.confidence,
            sample_size=metric.sample_size,
            zone=", ".join(sorted(empty)),
            evidence=(
                f"{len(empty)} of {len(zone_items)} pitch zones recorded zero "
                f"tracked presence"
                + (
                    f" (weak_zone_map under_occupied_zone_count={count:.0f})"
                    if count is not None
                    else ""
                )
                + (f"; presence concentrated in {occupied_text}" if occupied_text else "")
                + caveat
            ),
        )
    ]


def build_weakness_map(
    team_metrics: Sequence[MetricInput],
    player_metrics: Sequence[MetricInput] = (),
) -> WeaknessMap:
    """Rank measured weaknesses and strengths.

    Unavailable metrics are recorded in `unmeasured` and never scored --
    a metric that could not be computed says nothing about the opponent,
    and treating it as neutral would quietly invent a finding.
    """
    result = WeaknessMap()

    for metric in list(team_metrics) + list(player_metrics):
        if not metric.available:
            result.unmeasured.append(metric.metric_name)
            continue

        if metric.metric_name == "weak_zone_map":
            result.weaknesses.extend(_zone_weaknesses(metric))
            continue

        raw = _numeric(metric.value)
        if raw is None:
            # A label-valued metric (e.g. formation "3-4-3") is context, not
            # something with a good/bad direction.
            continue

        oriented = _oriented(metric.metric_name, raw)
        label = metric_label(metric.metric_name)
        evidence = f"{metric.metric_name}={raw:.1f}"

        if oriented < WEAKNESS_THRESHOLD:
            result.weaknesses.append(
                Weakness(
                    key=metric.metric_name,
                    label=label,
                    severity=_severity(oriented),
                    value=raw,
                    supporting_metrics=[metric.metric_name],
                    confidence=metric.confidence,
                    sample_size=metric.sample_size,
                    evidence=(
                        f"{evidence}, below the {WEAKNESS_THRESHOLD:.0f} weakness threshold"
                    ),
                )
            )
        elif oriented >= STRENGTH_THRESHOLD:
            result.strengths.append(
                Strength(
                    key=metric.metric_name,
                    label=label,
                    value=raw,
                    supporting_metrics=[metric.metric_name],
                    confidence=metric.confidence,
                    sample_size=metric.sample_size,
                    evidence=(
                        f"{evidence}, at or above the {STRENGTH_THRESHOLD:.0f} "
                        "strength threshold"
                    ),
                )
            )

    severity_rank = {"major": 0, "moderate": 1, "minor": 2}
    result.weaknesses.sort(key=lambda w: (severity_rank[w.severity], w.value))
    result.strengths.sort(key=lambda s: -s.value)
    return result
