"""Deterministic game plan.

Builds the game-plan section of a CoachReport from measured metrics only,
using `ai/opponent_intelligence/`'s weakness map and counter-strategy
generator. Like `tactical.derive_findings`, there is no LLM call in this
module: the plan is Python arithmetic plus a fixed playbook, and the LLM's
only job downstream is to narrate what is already decided here.

That ordering is what makes the grounding guarantee testable. If the model
were asked to invent the plan, "every number is traceable" would be a hope;
because the plan is built first, it is a property of the data structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ai.opponent_intelligence.counter_strategy_generator.counter_strategy import (
    Adjustment,
    GamePlanShape,
    Principle,
    derive_principles,
    generate_adjustments,
    propose_shape,
    uncovered_weaknesses,
)
from ai.opponent_intelligence.weakness_map.weakness_map import (
    MetricInput,
    Strength,
    Weakness,
    build_weakness_map,
)
from nexus.sports.adapter import MatchAnalysis, SportsMetric

# Below this, the report is explicitly a partial one. Set where it is
# because a plan built on under a third of the intended metrics is a
# sampling artefact, not a read on the opponent.
LOW_COVERAGE_THRESHOLD = 0.35


@dataclass
class GamePlan:
    """The plan and everything needed to audit it."""

    opponent_strengths: list[Strength] = field(default_factory=list)
    opponent_weaknesses: list[Weakness] = field(default_factory=list)
    proposed_shape: GamePlanShape | None = None
    principles: list[Principle] = field(default_factory=list)
    adjustments: list[Adjustment] = field(default_factory=list)
    uncovered: list[str] = field(default_factory=list)
    unmeasured: list[str] = field(default_factory=list)
    is_partial: bool = False
    partial_reason: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.opponent_weaknesses and not self.opponent_strengths

    def as_prompt_context(self) -> str:
        """The plan rendered for the prompt. Every number the narrative is
        allowed to use appears here verbatim."""
        blocks: list[str] = []

        if self.is_partial:
            blocks.append(f"PARTIAL PLAN: {self.partial_reason}")

        blocks.append(
            "Opponent strengths (measured):\n"
            + (
                "\n".join(f"- {s.as_line()}" for s in self.opponent_strengths)
                or "- None measured."
            )
        )
        blocks.append(
            "Opponent weaknesses (measured):\n"
            + (
                "\n".join(f"- {w.as_line()}" for w in self.opponent_weaknesses)
                or "- None measured."
            )
        )

        if self.proposed_shape is not None:
            blocks.append(
                f"Proposed shape: {self.proposed_shape.shape}\n"
                f"- reason: {self.proposed_shape.reason}"
            )

        blocks.append(
            "Principles of play:\n"
            + ("\n".join(f"- {p.as_line()}" for p in self.principles) or "- None derived.")
        )
        blocks.append(
            "In-game adjustments (each tied to a named observed weakness):\n"
            + ("\n".join(f"- {a.as_line()}" for a in self.adjustments) or "- None derived.")
        )

        if self.uncovered:
            blocks.append(
                "Weaknesses with no adjustment in the playbook (report as open):\n"
                + "\n".join(f"- {label}" for label in self.uncovered)
            )
        if self.unmeasured:
            blocks.append(
                "Metrics that could not be measured (never treat as neutral or good):\n"
                + "\n".join(f"- {name}" for name in sorted(set(self.unmeasured)))
            )

        return "\n\n".join(blocks)


def _to_input(metric: SportsMetric) -> MetricInput:
    return MetricInput(
        metric_name=metric.metric_name,
        value=metric.value,
        confidence=metric.confidence,
        sample_size=metric.sample_size,
        sub_scores=metric.sub_scores,
        available=metric.is_available,
    )


def _aggregate_player_metrics(player_metrics: list[SportsMetric]) -> list[MetricInput]:
    """Collapse per-player metrics into one team-wide row per metric name.

    A weakness map over 30 separate `decision_making_score` rows would rank
    the same weakness thirty times. Averaging matches what
    `tactical._player_findings` already does, so the two agree.
    """
    by_name: dict[str, list[SportsMetric]] = {}
    unavailable: dict[str, SportsMetric] = {}
    for metric in player_metrics:
        if metric.is_available and isinstance(metric.value, (int, float)):
            by_name.setdefault(metric.metric_name, []).append(metric)
        else:
            unavailable.setdefault(metric.metric_name, metric)

    rows: list[MetricInput] = []
    for name, metrics in by_name.items():
        values = [float(m.value) for m in metrics]
        rows.append(
            MetricInput(
                metric_name=name,
                value=sum(values) / len(values),
                confidence=(
                    "normal" if all(m.confidence == "normal" for m in metrics) else "low_sample"
                ),
                sample_size=sum(m.sample_size for m in metrics),
                sub_scores={"n_players": len(metrics)},
                available=True,
            )
        )

    # Only report a metric as unmeasured when NO player produced a value.
    for name, metric in unavailable.items():
        if name not in by_name:
            rows.append(_to_input(metric))

    return rows


def build_game_plan(
    analysis: MatchAnalysis,
    *,
    low_coverage_threshold: float = LOW_COVERAGE_THRESHOLD,
) -> GamePlan:
    """Derive the full game plan from one match analysis.

    On low coverage the plan is still built -- but flagged partial, so the
    narrative is required to say so rather than presenting a thin read as a
    confident one.
    """
    team_inputs = [_to_input(m) for m in analysis.team_metrics]
    player_inputs = _aggregate_player_metrics(analysis.player_metrics)

    weakness_map = build_weakness_map(team_inputs, player_inputs)
    shape = propose_shape(weakness_map)
    principles = derive_principles(weakness_map)
    adjustments = generate_adjustments(weakness_map)

    plan = GamePlan(
        opponent_strengths=weakness_map.strengths,
        opponent_weaknesses=weakness_map.weaknesses,
        proposed_shape=shape,
        principles=principles,
        adjustments=adjustments,
        uncovered=uncovered_weaknesses(weakness_map.weaknesses, adjustments),
        unmeasured=weakness_map.unmeasured,
    )

    reasons: list[str] = []
    if analysis.coverage < low_coverage_threshold:
        reasons.append(
            f"metric coverage is {analysis.coverage:.1%}, below the "
            f"{low_coverage_threshold:.0%} threshold for a full plan"
        )
    if analysis.timeline is None:
        reasons.append(
            "no tactical timeline was available, so phase, pressing, "
            "transition and territory context is missing"
        )
    if not analysis.team_metrics:
        reasons.append("no team-level metrics were returned for this match")

    if reasons:
        plan.is_partial = True
        plan.partial_reason = "; ".join(reasons)

    return plan
