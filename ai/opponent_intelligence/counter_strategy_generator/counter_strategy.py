"""Counter-strategy generator.

Maps named weaknesses from `weakness_map` onto concrete, coachable
adjustments. Every adjustment is anchored to the weakness that produced it
and to that weakness's supporting metrics, so nothing here can be stated
without a measured reason behind it.

The playbook below is football knowledge, not measurement: the *trigger* for
each entry comes from data, the phrasing of the instruction does not. That
split is deliberate -- it keeps tactical advice out of the LLM (where it
would be ungrounded) while keeping invented numbers out of the advice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ai.opponent_intelligence.weakness_map.weakness_map import (
    Strength,
    Weakness,
    WeaknessMap,
)


@dataclass(frozen=True)
class Adjustment:
    """One in-game change, tied to the weakness that motivated it."""

    instruction: str
    targets_weakness: str
    rationale: str
    supporting_metrics: list[str]
    priority: int = 2

    def as_line(self) -> str:
        return (
            f"{self.instruction} "
            f"[targets: {self.targets_weakness}; because {self.rationale}; "
            f"from {', '.join(self.supporting_metrics)}]"
        )


@dataclass(frozen=True)
class Principle:
    """One principle of play, with the observation that justifies it."""

    statement: str
    grounded_in: str
    supporting_metrics: list[str]

    def as_line(self) -> str:
        metrics = ", ".join(self.supporting_metrics) or "no metric"
        return f"{self.statement} [grounded in: {self.grounded_in}; from {metrics}]"


@dataclass
class GamePlanShape:
    shape: str
    reason: str
    supporting_metrics: list[str] = field(default_factory=list)


# weakness key -> (instruction, rationale template, priority)
_PLAYBOOK: dict[str, tuple[str, str, int]] = {
    "compactness_score": (
        "Attack the space between their lines with a runner from deep rather "
        "than feet-to-feet passes into a congested block.",
        "their block measured very compact, so the space is behind and around "
        "it rather than through it",
        1,
    ),
    "formation_stability_score": (
        "Switch the point of attack early and often; make them re-form on the "
        "move before they settle into shape.",
        "their shape was measured as unstable frame to frame",
        1,
    ),
    "pressing_intensity_score": (
        "Build out from the back through the goalkeeper and centre-backs; take "
        "the free first pass they are conceding.",
        "their pressing intensity measured low",
        1,
    ),
    "press_resistance_score": (
        "Press their first receiver on the half-turn and screen the backward "
        "pass to force a turnover high.",
        "their press resistance measured low",
        1,
    ),
    "decision_making_score": (
        "Show them the sideline and delay rather than diving in; let the "
        "hurried choice come to you.",
        "their decision making under pressure measured low",
        2,
    ),
    "passing_vision_score": (
        "Screen the long switch and compress the near side; their release "
        "valve is the pass they were not finding.",
        "their passing vision measured low",
        2,
    ),
    "first_touch_score": (
        "Press the moment the ball travels, not once it is controlled -- "
        "arrive on the touch.",
        "their first touch measured low",
        2,
    ),
    "body_orientation_score": (
        "Press from their blind side so the receiver has to turn into "
        "pressure they have not scanned.",
        "their body orientation on receiving measured low",
        2,
    ),
    "scanning_behavior_score": (
        "Disguise the pressing trigger and arrive late into the tackle zone; "
        "they are not checking their shoulder before receiving.",
        "their scanning behaviour measured low",
        2,
    ),
    "defensive_positioning_score": (
        "Overload the far post and the cut-back zone; their defensive "
        "positioning is the measured gap.",
        "their defensive positioning measured low",
        1,
    ),
    "off_ball_movement_score": (
        "Man-mark their most advanced runner and hold a spare defender; "
        "there is little off-ball movement to track beyond that.",
        "their off-ball movement measured low",
        3,
    ),
    "finishing_efficiency_score": (
        "Concede the low-value shot from distance and defend the box "
        "instead of stepping out.",
        "their finishing efficiency measured low",
        3,
    ),
    "under_occupied_zones": (
        "Target the zones where they recorded no presence -- circulate the "
        "ball there deliberately and early.",
        "measured tracking showed zero presence in those pitch zones",
        1,
    ),
}

# Which shape to propose against a dominant weakness. First match wins.
_SHAPE_BY_WEAKNESS: dict[str, tuple[str, str]] = {
    "compactness_score": (
        "4-3-3",
        "wide, high wingers stretch a very compact block horizontally and "
        "create the gaps it does not concede centrally",
    ),
    "pressing_intensity_score": (
        "4-2-3-1",
        "a double pivot takes the free build-out their low pressing intensity "
        "concedes and turns it into controlled possession",
    ),
    "formation_stability_score": (
        "3-4-3",
        "wing-backs give repeated width on both flanks to keep an unstable "
        "shape re-forming on the move",
    ),
    "defensive_positioning_score": (
        "4-2-3-1",
        "a No.10 plus overlapping full-backs attack the cut-back zone their "
        "defensive positioning leaves open",
    ),
    "under_occupied_zones": (
        "4-3-3",
        "three forwards occupy the full width of the pitch, including the "
        "zones where they recorded no presence",
    ),
}

_DEFAULT_SHAPE = (
    "no shape proposed",
    "no measured weakness was strong enough to justify a specific shape; "
    "proposing one anyway would be a guess",
)


def propose_shape(weakness_map: WeaknessMap) -> GamePlanShape:
    """Pick a shape from the highest-severity weakness that maps to one.

    Returns an explicit "no shape proposed" rather than a default formation
    when nothing measured supports a choice.
    """
    for weakness in weakness_map.weaknesses:
        match = _SHAPE_BY_WEAKNESS.get(weakness.key)
        if match is not None:
            shape, reason = match
            return GamePlanShape(
                shape=shape,
                reason=f"{reason} ({weakness.evidence})",
                supporting_metrics=list(weakness.supporting_metrics),
            )
    return GamePlanShape(shape=_DEFAULT_SHAPE[0], reason=_DEFAULT_SHAPE[1])


def derive_principles(
    weakness_map: WeaknessMap, *, max_principles: int = 5
) -> list[Principle]:
    """Principles of play, one per measured weakness or strength.

    Opponent strengths become "deny" principles; weaknesses become "attack"
    principles. Capped at `max_principles` because a plan a coach cannot
    hold in their head is not a plan.
    """
    principles: list[Principle] = []

    for strength in weakness_map.strengths[:2]:
        principles.append(
            Principle(
                statement=f"Deny their {strength.label} -- do not contest it on their terms.",
                grounded_in=strength.evidence,
                supporting_metrics=list(strength.supporting_metrics),
            )
        )

    for weakness in weakness_map.weaknesses:
        if len(principles) >= max_principles:
            break
        principles.append(
            Principle(
                statement=f"Attack their {weakness.label} repeatedly and early.",
                grounded_in=weakness.evidence,
                supporting_metrics=list(weakness.supporting_metrics),
            )
        )

    return principles[:max_principles]


def generate_adjustments(
    weakness_map: WeaknessMap, *, max_adjustments: int = 5
) -> list[Adjustment]:
    """One adjustment per weakness that the playbook covers.

    A weakness with no playbook entry yields nothing: inventing an
    instruction for an unrecognised metric is exactly the failure mode this
    module exists to prevent.
    """
    adjustments: list[Adjustment] = []
    seen: set[str] = set()

    for weakness in weakness_map.weaknesses:
        if len(adjustments) >= max_adjustments:
            break
        entry = _PLAYBOOK.get(weakness.key)
        if entry is None or weakness.key in seen:
            continue
        seen.add(weakness.key)
        instruction, rationale, priority = entry
        adjustments.append(
            Adjustment(
                instruction=instruction,
                targets_weakness=weakness.label,
                rationale=f"{rationale} ({weakness.evidence})",
                supporting_metrics=list(weakness.supporting_metrics),
                priority=priority,
            )
        )

    adjustments.sort(key=lambda a: a.priority)
    return adjustments


def uncovered_weaknesses(
    weaknesses: Sequence[Weakness], adjustments: Sequence[Adjustment]
) -> list[str]:
    """Weaknesses that produced no adjustment, so the gap can be reported."""
    covered = {a.targets_weakness for a in adjustments}
    return [w.label for w in weaknesses if w.label not in covered]


__all__ = [
    "Adjustment",
    "GamePlanShape",
    "Principle",
    "Strength",
    "Weakness",
    "derive_principles",
    "generate_adjustments",
    "propose_shape",
    "uncovered_weaknesses",
]
