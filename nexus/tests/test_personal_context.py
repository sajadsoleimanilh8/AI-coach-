from __future__ import annotations

from nexus.personal.context import build_personal_context_message
from nexus.personal.state import DimensionState, PersonalState
from nexus.personal.weakness import Weakness


def _empty_state() -> PersonalState:
    return PersonalState(user_id="u1", dimensions={}, computed_at=0.0)


def _state_with_dimension() -> PersonalState:
    return PersonalState(
        user_id="u1",
        dimensions={
            "physical.energy": DimensionState(
                dimension="physical.energy", value=0.42, confidence=0.8, sample_count=6, latest_at=1.0
            )
        },
        computed_at=0.0,
    )


def _weakness() -> Weakness:
    return Weakness(
        dimension="mental.stress",
        current=0.8,
        baseline=0.5,
        deviation=0.3,
        priority="HIGH",
        confidence=0.85,
        trend=None,
        explanation="mental.stress is currently 0.80 vs. a 90-day baseline of 0.50 (0.30 worse).",
    )


def test_returns_none_when_nothing_recorded_yet() -> None:
    message = build_personal_context_message(_empty_state(), [], {})
    assert message is None


def test_renders_current_state_dimensions() -> None:
    message = build_personal_context_message(_state_with_dimension(), [], {})

    assert message is not None
    assert message.role == "system"
    assert "physical.energy" in message.content
    assert "0.42" in message.content


def test_renders_weaknesses() -> None:
    message = build_personal_context_message(_empty_state(), [_weakness()], {})

    assert message is not None
    assert "mental.stress" in message.content
    assert "HIGH" in message.content


def test_renders_profile() -> None:
    message = build_personal_context_message(
        _empty_state(), [], {"primary_goal": "improve endurance"}
    )

    assert message is not None
    assert "primary_goal" in message.content
    assert "improve endurance" in message.content


def test_renders_all_three_sections_when_all_present() -> None:
    message = build_personal_context_message(
        _state_with_dimension(), [_weakness()], {"primary_goal": "improve endurance"}
    )

    assert message is not None
    assert "physical.energy" in message.content
    assert "mental.stress" in message.content
    assert "primary_goal" in message.content


def test_caps_dimensions_shown_at_max_dimensions() -> None:
    dimensions = {
        f"physical.{name}": DimensionState(
            dimension=f"physical.{name}", value=0.5, confidence=0.9 - i * 0.01,
            sample_count=5, latest_at=1.0,
        )
        for i, name in enumerate(
            ["energy", "recovery", "activity", "mobility", "strength", "endurance"]
        )
    }
    state = PersonalState(user_id="u1", dimensions=dimensions, computed_at=0.0)

    message = build_personal_context_message(state, [], {}, max_dimensions=2)

    assert message is not None
    shown = sum(1 for name in dimensions if name in message.content)
    assert shown == 2
