from __future__ import annotations

from typing import Any

from nexus.core.types import Message
from nexus.personal.state import PersonalState
from nexus.personal.weakness import Weakness

_MAX_WEAKNESSES_SHOWN = 5


def build_personal_context_message(
    state: PersonalState,
    weaknesses: list[Weakness],
    profile: dict[str, Any],
    *,
    max_dimensions: int = 8,
) -> Message | None:
    """Compact system-message summary of who the user is right now, for
    injection into a chat prompt. Returns None rather than an empty/
    all-defaults block when there is genuinely nothing recorded yet — an
    empty personal-context message would just be noise in the prompt."""
    if not state.dimensions and not weaknesses and not profile:
        return None

    sections: list[str] = ["User's current personal context (confidence-weighted from recorded signals):"]

    if state.dimensions:
        ranked = sorted(
            state.dimensions.values(), key=lambda d: d.confidence, reverse=True
        )[:max_dimensions]
        lines = [
            f"  - {d.dimension}: {d.value:.2f} "
            f"(confidence {d.confidence:.0%}, {d.sample_count} sample(s))"
            for d in ranked
        ]
        sections.append("Current state:\n" + "\n".join(lines))

    if weaknesses:
        top = weaknesses[:_MAX_WEAKNESSES_SHOWN]
        lines = [f"  - [{w.priority}] {w.explanation}" for w in top]
        sections.append("Notable deviations from baseline:\n" + "\n".join(lines))

    if profile:
        lines = [f"  - {key}: {value}" for key, value in profile.items()]
        sections.append("Stated goals/preferences/constraints:\n" + "\n".join(lines))

    return Message(role="system", content="\n\n".join(sections))
