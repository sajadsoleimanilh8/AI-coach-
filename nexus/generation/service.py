from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nexus.core.router import ModelRouter
from nexus.core.types import Message, TaskType, Usage
from nexus.generation.planner import BriefBuilder, GenerationBrief
from nexus.health.safety import check_output

_SYSTEM_PROMPT = (
    "You generate a personalized plan from a structured brief provided below. "
    "The brief's target_intensity, weaknesses, and health_patterns were computed "
    "deterministically — you do not choose or second-guess them. Every element of "
    "the plan you write must be justified against something specific in the "
    "brief (a weakness, a health pattern, the stated intensity, or a stated "
    "constraint) — do not add elements the brief gives you no basis for. If the "
    "brief says this is a generic (non-personalized) plan, say so plainly in "
    "your response rather than implying you know things about the user you "
    "don't."
)

# request_type -> TaskType. "study"/"daily_plan" aren't fitness-specific, so
# they route as general planning rather than forcing a FITNESS-capability
# model on a request that isn't about physical training.
_TASK_TYPE_BY_REQUEST: dict[str, TaskType] = {
    "workout": TaskType.FITNESS,
    "recovery": TaskType.FITNESS,
    "nutrition": TaskType.FITNESS,
    "study": TaskType.PLANNING,
    "daily_plan": TaskType.PLANNING,
}

# Training advice layered on top of a concerning health pattern is exactly
# where unsafe text (e.g. "push through the pain") could appear — gate
# those two request types through the same output filter HealthAgent uses.
_SAFETY_GATED_REQUEST_TYPES = frozenset({"workout", "recovery"})


@dataclass
class PersonalizedPlan:
    request_type: str
    content: str
    brief_rationale: list[str]
    addressed_weaknesses: list[str]
    target_intensity: float
    personalized: bool
    model_used: str
    usage: Usage


def _serialize_brief(brief: GenerationBrief) -> str:
    state_lines = (
        "\n".join(
            f"- {dim}: {ds.value:.2f} (confidence={ds.confidence:.0%}, n={ds.sample_count})"
            for dim, ds in brief.state.dimensions.items()
        )
        or "None recorded."
    )
    weakness_lines = (
        "\n".join(f"- [{w.priority}] {w.explanation}" for w in brief.weaknesses) or "None."
    )
    pattern_lines = (
        "\n".join(f"- [{p.severity}] {p.name}: {p.explanation}" for p in brief.health_patterns)
        or "None."
    )
    constraint_lines = (
        "\n".join(f"- {k}: {v}" for k, v in brief.constraints.items()) or "None stated."
    )
    profile_lines = "\n".join(f"- {k}: {v}" for k, v in brief.profile.items()) or "None stated."
    rationale_lines = "\n".join(f"- {line}" for line in brief.rationale)

    return (
        f"Request type: {brief.request_type}\n"
        f"Target intensity: {brief.target_intensity:.2f} (0..1)\n\n"
        f"Intensity rationale:\n{rationale_lines}\n\n"
        f"Current state:\n{state_lines}\n\n"
        f"Weaknesses:\n{weakness_lines}\n\n"
        f"Health patterns:\n{pattern_lines}\n\n"
        f"Constraints:\n{constraint_lines}\n\n"
        f"Stated goals/preferences:\n{profile_lines}"
    )


class PersonalizedGenerator:
    """Renders a GenerationBrief into a plan via the LLM. The model
    receives the brief as structured facts and must justify each element
    against it — it never chooses the intensity or invents the user's
    state (principle 1)."""

    def __init__(self, router: ModelRouter, brief_builder: BriefBuilder) -> None:
        self._router = router
        self._brief_builder = brief_builder

    async def generate(
        self, *, user_id: str, request_type: str, constraints: dict[str, Any]
    ) -> PersonalizedPlan:
        brief = await self._brief_builder.build(
            user_id=user_id, request_type=request_type, constraints=constraints
        )
        personalized = bool(brief.state.dimensions)

        task_type = _TASK_TYPE_BY_REQUEST.get(request_type, TaskType.PLANNING)
        decision, provider = await self._router.route_with_failover(task_type=task_type)

        result = await provider.generate(
            [
                Message(role="system", content=_SYSTEM_PROMPT),
                Message(role="user", content=_serialize_brief(brief)),
            ],
            model_id=decision.model_id,
            temperature=0.7,
        )

        content = result.content
        has_significant_pattern = any(p.severity == "significant" for p in brief.health_patterns)
        if request_type in _SAFETY_GATED_REQUEST_TYPES and has_significant_pattern:
            verdict = check_output(content)
            if not verdict.allowed:
                content = verdict.rewritten_text or content

        return PersonalizedPlan(
            request_type=request_type,
            content=content,
            brief_rationale=brief.rationale,
            addressed_weaknesses=[w.dimension for w in brief.weaknesses],
            target_intensity=brief.target_intensity,
            personalized=personalized,
            model_used=result.model_used,
            usage=result.usage,
        )
