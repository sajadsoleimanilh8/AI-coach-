from __future__ import annotations

from nexus.agents.base import Agent, AgentContext
from nexus.core.types import Message, TaskType
from nexus.personal.context import build_personal_context_message


class PlanningAgent(Agent):
    name = "planning"
    description = "Decomposes goals into sequenced, capacity-aware plans."
    allowed_tools = ["files"]
    task_type = TaskType.PLANNING

    def system_prompt(self, context: AgentContext) -> str:
        return (
            "You are a planning agent. Decompose the goal into concrete steps, "
            "sequenced by dependency (what has to happen before what). Give "
            "realistic time/effort estimates rather than optimistic ones. If personal "
            "context about the user's current state is provided below, explicitly "
            "adapt the plan's intensity and pacing to it — do not propose a demanding "
            "plan for someone whose current state shows low energy, poor recovery, or "
            "high stress/fatigue without accounting for that."
        )

    async def prepare_context(self, context: AgentContext) -> list[Message]:
        state = await context.personal_state.get_state(context.user_id)

        weakness_engine = context.extra.get("weakness_engine")
        weaknesses = await weakness_engine.detect(context.user_id) if weakness_engine else []

        profile_store = context.extra.get("profile_store")
        profile = await profile_store.get_profile(context.user_id) if profile_store else {}

        message = build_personal_context_message(state, weaknesses, profile)
        return [message] if message is not None else []
