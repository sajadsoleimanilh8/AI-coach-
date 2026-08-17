from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from nexus.agents.base import Agent, AgentContext
from nexus.core.types import TaskType
from nexus.logging_setup.logger import get_logger
from nexus.tools.registry import Tool, ToolResult

logger = get_logger("agents.orchestrator")

DELEGATE_TOOL_NAME = "delegate"

_DEFAULT_MAX_DEPTH = 2
_DEFAULT_MAX_TOTAL_DELEGATIONS = 6
_SUMMARY_CHAR_LIMIT = 600


@dataclass
class DelegationGuard:
    """Hard caps on delegation, enforced in the runtime rather than asked
    for in a prompt (principle 6).
    """

    max_depth: int = _DEFAULT_MAX_DEPTH
    max_total_delegations: int = _DEFAULT_MAX_TOTAL_DELEGATIONS
    total_delegations: int = 0
    seen: set[tuple[str, str]] = field(default_factory=set)

    def try_acquire(self, *, agent_name: str, sub_goal: str, depth: int) -> str | None:
        """Returns None when the delegation may proceed, or a refusal
        string explaining which cap was hit. The refusal text goes back to
        the orchestrator as a tool result, so it learns it has run out of
        budget and must synthesize from what it already has."""
        if depth > self.max_depth:
            return (
                f"Delegation refused: depth {depth} exceeds max_depth={self.max_depth}. "
                f"Synthesize an answer from the results you already have."
            )
        if self.total_delegations >= self.max_total_delegations:
            return (
                f"Delegation refused: this run has already used all "
                f"{self.max_total_delegations} delegations. Synthesize an answer from the "
                f"results you already have."
            )

        key = (agent_name, sub_goal.strip().lower())
        if key in self.seen:
            return (
                f"Delegation refused: {agent_name!r} was already asked this exact sub-goal in "
                f"this run. Re-running it would return the same result — use the result you "
                f"already have, or ask something different."
            )

        self.seen.add(key)
        self.total_delegations += 1
        return None


@dataclass
class DelegationStep:
    agent_name: str
    sub_goal: str
    result_summary: str
    depth: int
    usage: Any = None


class DelegateTool(Tool):
    """The `delegate` pseudo-tool. Exposed through the EXISTING
    ToolRegistry so delegation flows through the same permission checks,
    tool-call loop, and step tracing as every other tool — there is no
    parallel control path for multi-agent work.
    """

    name = DELEGATE_TOOL_NAME
    description = (
        "Delegate a self-contained sub-goal to a specialist agent and get its answer back. "
        "Use one call per sub-goal. Available agents are listed in your instructions."
    )
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "agent_name": {
                "type": "string",
                "description": "Which specialist agent should handle this sub-goal.",
            },
            "sub_goal": {
                "type": "string",
                "description": "A single, self-contained sub-goal stated in full.",
            },
        },
        "required": ["agent_name", "sub_goal"],
    }

    def __init__(
        self,
        *,
        guard: DelegationGuard,
        depth: int,
        run_sub_agent: Callable[[str, str, int], Awaitable[tuple[str, Any]]],
        available_agents: list[str],
    ) -> None:
        self._guard = guard
        self._depth = depth
        self._run_sub_agent = run_sub_agent
        self._available_agents = available_agents
        self.steps: list[DelegationStep] = []

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        agent_name = str(arguments.get("agent_name", "")).strip()
        sub_goal = str(arguments.get("sub_goal", "")).strip()

        if not agent_name or not sub_goal:
            return ToolResult(
                success=False,
                output="",
                error="delegate requires both 'agent_name' and 'sub_goal'.",
            )

        if agent_name not in self._available_agents:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Unknown agent {agent_name!r}. Available: "
                    f"{', '.join(sorted(self._available_agents))}."
                ),
            )

        child_depth = self._depth + 1
        refusal = self._guard.try_acquire(
            agent_name=agent_name, sub_goal=sub_goal, depth=child_depth
        )
        if refusal is not None:
            logger.info("delegation refused: %s", refusal)
            return ToolResult(success=False, output="", error=refusal)

        answer, usage = await self._run_sub_agent(agent_name, sub_goal, child_depth)
        summary = answer if len(answer) <= _SUMMARY_CHAR_LIMIT else answer[:_SUMMARY_CHAR_LIMIT] + "..."
        self.steps.append(
            DelegationStep(
                agent_name=agent_name,
                sub_goal=sub_goal,
                result_summary=summary,
                depth=child_depth,
                usage=usage,
            )
        )
        return ToolResult(success=True, output=answer)


class OrchestratorAgent(Agent):
    name = "orchestrator"
    description = (
        "Decomposes a multi-part goal into sub-goals, delegates each to the best-suited "
        "specialist agent, and synthesizes the results into one answer."
    )
    allowed_tools = [DELEGATE_TOOL_NAME]
    task_type = TaskType.AGENT_EXECUTION

    def system_prompt(self, context: AgentContext) -> str:
        available = context.extra.get("delegatable_agents", [])
        roster = ", ".join(sorted(available)) if available else "none available"
        return (
            "You are an orchestrator. Break the goal into the smallest set of self-contained "
            "sub-goals that together answer it, then delegate each one to the specialist best "
            f"suited to it using the `delegate` tool. Available specialists: {roster}.\n\n"
            "Rules:\n"
            "- One `delegate` call per sub-goal, each stated in full — a specialist sees only "
            "the sub-goal you send, not the original goal or the other results.\n"
            "- Do not delegate a sub-goal you can answer directly.\n"
            "- Never repeat an identical sub-goal to the same agent; you will get the same "
            "answer back.\n"
            "- Delegation is capped. If a delegate call comes back refused, that budget is "
            "gone — synthesize the best answer you can from what you already have rather than "
            "retrying.\n"
            "- Finish by synthesizing the specialists' results into one coherent answer that "
            "addresses the original goal, noting anything that stayed unresolved."
        )
