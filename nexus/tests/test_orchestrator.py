from __future__ import annotations

from typing import Any

import pytest

from nexus.agents.base import Agent, AgentContext
from nexus.agents.orchestrator import DelegationGuard, OrchestratorAgent
from nexus.agents.runtime import AgentRuntime
from nexus.core.providers import AIProvider
from nexus.core.router import RoutingDecision
from nexus.core.types import (
    GenerationChunk,
    GenerationResult,
    ModelInfo,
    RoutingPolicy,
    TaskType,
    ToolCall,
    Usage,
)
from nexus.tools.registry import ToolRegistry


class _SpecialistAgent(Agent):
    name = "research"
    description = "Specialist."
    allowed_tools: list[str] = []
    task_type = TaskType.RESEARCH

    def system_prompt(self, context: AgentContext) -> str:
        return "Answer the sub-goal."


class _ScriptedProvider(AIProvider):
    name = "fake"

    def __init__(self, script: list[GenerationResult]) -> None:
        self._script = list(script)
        self.call_count = 0

    async def generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        self.call_count += 1
        if self._script:
            return self._script.pop(0)
        return GenerationResult(
            content="Done.", model_used=model_id, provider_name=self.name, usage=Usage()
        )

    async def stream_generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        yield GenerationChunk(delta="", done=True, usage=Usage())

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> bool:
        return True

    def count_tokens(self, text: str, *, model_id: str) -> int:
        return len(text)


class _FakeRouter:
    def __init__(self, provider: AIProvider) -> None:
        self._provider = provider

    async def route_with_failover(self, **kwargs):
        decision = RoutingDecision(
            provider_name=self._provider.name, model_id="fake-model",
            task_type=kwargs.get("task_type", TaskType.GENERAL),
            policy=RoutingPolicy.BALANCED, reason="fake",
        )
        return decision, self._provider


def _delegate_call(agent_name: str, sub_goal: str, call_id: str = "1") -> GenerationResult:
    return GenerationResult(
        content="Delegating.", model_used="fake-model", provider_name="fake", usage=Usage(),
        tool_calls=[
            ToolCall(id=call_id, name="delegate", arguments={"agent_name": agent_name, "sub_goal": sub_goal})
        ],
    )


def _plain(content: str) -> GenerationResult:
    return GenerationResult(
        content=content, model_used="fake-model", provider_name="fake", usage=Usage()
    )


def _runtime(provider: AIProvider, **kwargs) -> AgentRuntime:
    defaults: dict[str, Any] = {
        "max_iterations": 6,
        "agent_factory": lambda name: _SpecialistAgent(),
        "enabled_agents": ["research", "coding", "orchestrator"],
    }
    defaults.update(kwargs)
    return AgentRuntime(_FakeRouter(provider), ToolRegistry({}), **defaults)


def _context(goal: str = "Do two things.") -> AgentContext:
    return AgentContext(
        goal=goal, user_id="u1", session_id=None,
        rag_service=None, long_term_memory=None, personal_state=None,
    )


# --- DelegationGuard: the caps themselves ---------------------------------


def test_guard_allows_delegation_within_every_cap() -> None:
    guard = DelegationGuard(max_depth=2, max_total_delegations=6)

    assert guard.try_acquire(agent_name="research", sub_goal="a", depth=1) is None
    assert guard.try_acquire(agent_name="coding", sub_goal="b", depth=1) is None


def test_guard_enforces_max_depth() -> None:
    guard = DelegationGuard(max_depth=2, max_total_delegations=6)

    assert guard.try_acquire(agent_name="research", sub_goal="a", depth=2) is None
    refusal = guard.try_acquire(agent_name="research", sub_goal="b", depth=3)

    assert refusal is not None
    assert "max_depth=2" in refusal
    assert "Synthesize" in refusal  # tells the orchestrator what to do next


def test_guard_enforces_max_total_delegations() -> None:
    guard = DelegationGuard(max_depth=3, max_total_delegations=2)

    assert guard.try_acquire(agent_name="a", sub_goal="1", depth=1) is None
    assert guard.try_acquire(agent_name="b", sub_goal="2", depth=1) is None
    refusal = guard.try_acquire(agent_name="c", sub_goal="3", depth=1)

    assert refusal is not None
    assert "all 2 delegations" in refusal


def test_guard_rejects_an_exact_repeat() -> None:
    guard = DelegationGuard()
    guard.try_acquire(agent_name="research", sub_goal="find the cause", depth=1)

    refusal = guard.try_acquire(agent_name="research", sub_goal="find the cause", depth=1)

    assert refusal is not None
    assert "already asked this exact sub-goal" in refusal


def test_guard_normalizes_case_and_whitespace_when_detecting_repeats() -> None:
    guard = DelegationGuard()
    guard.try_acquire(agent_name="research", sub_goal="Find The Cause", depth=1)

    assert guard.try_acquire(agent_name="research", sub_goal="  find the cause ", depth=1) is not None


def test_guard_allows_the_same_sub_goal_to_a_different_agent() -> None:
    guard = DelegationGuard()
    guard.try_acquire(agent_name="research", sub_goal="analyze this", depth=1)

    assert guard.try_acquire(agent_name="coding", sub_goal="analyze this", depth=1) is None


def test_a_refused_delegation_does_not_consume_budget() -> None:
    guard = DelegationGuard(max_total_delegations=3)
    guard.try_acquire(agent_name="research", sub_goal="a", depth=1)
    guard.try_acquire(agent_name="research", sub_goal="a", depth=1)  # refused repeat

    assert guard.total_delegations == 1


# --- The orchestrator running through the real runtime --------------------


@pytest.mark.asyncio
async def test_orchestrator_decomposes_and_delegates() -> None:
    provider = _ScriptedProvider(
        [
            _delegate_call("research", "Find the cause", "1"),
            _plain("The cause was a removed cache."),  # specialist's answer
            _delegate_call("coding", "Write the fix", "2"),
            _plain("Patch written."),
            _plain("Synthesized: the cache was removed; a patch restores it."),
        ]
    )
    runtime = _runtime(provider)

    result = await runtime.run(OrchestratorAgent(), _context())

    assert [s.agent_name for s in result.delegation_steps] == ["research", "coding"]
    assert result.delegation_steps[0].sub_goal == "Find the cause"
    assert result.delegation_steps[0].depth == 1
    assert "Synthesized" in result.final_answer


@pytest.mark.asyncio
async def test_delegation_results_are_summarized_into_the_step() -> None:
    provider = _ScriptedProvider(
        [_delegate_call("research", "Find it"), _plain("A specific finding."), _plain("Done.")]
    )

    result = await _runtime(provider).run(OrchestratorAgent(), _context())

    assert result.delegation_steps[0].result_summary == "A specific finding."


@pytest.mark.asyncio
async def test_runtime_enforces_the_total_cap_not_the_prompt() -> None:
    """Principle 6: the cap is a wall in the runtime, not a request in a
    prompt. A model that keeps calling delegate gets refused."""
    script: list[GenerationResult] = []
    for i in range(6):
        script.append(_delegate_call("research", f"sub-goal {i}", str(i)))
        script.append(_plain(f"answer {i}"))
    script.append(_plain("Synthesized from what I have."))

    provider = _ScriptedProvider(script)
    runtime = _runtime(provider, max_iterations=12, orchestration_max_total_delegations=2)

    result = await runtime.run(OrchestratorAgent(), _context())

    assert len(result.delegation_steps) == 2


@pytest.mark.asyncio
async def test_unknown_agent_is_refused_without_creating_a_step() -> None:
    def _factory(name: str):
        raise AssertionError(f"should never construct {name!r}")

    provider = _ScriptedProvider(
        [_delegate_call("nonexistent", "do a thing"), _plain("Synthesized.")]
    )
    runtime = _runtime(provider, agent_factory=_factory)

    result = await runtime.run(OrchestratorAgent(), _context())

    assert result.delegation_steps == []


@pytest.mark.asyncio
async def test_delegate_missing_arguments_is_refused() -> None:
    provider = _ScriptedProvider(
        [
            GenerationResult(
                content="", model_used="m", provider_name="fake", usage=Usage(),
                tool_calls=[ToolCall(id="1", name="delegate", arguments={"agent_name": "research"})],
            ),
            _plain("Synthesized."),
        ]
    )

    result = await _runtime(provider).run(OrchestratorAgent(), _context())

    assert result.delegation_steps == []


@pytest.mark.asyncio
async def test_an_orchestrator_cannot_delegate_to_itself() -> None:
    provider = _ScriptedProvider([_plain("Done.")])
    runtime = _runtime(provider)
    context = _context()

    await runtime.run(OrchestratorAgent(), context)

    assert "orchestrator" not in context.extra["delegatable_agents"]


@pytest.mark.asyncio
async def test_non_delegating_agents_get_the_shared_registry_untouched() -> None:
    provider = _ScriptedProvider([_plain("Answer.")])
    runtime = _runtime(provider)

    result = await runtime.run(_SpecialistAgent(), _context())

    # No delegate tool, no delegation steps, nothing changed for a plain agent.
    assert result.delegation_steps == []
    assert result.final_answer == "Answer."
