from __future__ import annotations

from typing import Any

import pytest

from nexus.agents.base import Agent, AgentContext
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
from nexus.tools.registry import Tool, ToolRegistry, ToolResult


class _CalculatorTool(Tool):
    name = "calculator"
    description = "Adds two numbers."
    parameters_schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, output=str(arguments["a"] + arguments["b"]))


class _FakeProvider(AIProvider):
    name = "fake"

    def __init__(self, *, tool_call_rounds: int) -> None:
        self._tool_call_rounds = tool_call_rounds
        self.call_count = 0

    async def generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        self.call_count += 1
        if self.call_count <= self._tool_call_rounds:
            return GenerationResult(
                content="Let me add those.",
                model_used=model_id,
                provider_name=self.name,
                usage=Usage(prompt_tokens=10, completion_tokens=5),
                tool_calls=[
                    ToolCall(id=f"call_{self.call_count}", name="calculator", arguments={"a": 2, "b": 3})
                ],
            )
        return GenerationResult(
            content="The answer is 5.",
            model_used=model_id,
            provider_name=self.name,
            usage=Usage(prompt_tokens=20, completion_tokens=8),
        )

    async def stream_generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        yield GenerationChunk(delta="unused", done=True, usage=Usage())

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> bool:
        return True

    def count_tokens(self, text: str, *, model_id: str) -> int:
        return len(text)


class _FakeRouter:
    def __init__(self, provider: AIProvider) -> None:
        self._provider = provider
        self.received_kwargs: dict[str, Any] | None = None

    async def route_with_failover(self, **kwargs):
        self.received_kwargs = kwargs
        decision = RoutingDecision(
            provider_name=self._provider.name,
            model_id="fake-model",
            task_type=kwargs.get("task_type", TaskType.GENERAL),
            policy=kwargs.get("policy") or RoutingPolicy.BALANCED,
            reason="fake routing",
        )
        return decision, self._provider


class _TestAgent(Agent):
    name = "test-agent"
    description = "A minimal agent for runtime tests."
    allowed_tools = ["calculator"]
    task_type = TaskType.GENERAL

    def system_prompt(self, context: AgentContext) -> str:
        return "Solve the goal using the calculator tool if needed."


def _context(goal: str = "What is 2 + 3?", **extra: Any) -> AgentContext:
    return AgentContext(
        goal=goal,
        user_id="u1",
        session_id=None,
        rag_service=None,
        long_term_memory=None,
        personal_state=None,
        extra=extra,
    )


@pytest.mark.asyncio
async def test_run_completes_after_one_tool_call_then_a_final_answer() -> None:
    provider = _FakeProvider(tool_call_rounds=1)
    router = _FakeRouter(provider)
    tool_registry = ToolRegistry({"calculator": _CalculatorTool()})
    runtime = AgentRuntime(router, tool_registry, max_iterations=5)

    result = await runtime.run(_TestAgent(), _context())

    assert result.final_answer == "The answer is 5."
    assert result.completed is True
    assert result.iterations_used == 2
    assert len(result.steps) == 1
    step = result.steps[0]
    assert step.index == 0
    assert step.tool_name == "calculator"
    assert step.tool_arguments == {"a": 2, "b": 3}
    assert step.tool_output == "5"
    assert step.thought == "Let me add those."
    assert result.usage.prompt_tokens == 30
    assert result.usage.completion_tokens == 13
    assert result.model_used == "fake-model"
    assert result.provider_name == "fake"

    assert router.received_kwargs["require_tool_calling"] is True
    assert router.received_kwargs["task_type"] == TaskType.GENERAL


@pytest.mark.asyncio
async def test_run_reports_incomplete_when_the_iteration_cap_is_hit() -> None:
    provider = _FakeProvider(tool_call_rounds=10)
    router = _FakeRouter(provider)
    tool_registry = ToolRegistry({"calculator": _CalculatorTool()})
    runtime = AgentRuntime(router, tool_registry, max_iterations=3)

    result = await runtime.run(_TestAgent(), _context())

    assert result.completed is False
    assert result.iterations_used == 3
    assert len(result.steps) == 3


@pytest.mark.asyncio
async def test_per_run_max_iterations_override_via_context_extra() -> None:
    provider = _FakeProvider(tool_call_rounds=10)
    router = _FakeRouter(provider)
    tool_registry = ToolRegistry({"calculator": _CalculatorTool()})
    runtime = AgentRuntime(router, tool_registry, max_iterations=8)

    result = await runtime.run(_TestAgent(), _context(max_iterations=2))

    assert result.iterations_used == 2
    assert result.completed is False
