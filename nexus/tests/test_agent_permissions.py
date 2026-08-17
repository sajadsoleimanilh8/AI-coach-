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
    parameters_schema = {"type": "object", "properties": {}}

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, output="5")


class _PythonTool(Tool):
    name = "python"
    description = "Executes code — deliberately NOT in the test agent's allowlist."
    parameters_schema = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self.executed = False

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        self.executed = True
        return ToolResult(success=True, output="ran")


class _ForcedPythonCallProvider(AIProvider):
    """Simulates a model that ignores instructions and asks for a tool
    outside its declared allowlist — the runtime must refuse it, not the
    model's own good behavior."""

    name = "fake"

    def __init__(self) -> None:
        self.call_count = 0
        self.received_tools_per_call: list[list[dict[str, Any]] | None] = []

    async def generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        self.call_count += 1
        self.received_tools_per_call.append(tools)
        if self.call_count == 1:
            return GenerationResult(
                content="",
                model_used=model_id,
                provider_name=self.name,
                usage=Usage(),
                tool_calls=[ToolCall(id="call_1", name="python", arguments={"code": "print(1)"})],
            )
        return GenerationResult(
            content="Done.", model_used=model_id, provider_name=self.name, usage=Usage()
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

    async def route_with_failover(self, **kwargs):
        decision = RoutingDecision(
            provider_name=self._provider.name,
            model_id="fake-model",
            task_type=kwargs.get("task_type", TaskType.GENERAL),
            policy=kwargs.get("policy") or RoutingPolicy.BALANCED,
            reason="fake routing",
        )
        return decision, self._provider


class _CalculatorOnlyAgent(Agent):
    name = "calculator-only"
    description = "Only allowed to use the calculator tool."
    allowed_tools = ["calculator"]
    task_type = TaskType.GENERAL

    def system_prompt(self, context: AgentContext) -> str:
        return "Use only the calculator tool."


@pytest.mark.asyncio
async def test_disallowed_tool_schema_is_never_shown_to_the_model() -> None:
    provider = _ForcedPythonCallProvider()
    router = _FakeRouter(provider)
    tool_registry = ToolRegistry({"calculator": _CalculatorTool(), "python": _PythonTool()})
    runtime = AgentRuntime(router, tool_registry, max_iterations=5)

    context = AgentContext(
        goal="Compute something",
        user_id="u1",
        session_id=None,
        rag_service=None,
        long_term_memory=None,
        personal_state=None,
    )
    await runtime.run(_CalculatorOnlyAgent(), context)

    first_call_schema_names = {s["name"] for s in provider.received_tools_per_call[0]}
    assert first_call_schema_names == {"calculator"}
    assert "python" not in first_call_schema_names


@pytest.mark.asyncio
async def test_forced_disallowed_tool_call_is_refused_not_executed() -> None:
    provider = _ForcedPythonCallProvider()
    python_tool = _PythonTool()
    router = _FakeRouter(provider)
    tool_registry = ToolRegistry({"calculator": _CalculatorTool(), "python": python_tool})
    runtime = AgentRuntime(router, tool_registry, max_iterations=5)

    context = AgentContext(
        goal="Compute something",
        user_id="u1",
        session_id=None,
        rag_service=None,
        long_term_memory=None,
        personal_state=None,
    )
    result = await runtime.run(_CalculatorOnlyAgent(), context)

    assert python_tool.executed is False
    assert len(result.steps) == 1
    refused_step = result.steps[0]
    assert refused_step.tool_name == "python"
    assert "not permitted" in refused_step.tool_output
    assert result.completed is True
    assert result.final_answer == "Done."
