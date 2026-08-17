from __future__ import annotations

from typing import Any

import pytest

from nexus.core.providers import AIProvider
from nexus.core.tool_loop import run_tool_loop
from nexus.core.types import GenerationChunk, GenerationResult, Message, ModelInfo, ToolCall, Usage
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


class _PythonTool(Tool):
    name = "python"
    description = "Would execute code."
    parameters_schema = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self.executed = False

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        self.executed = True
        return ToolResult(success=True, output="ran")


class _FakeProvider(AIProvider):
    """Returns a tool_call for `tool_call_rounds` generate() calls, then a
    final (non-tool-call) answer — or never stops if tool_call_rounds is
    >= max_iterations, to exercise the cap."""

    name = "fake"

    def __init__(self, *, tool_call_rounds: int, tool_name: str = "calculator") -> None:
        self._tool_call_rounds = tool_call_rounds
        self._tool_name = tool_name
        self.call_count = 0
        self.received_tools_per_call: list[list[dict[str, Any]] | None] = []

    async def generate(
        self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None
    ) -> GenerationResult:
        self.call_count += 1
        self.received_tools_per_call.append(tools)
        if self.call_count <= self._tool_call_rounds:
            return GenerationResult(
                content="calling the tool",
                model_used=model_id,
                provider_name=self.name,
                usage=Usage(prompt_tokens=10, completion_tokens=5),
                tool_calls=[
                    ToolCall(id=f"call_{self.call_count}", name=self._tool_name, arguments={"a": 2, "b": 3})
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


def _registry(**tools: Tool) -> ToolRegistry:
    return ToolRegistry(tools)


@pytest.mark.asyncio
async def test_terminates_on_no_tool_calls() -> None:
    provider = _FakeProvider(tool_call_rounds=0)

    outcome = await run_tool_loop(
        provider,
        [Message(role="user", content="hi")],
        model_id="m",
        temperature=0.7,
        max_tokens=None,
        tool_registry=_registry(calculator=_CalculatorTool()),
        max_iterations=5,
    )

    assert outcome.result.content == "The answer is 5."
    assert outcome.tool_calls_made == []
    assert outcome.hit_iteration_cap is False
    assert outcome.iterations_used == 1
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_respects_max_iterations() -> None:
    provider = _FakeProvider(tool_call_rounds=10)

    outcome = await run_tool_loop(
        provider,
        [Message(role="user", content="hi")],
        model_id="m",
        temperature=0.7,
        max_tokens=None,
        tool_registry=_registry(calculator=_CalculatorTool()),
        max_iterations=3,
    )

    assert provider.call_count == 3
    assert outcome.iterations_used == 3
    assert len(outcome.tool_calls_made) == 3
    assert outcome.hit_iteration_cap is True


@pytest.mark.asyncio
async def test_accumulates_usage_across_iterations() -> None:
    provider = _FakeProvider(tool_call_rounds=1)

    outcome = await run_tool_loop(
        provider,
        [Message(role="user", content="hi")],
        model_id="m",
        temperature=0.7,
        max_tokens=None,
        tool_registry=_registry(calculator=_CalculatorTool()),
        max_iterations=5,
    )

    assert outcome.result.usage.prompt_tokens == 30
    assert outcome.result.usage.completion_tokens == 13


@pytest.mark.asyncio
async def test_allowed_tools_none_exposes_every_registered_tool() -> None:
    provider = _FakeProvider(tool_call_rounds=0)

    await run_tool_loop(
        provider,
        [Message(role="user", content="hi")],
        model_id="m",
        temperature=0.7,
        max_tokens=None,
        tool_registry=_registry(calculator=_CalculatorTool(), python=_PythonTool()),
        max_iterations=5,
        allowed_tools=None,
    )

    schema_names = {s["name"] for s in provider.received_tools_per_call[0]}
    assert schema_names == {"calculator", "python"}


@pytest.mark.asyncio
async def test_allowed_tools_filters_schemas_shown_to_the_model() -> None:
    provider = _FakeProvider(tool_call_rounds=0)

    await run_tool_loop(
        provider,
        [Message(role="user", content="hi")],
        model_id="m",
        temperature=0.7,
        max_tokens=None,
        tool_registry=_registry(calculator=_CalculatorTool(), python=_PythonTool()),
        max_iterations=5,
        allowed_tools=["calculator"],
    )

    schema_names = {s["name"] for s in provider.received_tools_per_call[0]}
    assert schema_names == {"calculator"}


@pytest.mark.asyncio
async def test_rejects_a_tool_call_outside_allowed_tools_without_executing_it() -> None:
    python_tool = _PythonTool()
    provider = _FakeProvider(tool_call_rounds=1, tool_name="python")

    outcome = await run_tool_loop(
        provider,
        [Message(role="user", content="hi")],
        model_id="m",
        temperature=0.7,
        max_tokens=None,
        tool_registry=_registry(calculator=_CalculatorTool(), python=python_tool),
        max_iterations=5,
        allowed_tools=["calculator"],
    )

    assert python_tool.executed is False
    assert len(outcome.tool_calls_made) == 1
    refused = outcome.tool_calls_made[0]
    assert refused.name == "python"
    assert "not permitted" in refused.result_summary
    assert outcome.result.content == "The answer is 5."


@pytest.mark.asyncio
async def test_tool_calls_made_carry_the_accompanying_thought_and_a_timestamp() -> None:
    provider = _FakeProvider(tool_call_rounds=1)

    outcome = await run_tool_loop(
        provider,
        [Message(role="user", content="hi")],
        model_id="m",
        temperature=0.7,
        max_tokens=None,
        tool_registry=_registry(calculator=_CalculatorTool()),
        max_iterations=5,
    )

    call = outcome.tool_calls_made[0]
    assert call.thought == "calling the tool"
    assert call.timestamp > 0
