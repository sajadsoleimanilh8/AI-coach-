from __future__ import annotations

from typing import Any

import pytest

from nexus.core.exceptions import ToolNotFoundError
from nexus.tools.registry import Tool, ToolRegistry, ToolResult


class _EchoTool(Tool):
    name = "echo"
    description = "Echoes back the given text."
    parameters_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, output=arguments["text"])


class _BrokenTool(Tool):
    name = "broken"
    description = "Always raises."
    parameters_schema = {"type": "object", "properties": {}}

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        raise RuntimeError("kaboom")


def _registry() -> ToolRegistry:
    return ToolRegistry({"echo": _EchoTool(), "broken": _BrokenTool()})


def test_enabled_tool_schemas_shape() -> None:
    schemas = _registry().enabled_tool_schemas()
    schemas_by_name = {s["name"]: s for s in schemas}

    assert set(schemas_by_name) == {"echo", "broken"}
    assert schemas_by_name["echo"]["description"] == "Echoes back the given text."
    assert schemas_by_name["echo"]["parameters"] == _EchoTool.parameters_schema


@pytest.mark.asyncio
async def test_execute_success() -> None:
    result = await _registry().execute("echo", {"text": "hello"})

    assert result.success is True
    assert result.output == "hello"
    assert result.error is None


@pytest.mark.asyncio
async def test_execute_wraps_a_raised_exception_into_a_failed_result() -> None:
    result = await _registry().execute("broken", {})

    assert result.success is False
    assert result.output == ""
    assert "kaboom" in result.error


@pytest.mark.asyncio
async def test_execute_unknown_tool_raises_tool_not_found_error() -> None:
    with pytest.raises(ToolNotFoundError):
        await _registry().execute("does-not-exist", {})


def test_empty_registry_has_no_schemas() -> None:
    assert ToolRegistry({}).enabled_tool_schemas() == []
