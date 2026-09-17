from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from nexus.core.exceptions import ToolNotFoundError
from nexus.logging_setup.logger import get_logger

logger = get_logger("tools.registry")


@dataclass
class ToolResult:
    success: bool
    output: str
    error: str | None = None


class Tool(ABC):
    name: str
    description: str
    parameters_schema: dict[str, Any]  # JSON schema for the "parameters" field

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Run the tool with the given (already-parsed) arguments."""


class ToolRegistry:
    def __init__(self, tools: dict[str, Tool]) -> None:
        self._tools = tools

    def tools(self) -> dict[str, Tool]:
        """Copy of the registered tools, so callers that need to build an
        AUGMENTED registry (AgentRuntime adding a per-run `delegate` tool)
        can do so without mutating this one or reaching into _tools."""
        return dict(self._tools)

    def enabled_tool_schemas(self) -> list[dict[str, Any]]:
        """Canonical {"name","description","parameters"} dicts for every
        registered tool, ready to pass to AIProvider.generate(tools=...)."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_schema,
            }
            for tool in self._tools.values()
        ]

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(f"No tool registered under name={name!r}.")
        try:
            return await tool.execute(arguments)
        except Exception as exc:  # noqa: BLE001 - one bad tool call must never crash the request
            logger.warning("Tool %s raised during execute(): %s", name, exc)
            return ToolResult(success=False, output="", error=str(exc))
