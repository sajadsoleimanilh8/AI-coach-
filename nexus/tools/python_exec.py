from __future__ import annotations

import asyncio
import subprocess
import sys
from typing import Any

from nexus.tools.registry import Tool, ToolResult

_DEFAULT_TIMEOUT_SECONDS = 10


class PythonExecutionTool(Tool):
    """Runs a short Python snippet in a subprocess.

    SECURITY: this is NOT a sandbox. `-I` (isolated mode) only ignores the
    environment and user site-packages; the child process still has the full
    filesystem, the network, and the service account's privileges. The code it
    runs is supplied by an LLM, which in turn takes instruction from whatever
    arrived at /api/chat.

    Because of that this tool is disabled by default (see
    nexus/config/nexus.yaml) and only registers when an operator sets
    NEXUS_ENABLE_PYTHON_TOOL=1. Enable it only inside a container boundary you
    are prepared to have compromised, never on a host with credentials or data
    you care about.
    """

    name = "python"
    description = "Execute a short Python code snippet and return its stdout."
    parameters_schema = {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python source code to run."}},
        "required": ["code"],
    }

    def __init__(self, *, timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS) -> None:
        self._timeout_seconds = timeout_seconds

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        code = arguments.get("code") or ""
        if not code:
            return ToolResult(success=False, output="", error="Missing required argument: code")

        try:
            # subprocess.run() is blocking — run it off the event loop so
            # one slow tool call doesn't stall every other in-flight request.
            completed = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, "-I", "-c", code],
                timeout=self._timeout_seconds,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Execution timed out after {self._timeout_seconds}s",
            )

        if completed.returncode != 0:
            return ToolResult(
                success=False,
                output=completed.stdout,
                error=completed.stderr.strip() or f"Exited with code {completed.returncode}",
            )
        return ToolResult(success=True, output=completed.stdout)
