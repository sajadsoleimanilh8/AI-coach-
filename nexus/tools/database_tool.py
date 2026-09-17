from __future__ import annotations

import re
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from nexus.tools.registry import Tool, ToolResult

_DEFAULT_MAX_ROWS = 50
_SELECT_RE = re.compile(r"^\s*select\b", re.IGNORECASE)


class DatabaseTool(Tool):
    """Read-only SELECT access to NEXUS's own SQLite database — e.g. to
    answer "how much have I spent this session" against cost_records.
    """

    name = "database"
    description = (
        "Run a read-only SELECT query against NEXUS's own database "
        "(tables include cost_records, latency_records, sessions, "
        "messages) and return up to 50 rows."
    )
    parameters_schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "A single SELECT statement."}},
        "required": ["query"],
    }

    def __init__(self, engine: AsyncEngine, *, max_rows: int = _DEFAULT_MAX_ROWS) -> None:
        self._engine = engine
        self._max_rows = max_rows

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        query = (arguments.get("query") or "").strip()
        if query.endswith(";"):
            query = query[:-1].strip()  # tolerate one harmless trailing semicolon
        if not query:
            return ToolResult(success=False, output="", error="Missing required argument: query")
        if not _SELECT_RE.match(query):
            return ToolResult(success=False, output="", error="Only SELECT statements are allowed.")
        if ";" in query:
            return ToolResult(
                success=False, output="", error="Statement chaining (';') is not allowed."
            )

        try:
            async with self._engine.connect() as conn:
                result = await conn.execute(text(query))
                rows = result.fetchmany(self._max_rows)
                columns = list(result.keys())
        except Exception as exc:  # noqa: BLE001 - surface a bad query as a tool failure, not a crash
            return ToolResult(success=False, output="", error=str(exc))

        if not rows:
            return ToolResult(success=True, output="No rows returned.")

        lines = [", ".join(columns)]
        lines.extend(", ".join(str(value) for value in row) for row in rows)
        return ToolResult(success=True, output="\n".join(lines))
