from __future__ import annotations

from pathlib import Path
from typing import Any

from nexus.tools.registry import Tool, ToolResult

_DEFAULT_MAX_CHARS = 100_000


class FileSystemTool(Tool):
    """Read-only file access, resolved and confined to `allowed_root`."""

    name = "files"
    description = "Read a file's contents from within the allowed workspace root."
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the workspace root."}
        },
        "required": ["path"],
    }

    def __init__(self, *, allowed_root: str, max_chars: int = _DEFAULT_MAX_CHARS) -> None:
        self._allowed_root = Path(allowed_root).resolve()
        self._allowed_root.mkdir(parents=True, exist_ok=True)
        self._max_chars = max_chars

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        raw_path = arguments.get("path") or ""
        if not raw_path:
            return ToolResult(success=False, output="", error="Missing required argument: path")

        # An absolute raw_path makes `/` discard the left operand entirely
        # (pathlib joins to the absolute path), so the traversal guard must
        # be the resolved-path check below, not the join itself.
        candidate = (self._allowed_root / raw_path).resolve()
        if not candidate.is_relative_to(self._allowed_root):
            return ToolResult(
                success=False, output="", error="Path escapes the allowed workspace root."
            )
        if not candidate.exists() or not candidate.is_file():
            return ToolResult(success=False, output="", error=f"File not found: {raw_path}")

        try:
            content = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult(success=False, output="", error=str(exc))

        return ToolResult(success=True, output=content[: self._max_chars])
