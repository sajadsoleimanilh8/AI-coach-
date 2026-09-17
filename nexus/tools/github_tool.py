from __future__ import annotations

import base64
from typing import Any

import httpx

from nexus.tools.registry import Tool, ToolResult

_API_BASE = "https://api.github.com"
_DEFAULT_TIMEOUT_SECONDS = 15.0
_MAX_SEARCH_RESULTS = 10


class GitHubTool(Tool):
    """Read-only GitHub REST API access. Only constructed/registered by
    main.py when GITHUB_TOKEN is configured — cleanly absent otherwise.
    """

    name = "github"
    description = "Read-only GitHub access: get a file's contents or search code."
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["get_file_contents", "search_code"]},
            "repo": {
                "type": "string",
                "description": "owner/repo — required for get_file_contents",
            },
            "path": {"type": "string", "description": "file path — required for get_file_contents"},
            "query": {"type": "string", "description": "search query — required for search_code"},
        },
        "required": ["action"],
    }

    def __init__(
        self,
        token: str,
        *,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._token = token
        self._timeout_seconds = timeout_seconds
        self._transport = transport  # test seam: inject httpx.MockTransport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=_API_BASE,
            timeout=self._timeout_seconds,
            transport=self._transport,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/vnd.github+json",
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        action = arguments.get("action")
        try:
            if action == "get_file_contents":
                return await self._get_file_contents(arguments)
            if action == "search_code":
                return await self._search_code(arguments)
            return ToolResult(success=False, output="", error=f"Unknown action: {action!r}")
        except httpx.HTTPError as exc:
            return ToolResult(success=False, output="", error=f"GitHub API request failed: {exc}")

    async def _get_file_contents(self, arguments: dict[str, Any]) -> ToolResult:
        repo = arguments.get("repo")
        path = arguments.get("path")
        if not repo or not path:
            return ToolResult(
                success=False, output="", error="get_file_contents requires 'repo' and 'path'."
            )

        async with self._client() as client:
            response = await client.get(f"/repos/{repo}/contents/{path}")
        if response.status_code == 404:
            return ToolResult(success=False, output="", error=f"File not found: {repo}/{path}")
        response.raise_for_status()
        data = response.json()
        content = base64.b64decode(data.get("content", "")).decode("utf-8", errors="replace")
        return ToolResult(success=True, output=content)

    async def _search_code(self, arguments: dict[str, Any]) -> ToolResult:
        query = arguments.get("query")
        if not query:
            return ToolResult(success=False, output="", error="search_code requires 'query'.")

        async with self._client() as client:
            response = await client.get("/search/code", params={"q": query})
        response.raise_for_status()
        data = response.json()
        items = data.get("items", [])[:_MAX_SEARCH_RESULTS]
        if not items:
            return ToolResult(success=True, output="No results found.")

        lines = [
            f"{item.get('repository', {}).get('full_name', '?')}: {item.get('path', '?')}"
            for item in items
        ]
        return ToolResult(success=True, output="\n".join(lines))
