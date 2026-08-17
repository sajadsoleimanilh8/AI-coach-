from __future__ import annotations

from html.parser import HTMLParser
from typing import Any

import httpx

from nexus.tools.registry import Tool, ToolResult

_SEARCH_URL = "https://html.duckduckgo.com/html/"
_DEFAULT_TIMEOUT_SECONDS = 10.0


class _ResultLinkParser(HTMLParser):
    """Pulls {title, url} out of DuckDuckGo's HTML result markup
    (`<a class="result__a" href="...">title</a>`) — a minimal stdlib parse
    rather than a new BeautifulSoup dependency, since the structure needed
    here is this narrow.
    """

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._in_result_link = False
        self._current_title = ""
        self._current_href = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attrs_dict = dict(attrs)
        if "result__a" in (attrs_dict.get("class") or ""):
            self._in_result_link = True
            self._current_href = attrs_dict.get("href") or ""
            self._current_title = ""

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_result_link:
            self._in_result_link = False
            if self._current_title.strip():
                self.results.append(
                    {"title": self._current_title.strip(), "url": self._current_href}
                )

    def handle_data(self, data: str) -> None:
        if self._in_result_link:
            self._current_title += data


class WebSearchTool(Tool):
    """Queries DuckDuckGo's HTML endpoint — no API key required. This is
    an unofficial scrape of a public HTML page: DuckDuckGo doesn't publish
    a rate limit or ToS contract for it, so keep call volume low and treat
    parser breakage (if their markup changes) as expected maintenance, not
    """

    name = "web_search"
    description = "Search the web via DuckDuckGo and return the top result titles/links."
    parameters_schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The search query."}},
        "required": ["query"],
    }

    def __init__(
        self,
        *,
        max_results: int = 5,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._max_results = max_results
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        query = arguments.get("query") or ""
        if not query:
            return ToolResult(success=False, output="", error="Missing required argument: query")

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_seconds, transport=self._transport
            ) as client:
                response = await client.post(
                    _SEARCH_URL,
                    data={"q": query},
                    headers={"User-Agent": "Mozilla/5.0 (compatible; NexusBot/1.0)"},
                )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            return ToolResult(success=False, output="", error=f"Web search request failed: {exc}")

        parser = _ResultLinkParser()
        parser.feed(response.text)
        top_results = parser.results[: self._max_results]
        if not top_results:
            return ToolResult(success=True, output="No results found.")

        lines = [f"{i + 1}. {r['title']} — {r['url']}" for i, r in enumerate(top_results)]
        return ToolResult(success=True, output="\n".join(lines))
