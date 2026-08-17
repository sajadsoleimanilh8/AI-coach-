from __future__ import annotations

import httpx
import pytest

from nexus.tools.web_search import WebSearchTool

_SAMPLE_HTML = """
<html><body>
<div class="result">
  <a class="result__a" href="https://example.com/one">First Result Title</a>
</div>
<div class="result">
  <a class="result__a" href="https://example.com/two">Second Result Title</a>
</div>
</body></html>
"""


def _tool(handler, **kwargs) -> WebSearchTool:
    return WebSearchTool(transport=httpx.MockTransport(handler), **kwargs)


@pytest.mark.asyncio
async def test_parses_titles_and_links_from_html_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "html.duckduckgo.com"
        return httpx.Response(200, text=_SAMPLE_HTML)

    tool = _tool(handler)
    result = await tool.execute({"query": "arsenal fc"})

    assert result.success is True
    assert "First Result Title" in result.output
    assert "https://example.com/one" in result.output
    assert "Second Result Title" in result.output


@pytest.mark.asyncio
async def test_respects_max_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_SAMPLE_HTML)

    tool = _tool(handler, max_results=1)
    result = await tool.execute({"query": "arsenal fc"})

    assert "First Result Title" in result.output
    assert "Second Result Title" not in result.output


@pytest.mark.asyncio
async def test_no_results_found() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html><body>no matches</body></html>")

    tool = _tool(handler)
    result = await tool.execute({"query": "asdkfjhaslkdjfh"})

    assert result.success is True
    assert result.output == "No results found."


@pytest.mark.asyncio
async def test_http_error_is_reported_as_a_failed_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    tool = _tool(handler)
    result = await tool.execute({"query": "anything"})

    assert result.success is False
    assert "failed" in result.error.lower()


@pytest.mark.asyncio
async def test_missing_query_argument_fails_without_a_network_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should never make a network call without a query")

    tool = _tool(handler)
    result = await tool.execute({})

    assert result.success is False
    assert "query" in result.error.lower()
