from __future__ import annotations

import json

import httpx
import pytest

from nexus.core.exceptions import ModelNotFoundError, ProviderUnavailableError
from nexus.core.types import Message
from nexus.models.cloud.anthropic_provider import AnthropicProvider


def _provider(handler, api_key: str | None = "test-key") -> AnthropicProvider:
    return AnthropicProvider(api_key, transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_generate_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == "test-key"
        body = json.loads(request.content)
        assert body["model"] == "claude-sonnet-4-5"
        assert body["system"] == "be nice"
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "hi there"}],
                "usage": {"input_tokens": 5, "output_tokens": 3},
                "stop_reason": "end_turn",
            },
        )

    provider = _provider(handler)
    result = await provider.generate(
        [
            Message(role="system", content="be nice"),
            Message(role="user", content="hello"),
        ],
        model_id="claude-sonnet-4-5",
    )

    assert result.content == "hi there"
    assert result.model_used == "claude-sonnet-4-5"
    assert result.provider_name == "anthropic"
    assert result.usage.prompt_tokens == 5
    assert result.usage.completion_tokens == 3


@pytest.mark.asyncio
async def test_generate_missing_api_key_raises_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should never make a network call without an API key")

    provider = _provider(handler, api_key=None)
    with pytest.raises(ProviderUnavailableError):
        await provider.generate([Message(role="user", content="hi")], model_id="claude-sonnet-4-5")


@pytest.mark.asyncio
async def test_generate_connect_error_raises_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = _provider(handler)
    with pytest.raises(ProviderUnavailableError):
        await provider.generate([Message(role="user", content="hi")], model_id="claude-sonnet-4-5")


@pytest.mark.asyncio
async def test_generate_missing_model_raises_model_not_found() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "not found"}})

    provider = _provider(handler)
    with pytest.raises(ModelNotFoundError):
        await provider.generate([Message(role="user", content="hi")], model_id="does-not-exist")


@pytest.mark.asyncio
async def test_stream_generate_yields_deltas_then_done() -> None:
    events = [
        ("message_start", {"message": {"usage": {"input_tokens": 2}}}),
        ("content_block_delta", {"delta": {"type": "text_delta", "text": "Hel"}}),
        ("content_block_delta", {"delta": {"type": "text_delta", "text": "lo"}}),
        ("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}}),
        ("message_stop", {}),
    ]
    lines = []
    for event_name, data in events:
        lines.append(f"event: {event_name}")
        lines.append(f"data: {json.dumps(data)}")
        lines.append("")
    body = "\n".join(lines) + "\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body.encode("utf-8"))

    provider = _provider(handler)
    chunks = [
        chunk
        async for chunk in provider.stream_generate(
            [Message(role="user", content="hi")], model_id="claude-sonnet-4-5"
        )
    ]

    deltas = "".join(chunk.delta for chunk in chunks if not chunk.done)
    assert deltas == "Hello"
    assert chunks[-1].done is True
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.prompt_tokens == 2
    assert chunks[-1].usage.completion_tokens == 4
    assert chunks[-1].finish_reason == "end_turn"


@pytest.mark.asyncio
async def test_health_check_reports_true_when_reachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": []})

    assert await _provider(handler).health_check() is True


@pytest.mark.asyncio
async def test_health_check_reports_false_when_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    assert await _provider(handler).health_check() is False


@pytest.mark.asyncio
async def test_health_check_reports_false_when_api_key_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should never make a network call without an API key")

    assert await _provider(handler, api_key=None).health_check() is False


def test_count_tokens_is_a_rough_estimate() -> None:
    provider = AnthropicProvider("test-key")
    assert provider.count_tokens("a" * 40, model_id="claude-sonnet-4-5") == 10


@pytest.mark.asyncio
async def test_generate_with_tools_sends_canonical_schema_translated_to_anthropic_format() -> None:
    canonical_tools = [
        {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["tools"] == [
            {
                "name": "get_weather",
                "description": "Get the weather for a city.",
                "input_schema": canonical_tools[0]["parameters"],
            }
        ]
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4-5",
                "content": [
                    {"type": "tool_use", "id": "toolu_abc", "name": "get_weather", "input": {"city": "London"}}
                ],
                "usage": {"input_tokens": 12, "output_tokens": 6},
                "stop_reason": "tool_use",
            },
        )

    provider = _provider(handler)
    result = await provider.generate(
        [Message(role="user", content="weather in London?")],
        model_id="claude-sonnet-4-5",
        tools=canonical_tools,
    )

    assert result.content == ""
    assert result.tool_calls is not None
    [call] = result.tool_calls
    assert call.id == "toolu_abc"
    assert call.name == "get_weather"
    assert call.arguments == {"city": "London"}


@pytest.mark.asyncio
async def test_generate_without_tools_yields_no_tool_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "tools" not in json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "hi"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "stop_reason": "end_turn",
            },
        )

    provider = _provider(handler)
    result = await provider.generate(
        [Message(role="user", content="hi")], model_id="claude-sonnet-4-5"
    )

    assert result.tool_calls is None
