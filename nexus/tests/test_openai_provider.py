from __future__ import annotations

import json

import httpx
import pytest

from nexus.core.exceptions import ModelNotFoundError, ProviderUnavailableError
from nexus.core.types import Message
from nexus.models.cloud.openai_provider import OpenAIProvider


def _provider(handler, api_key: str | None = "test-key") -> OpenAIProvider:
    return OpenAIProvider(api_key, transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_generate_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["model"] == "gpt-4o-mini"
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [
                    {"message": {"role": "assistant", "content": "hi there"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            },
        )

    provider = _provider(handler)
    result = await provider.generate(
        [Message(role="user", content="hello")], model_id="gpt-4o-mini"
    )

    assert result.content == "hi there"
    assert result.model_used == "gpt-4o-mini"
    assert result.provider_name == "openai"
    assert result.usage.prompt_tokens == 5
    assert result.usage.completion_tokens == 3


@pytest.mark.asyncio
async def test_generate_missing_api_key_raises_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should never make a network call without an API key")

    provider = _provider(handler, api_key=None)
    with pytest.raises(ProviderUnavailableError):
        await provider.generate([Message(role="user", content="hi")], model_id="gpt-4o-mini")


@pytest.mark.asyncio
async def test_generate_connect_error_raises_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = _provider(handler)
    with pytest.raises(ProviderUnavailableError):
        await provider.generate([Message(role="user", content="hi")], model_id="gpt-4o-mini")


@pytest.mark.asyncio
async def test_generate_missing_model_raises_model_not_found() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "model not found"}})

    provider = _provider(handler)
    with pytest.raises(ModelNotFoundError):
        await provider.generate([Message(role="user", content="hi")], model_id="does-not-exist")


@pytest.mark.asyncio
async def test_stream_generate_yields_deltas_then_done() -> None:
    lines = [
        "data: " + json.dumps({"choices": [{"delta": {"content": "Hel"}}]}),
        "data: " + json.dumps({"choices": [{"delta": {"content": "lo"}}]}),
        "data: "
        + json.dumps({"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 4}}),
        "data: [DONE]",
    ]
    body = "\n\n".join(lines) + "\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body.encode("utf-8"))

    provider = _provider(handler)
    chunks = [
        chunk
        async for chunk in provider.stream_generate(
            [Message(role="user", content="hi")], model_id="gpt-4o-mini"
        )
    ]

    deltas = "".join(chunk.delta for chunk in chunks if not chunk.done)
    assert deltas == "Hello"
    assert chunks[-1].done is True
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.completion_tokens == 4


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


def test_count_tokens_returns_a_positive_estimate() -> None:
    provider = OpenAIProvider("test-key")
    assert provider.count_tokens("hello world", model_id="gpt-4o-mini") > 0


@pytest.mark.asyncio
async def test_generate_with_tools_sends_canonical_schema_translated_to_openai_format() -> None:
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
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a city.",
                    "parameters": canonical_tools[0]["parameters"],
                },
            }
        ]
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "London"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 6},
            },
        )

    provider = _provider(handler)
    result = await provider.generate(
        [Message(role="user", content="weather in London?")],
        model_id="gpt-4o-mini",
        tools=canonical_tools,
    )

    assert result.content == ""
    assert result.tool_calls is not None
    [call] = result.tool_calls
    assert call.id == "call_abc"
    assert call.name == "get_weather"
    assert call.arguments == {"city": "London"}


@pytest.mark.asyncio
async def test_generate_without_tools_yields_no_tool_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "tools" not in json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    provider = _provider(handler)
    result = await provider.generate([Message(role="user", content="hi")], model_id="gpt-4o-mini")

    assert result.tool_calls is None
