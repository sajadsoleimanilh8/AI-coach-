from __future__ import annotations

import json

import httpx
import pytest

from nexus.core.exceptions import ModelNotFoundError, ProviderUnavailableError
from nexus.core.types import Message
from nexus.models.cloud.gemini_provider import GeminiProvider

_MODEL = "gemini-2.5-flash"


def _provider(handler, api_key: str | None = "test-key") -> GeminiProvider:
    return GeminiProvider(api_key, transport=httpx.MockTransport(handler))


def _response(text: str = "hi there", **extra) -> dict:
    payload = {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}
        ],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
        "modelVersion": _MODEL,
    }
    payload.update(extra)
    return payload


@pytest.mark.asyncio
async def test_generate_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith(f"/models/{_MODEL}:generateContent")
        assert request.headers["x-goog-api-key"] == "test-key"
        body = json.loads(request.content)
        assert body["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
        assert body["generationConfig"]["temperature"] == 0.7
        return httpx.Response(200, json=_response())

    provider = _provider(handler)
    result = await provider.generate([Message(role="user", content="hello")], model_id=_MODEL)

    assert result.content == "hi there"
    assert result.model_used == _MODEL
    assert result.provider_name == "gemini"
    assert result.usage.prompt_tokens == 5
    assert result.usage.completion_tokens == 3
    assert result.finish_reason == "STOP"


@pytest.mark.asyncio
async def test_api_key_is_sent_as_header_not_query_string() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "test-key" not in str(request.url)
        assert request.headers["x-goog-api-key"] == "test-key"
        return httpx.Response(200, json=_response())

    provider = _provider(handler)
    await provider.generate([Message(role="user", content="hi")], model_id=_MODEL)


@pytest.mark.asyncio
async def test_assistant_role_maps_to_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        roles = [c["role"] for c in json.loads(request.content)["contents"]]
        assert roles == ["user", "model", "user"]
        return httpx.Response(200, json=_response())

    provider = _provider(handler)
    await provider.generate(
        [
            Message(role="user", content="one"),
            Message(role="assistant", content="two"),
            Message(role="user", content="three"),
        ],
        model_id=_MODEL,
    )


@pytest.mark.asyncio
async def test_system_message_goes_to_system_instruction_not_contents() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["systemInstruction"] == {"parts": [{"text": "be nice"}]}
        assert body["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
        return httpx.Response(200, json=_response())

    provider = _provider(handler)
    await provider.generate(
        [Message(role="system", content="be nice"), Message(role="user", content="hello")],
        model_id=_MODEL,
    )


@pytest.mark.asyncio
async def test_multiple_system_messages_are_concatenated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["systemInstruction"] == {"parts": [{"text": "first\nsecond"}]}
        return httpx.Response(200, json=_response())

    provider = _provider(handler)
    await provider.generate(
        [
            Message(role="system", content="first"),
            Message(role="system", content="second"),
            Message(role="user", content="hello"),
        ],
        model_id=_MODEL,
    )


@pytest.mark.asyncio
async def test_no_system_message_omits_system_instruction() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "systemInstruction" not in json.loads(request.content)
        return httpx.Response(200, json=_response())

    provider = _provider(handler)
    await provider.generate([Message(role="user", content="hello")], model_id=_MODEL)


@pytest.mark.asyncio
async def test_tools_translate_to_function_declarations() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        tools = json.loads(request.content)["tools"]
        assert tools == [
            {
                "functionDeclarations": [
                    {
                        "name": "get_weather",
                        "description": "Look up weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                ]
            }
        ]
        return httpx.Response(200, json=_response())

    provider = _provider(handler)
    await provider.generate(
        [Message(role="user", content="weather?")],
        model_id=_MODEL,
        tools=[
            {
                "name": "get_weather",
                "description": "Look up weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ],
    )


@pytest.mark.asyncio
async def test_function_call_response_parses_into_tool_call_with_synthesized_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"functionCall": {"name": "get_weather", "args": {"city": "Oslo"}}}
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 6},
            },
        )

    provider = _provider(handler)
    result = await provider.generate([Message(role="user", content="weather?")], model_id=_MODEL)

    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.name == "get_weather"
    assert call.arguments == {"city": "Oslo"}
    assert call.id


@pytest.mark.asyncio
async def test_synthesized_tool_call_ids_are_unique_per_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"functionCall": {"name": "a", "args": {}}},
                                {"functionCall": {"name": "b", "args": {}}},
                            ],
                        }
                    }
                ]
            },
        )

    provider = _provider(handler)
    result = await provider.generate([Message(role="user", content="hi")], model_id=_MODEL)

    assert result.tool_calls is not None
    assert result.tool_calls[0].id != result.tool_calls[1].id


@pytest.mark.asyncio
async def test_generate_without_tool_calls_returns_none() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_response()))
    result = await provider.generate([Message(role="user", content="hi")], model_id=_MODEL)
    assert result.tool_calls is None


@pytest.mark.asyncio
async def test_generate_missing_api_key_raises_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should never make a network call without an API key")

    provider = _provider(handler, api_key=None)
    with pytest.raises(ProviderUnavailableError):
        await provider.generate([Message(role="user", content="hi")], model_id=_MODEL)


@pytest.mark.asyncio
async def test_generate_connect_error_raises_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = _provider(handler)
    with pytest.raises(ProviderUnavailableError):
        await provider.generate([Message(role="user", content="hi")], model_id=_MODEL)


@pytest.mark.asyncio
async def test_generate_timeout_raises_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    provider = _provider(handler)
    with pytest.raises(ProviderUnavailableError):
        await provider.generate([Message(role="user", content="hi")], model_id=_MODEL)


@pytest.mark.asyncio
async def test_generate_missing_model_raises_model_not_found() -> None:
    provider = _provider(lambda request: httpx.Response(404, json={"error": {"message": "nope"}}))
    with pytest.raises(ModelNotFoundError):
        await provider.generate([Message(role="user", content="hi")], model_id="does-not-exist")


@pytest.mark.asyncio
async def test_generate_server_error_raises_provider_unavailable() -> None:
    provider = _provider(lambda request: httpx.Response(500, json={"error": {}}))
    with pytest.raises(ProviderUnavailableError):
        await provider.generate([Message(role="user", content="hi")], model_id=_MODEL)


@pytest.mark.asyncio
async def test_generate_rejected_key_raises_provider_unavailable_without_leaking_key() -> None:
    provider = _provider(lambda request: httpx.Response(403, json={"error": {}}))
    with pytest.raises(ProviderUnavailableError) as exc_info:
        await provider.generate([Message(role="user", content="hi")], model_id=_MODEL)
    assert "test-key" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_stream_generate_yields_deltas_then_done() -> None:
    events = [
        {"candidates": [{"content": {"role": "model", "parts": [{"text": "Hel"}]}}]},
        {"candidates": [{"content": {"role": "model", "parts": [{"text": "lo"}]}}]},
        {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": ""}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 4},
        },
    ]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith(f"/models/{_MODEL}:streamGenerateContent")
        assert request.url.params["alt"] == "sse"
        return httpx.Response(200, text=body)

    provider = _provider(handler)
    chunks = [
        chunk
        async for chunk in provider.stream_generate(
            [Message(role="user", content="hi")], model_id=_MODEL
        )
    ]

    assert [c.delta for c in chunks[:2]] == ["Hel", "lo"]
    assert all(c.done is False for c in chunks[:2])
    final = chunks[-1]
    assert final.done is True
    assert final.finish_reason == "STOP"
    assert final.usage is not None
    assert final.usage.prompt_tokens == 2
    assert final.usage.completion_tokens == 4


@pytest.mark.asyncio
async def test_stream_generate_missing_api_key_raises_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should never make a network call without an API key")

    provider = _provider(handler, api_key=None)
    with pytest.raises(ProviderUnavailableError):
        async for _ in provider.stream_generate(
            [Message(role="user", content="hi")], model_id=_MODEL
        ):
            pass


@pytest.mark.asyncio
async def test_stream_generate_missing_model_raises_model_not_found() -> None:
    provider = _provider(lambda request: httpx.Response(404, json={"error": {}}))
    with pytest.raises(ModelNotFoundError):
        async for _ in provider.stream_generate(
            [Message(role="user", content="hi")], model_id="does-not-exist"
        ):
            pass


@pytest.mark.asyncio
async def test_health_check_true_when_models_endpoint_returns_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models")
        return httpx.Response(200, json={"models": []})

    assert await _provider(handler).health_check() is True


@pytest.mark.asyncio
async def test_health_check_false_on_error_status() -> None:
    provider = _provider(lambda request: httpx.Response(503, json={}))
    assert await provider.health_check() is False


@pytest.mark.asyncio
async def test_health_check_false_on_connect_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    assert await _provider(handler).health_check() is False


@pytest.mark.asyncio
async def test_health_check_false_without_api_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should never make a network call without an API key")

    assert await _provider(handler, api_key=None).health_check() is False


def test_count_tokens_uses_local_approximation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("count_tokens must not make a network call")

    provider = _provider(handler)
    assert provider.count_tokens("a" * 40, model_id=_MODEL) == 10
    assert provider.count_tokens("", model_id=_MODEL) == 1
