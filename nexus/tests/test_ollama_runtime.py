from __future__ import annotations

import json

import httpx
import pytest

from nexus.core.exceptions import ModelNotFoundError, ProviderUnavailableError
from nexus.core.types import Message
from nexus.models.local.runtime import OllamaRuntime


def _runtime(handler) -> OllamaRuntime:
    return OllamaRuntime("http://fake-ollama:11434", transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_generate_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        assert body["model"] == "mistral:7b"
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "hi there"},
                "done": True,
                "prompt_eval_count": 5,
                "eval_count": 3,
            },
        )

    runtime = _runtime(handler)
    result = await runtime.generate([Message(role="user", content="hello")], model_id="mistral:7b")

    assert result.content == "hi there"
    assert result.model_used == "mistral:7b"
    assert result.usage.prompt_tokens == 5
    assert result.usage.completion_tokens == 3
    assert result.usage.total_tokens == 8


@pytest.mark.asyncio
async def test_generate_connect_error_raises_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    runtime = _runtime(handler)
    with pytest.raises(ProviderUnavailableError):
        await runtime.generate([Message(role="user", content="hi")], model_id="mistral:7b")


@pytest.mark.asyncio
async def test_generate_missing_model_raises_model_not_found() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    runtime = _runtime(handler)
    with pytest.raises(ModelNotFoundError):
        await runtime.generate([Message(role="user", content="hi")], model_id="does-not-exist")


@pytest.mark.asyncio
async def test_list_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "mistral:7b"}]})

    runtime = _runtime(handler)
    models = await runtime.list_models()

    assert [m.id for m in models] == ["mistral:7b"]


@pytest.mark.asyncio
async def test_health_check_reports_true_when_reachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": []})

    assert await _runtime(handler).health_check() is True


@pytest.mark.asyncio
async def test_health_check_reports_false_when_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    assert await _runtime(handler).health_check() is False


@pytest.mark.asyncio
async def test_stream_generate_yields_deltas_then_done() -> None:
    lines = [
        {"message": {"content": "Hel"}, "done": False},
        {"message": {"content": "lo"}, "done": False},
        {"done": True, "prompt_eval_count": 2, "eval_count": 4},
    ]
    body = "\n".join(json.dumps(line) for line in lines)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body.encode("utf-8"))

    runtime = _runtime(handler)
    chunks = [
        chunk
        async for chunk in runtime.stream_generate(
            [Message(role="user", content="hi")], model_id="mistral:7b"
        )
    ]

    deltas = "".join(chunk.delta for chunk in chunks if not chunk.done)
    assert deltas == "Hello"
    assert chunks[-1].done is True
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.completion_tokens == 4


def test_count_tokens_is_a_rough_estimate() -> None:
    runtime = OllamaRuntime("http://fake-ollama:11434")
    assert runtime.count_tokens("a" * 40, model_id="mistral:7b") == 10
