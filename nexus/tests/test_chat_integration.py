from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from nexus.api.main import create_app
from nexus.config.settings import get_settings
from nexus.core.provider_manager import ProviderManager
from nexus.core.providers import AIProvider
from nexus.core.router import ModelRouter
from nexus.core.types import GenerationChunk, GenerationResult, Message, ModelInfo, Usage
from nexus.models.registry import list_models


class FakeProvider(AIProvider):
    """Stands in for OllamaRuntime so this test needs no running backend."""

    name = "fake"

    async def generate(
        self, messages: list[Message], *, model_id: str, temperature: float = 0.7, max_tokens=None, tools=None
    ) -> GenerationResult:
        last_user = messages[-1].content if messages else ""
        return GenerationResult(
            content=f"echo: {last_user}",
            model_used=model_id,
            provider_name=self.name,
            usage=Usage(prompt_tokens=1, completion_tokens=1),
        )

    async def stream_generate(self, messages: list[Message], *, model_id: str, temperature: float = 0.7, max_tokens=None, tools=None):
        for piece in ("ech", "o"):
            yield GenerationChunk(delta=piece, done=False)
        yield GenerationChunk(delta="", done=True, usage=Usage(prompt_tokens=1, completion_tokens=2))

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id="fake-model", provider=self.name)]

    async def health_check(self) -> bool:
        return True

    def count_tokens(self, text: str, *, model_id: str) -> int:
        return len(text)


@pytest.fixture
async def client(_memory_db_env: str) -> AsyncIterator[AsyncClient]:
    get_settings(refresh=True)
    app = create_app()

    async with app.router.lifespan_context(app):
        # FakeProvider registered under the "local" key so it's what
        # requesting the real registry's "mistral:7b" (provider: local)
        # resolves to, and so it's the only chat candidate the router sees
        # (no openai/anthropic keys are registered in this fixture) —
        # matching Phase 1's local-only test setup without a running Ollama.
        fake_provider = FakeProvider()
        provider_manager = ProviderManager({"local": fake_provider})
        app.state.services.provider_manager = provider_manager
        app.state.services.provider = fake_provider
        app.state.services.router = ModelRouter(provider_manager, list_models())

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest.mark.asyncio
async def test_health_endpoint_reports_provider_status(client: AsyncClient) -> None:
    response = await client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["providers"] == {"local": True}
    assert body["local_models"] == ["fake-model"]


@pytest.mark.asyncio
async def test_chat_end_to_end_creates_and_reuses_session(client: AsyncClient) -> None:
    first = await client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "hello"}], "model_id": "mistral:7b"},
    )
    assert first.status_code == 200
    body = first.json()
    assert body["content"] == "echo: hello"
    assert body["model_used"] == "mistral:7b"
    assert body["usage"]["total_tokens"] == 2
    assert body["provider_name"] == "local"
    assert body["cost_usd"] == 0.0
    session_id = body["session_id"]
    assert session_id

    second = await client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "again"}], "session_id": session_id},
    )
    assert second.status_code == 200
    assert second.json()["session_id"] == session_id


@pytest.mark.asyncio
async def test_chat_with_unknown_session_id_returns_404(client: AsyncClient) -> None:
    response = await client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "session_id": "does-not-exist"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_chat_streaming_returns_sse_chunks(client: AsyncClient) -> None:
    async with client.stream(
        "POST",
        "/api/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = [line async for line in response.aiter_lines() if line]

    assert any('"delta": "ech"' in line for line in lines)
    assert any('"done": true' in line for line in lines)
