from __future__ import annotations

import os
from typing import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from nexus.api.main import create_app
from nexus.config.settings import get_settings
from nexus.core.provider_manager import ProviderManager
from nexus.core.providers import AIProvider
from nexus.core.router import ModelRouter
from nexus.core.types import GenerationChunk, GenerationResult, Message, ModelInfo, Usage
from nexus.models.registry import list_models


class FakeLocalProvider(AIProvider):
    name = "fake-local"

    async def generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        return GenerationResult(
            content="local echo", model_used=model_id, provider_name=self.name, usage=Usage()
        )

    async def stream_generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        yield GenerationChunk(delta="", done=True, usage=Usage())

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> bool:
        return True

    def count_tokens(self, text: str, *, model_id: str) -> int:
        return len(text)


class FakeCloudProvider(AIProvider):
    """Stands in for OpenAIProvider so this test needs no real API key/network call."""

    name = "fake-cloud"

    async def generate(
        self, messages: list[Message], *, model_id: str, temperature: float = 0.7, max_tokens=None, tools=None
    ) -> GenerationResult:
        return GenerationResult(
            content="cloud response",
            model_used=model_id,
            provider_name=self.name,
            usage=Usage(prompt_tokens=1000, completion_tokens=500),
        )

    async def stream_generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        yield GenerationChunk(delta="cloud", done=False)
        yield GenerationChunk(
            delta="", done=True, usage=Usage(prompt_tokens=1000, completion_tokens=500)
        )

    async def list_models(self) -> list[ModelInfo]:
        return [m for m in list_models() if m.provider == "openai"]

    async def health_check(self) -> bool:
        return True

    def count_tokens(self, text: str, *, model_id: str) -> int:
        return len(text)


@pytest.fixture(scope="module")
def _memory_db_env(tmp_path_factory) -> str:
    db_path = str(tmp_path_factory.mktemp("nexus-cloud") / "nexus.db")
    os.environ["NEXUS_MEMORY__DATABASE_PATH"] = db_path
    yield db_path
    os.environ.pop("NEXUS_MEMORY__DATABASE_PATH", None)


@pytest.fixture
async def client(_memory_db_env: str) -> AsyncIterator[AsyncClient]:
    get_settings(refresh=True)
    app = create_app()

    async with app.router.lifespan_context(app):
        provider_manager = ProviderManager(
            {"local": FakeLocalProvider(), "openai": FakeCloudProvider()}
        )
        app.state.provider_manager = provider_manager
        app.state.provider = provider_manager.get("local")
        app.state.router = ModelRouter(provider_manager, list_models())

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest.mark.asyncio
async def test_chat_routes_to_cloud_provider_and_reports_cost(client: AsyncClient) -> None:
    response = await client.post(
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "model_id": "gpt-4o-mini",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["content"] == "cloud response"
    assert body["provider_name"] == "openai"
    assert "Explicit model_id=gpt-4o-mini" in body["routing_reason"]
    assert body["cost_usd"] == pytest.approx(0.00015 + 0.0003)


@pytest.mark.asyncio
async def test_chat_streaming_to_cloud_provider_reports_cost_in_final_event(
    client: AsyncClient,
) -> None:
    async with client.stream(
        "POST",
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "model_id": "gpt-4o-mini",
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines() if line]

    assert any('"provider_name": "openai"' in line for line in lines)
    assert any('"cost_usd"' in line for line in lines)
