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


class _FakeProvider(AIProvider):
    """Captures the messages it was called with, so tests can verify
    whether the synthetic personal-context system message actually reached
    the provider — the same technique test_chat_rag_integration.py uses
    for the RAG context message."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.received_messages: list[Message] | None = None

    async def generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        self.received_messages = messages
        return GenerationResult(
            content="Here is my answer.",
            model_used=model_id,
            provider_name=self.name,
            usage=Usage(prompt_tokens=5, completion_tokens=5),
        )

    async def stream_generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        self.received_messages = messages
        yield GenerationChunk(delta="Here", done=False)
        yield GenerationChunk(delta="", done=True, usage=Usage(prompt_tokens=5, completion_tokens=5))

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> bool:
        return True

    def count_tokens(self, text: str, *, model_id: str) -> int:
        return len(text)


@pytest.fixture
async def client_and_providers(
    _memory_db_env: str,
) -> AsyncIterator[tuple[AsyncClient, _FakeProvider, _FakeProvider]]:
    get_settings(refresh=True)
    app = create_app()

    async with app.router.lifespan_context(app):
        local_provider = _FakeProvider("local")
        cloud_provider = _FakeProvider("openai")
        provider_manager = ProviderManager({"local": local_provider, "openai": cloud_provider})
        app.state.services.provider_manager = provider_manager
        app.state.services.router = ModelRouter(provider_manager, list_models())

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac, local_provider, cloud_provider


async def _record_a_signal(client: AsyncClient, user_id: str) -> None:
    response = await client.post(
        f"/api/personal/{user_id}/signal",
        json={"dimension": "physical.energy", "value": 0.8, "source": "explicit"},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_personal_context_is_injected_on_a_local_route(
    client_and_providers: tuple[AsyncClient, _FakeProvider, _FakeProvider],
) -> None:
    client, local_provider, _cloud_provider = client_and_providers
    await _record_a_signal(client, "u1")

    response = await client.post(
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "How am I doing?"}],
            "user_id": "u1",
            "use_personal_context": True,
            "model_id": "mistral:7b",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["personal_context_used"] is True
    assert body["provider_name"] == "local"

    assert local_provider.received_messages is not None
    system_messages = [m for m in local_provider.received_messages if m.role == "system"]
    assert any("physical.energy" in m.content for m in system_messages)


@pytest.mark.asyncio
async def test_personal_context_is_withheld_from_a_cloud_provider(
    client_and_providers: tuple[AsyncClient, _FakeProvider, _FakeProvider],
) -> None:
    client, _local_provider, cloud_provider = client_and_providers
    await _record_a_signal(client, "u2")

    response = await client.post(
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "How am I doing?"}],
            "user_id": "u2",
            "use_personal_context": True,
            "model_id": "gpt-4o-mini",  # cloud provider (openai)
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["personal_context_used"] is False
    assert body["provider_name"] == "openai"

    assert cloud_provider.received_messages is not None
    assert not any(
        "physical.energy" in m.content for m in cloud_provider.received_messages
    )


@pytest.mark.asyncio
async def test_personal_context_not_requested_leaves_field_null(
    client_and_providers: tuple[AsyncClient, _FakeProvider, _FakeProvider],
) -> None:
    client, _local_provider, _cloud_provider = client_and_providers
    await _record_a_signal(client, "u3")

    response = await client.post(
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "user_id": "u3",
            "model_id": "mistral:7b",
        },
    )

    assert response.status_code == 200
    assert response.json()["personal_context_used"] is None


@pytest.mark.asyncio
async def test_personal_context_false_when_nothing_recorded_yet(
    client_and_providers: tuple[AsyncClient, _FakeProvider, _FakeProvider],
) -> None:
    client, _local_provider, _cloud_provider = client_and_providers

    response = await client.post(
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "How am I doing?"}],
            "user_id": "a-user-with-no-signals",
            "use_personal_context": True,
            "model_id": "mistral:7b",
        },
    )

    assert response.status_code == 200
    assert response.json()["personal_context_used"] is False
