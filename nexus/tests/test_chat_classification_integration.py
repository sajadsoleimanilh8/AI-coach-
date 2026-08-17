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
from nexus.core.types import GenerationChunk, GenerationResult, ModelInfo, Usage
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

    async def generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        return GenerationResult(
            content="cloud response",
            model_used=model_id,
            provider_name=self.name,
            usage=Usage(prompt_tokens=100, completion_tokens=50),
        )

    async def stream_generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        yield GenerationChunk(delta="cloud", done=False)
        yield GenerationChunk(
            delta="", done=True, usage=Usage(prompt_tokens=100, completion_tokens=50)
        )

    async def list_models(self) -> list[ModelInfo]:
        return [m for m in list_models() if m.provider == "openai"]

    async def health_check(self) -> bool:
        return True

    def count_tokens(self, text: str, *, model_id: str) -> int:
        return len(text)


@pytest.fixture(scope="module")
def _memory_db_env(tmp_path_factory) -> str:
    db_path = str(tmp_path_factory.mktemp("nexus-classification") / "nexus.db")
    os.environ["NEXUS_MEMORY__DATABASE_PATH"] = db_path
    yield db_path
    os.environ.pop("NEXUS_MEMORY__DATABASE_PATH", None)


async def _make_client(app) -> AsyncIterator[AsyncClient]:
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


@pytest.fixture
async def client(_memory_db_env: str) -> AsyncIterator[AsyncClient]:
    get_settings(refresh=True)
    async for ac in _make_client(create_app()):
        yield ac


@pytest.fixture
async def client_with_classification_disabled(_memory_db_env: str) -> AsyncIterator[AsyncClient]:
    os.environ["NEXUS_ROUTING__CLASSIFICATION__ENABLED"] = "false"
    os.environ["NEXUS_ROUTING__PRIVACY__ENABLED"] = "false"
    get_settings(refresh=True)
    try:
        async for ac in _make_client(create_app()):
            yield ac
    finally:
        os.environ.pop("NEXUS_ROUTING__CLASSIFICATION__ENABLED", None)
        os.environ.pop("NEXUS_ROUTING__PRIVACY__ENABLED", None)
        get_settings(refresh=True)


@pytest.mark.asyncio
async def test_coding_query_classified_and_routed_via_coding_capability(
    client: AsyncClient,
) -> None:
    response = await client.post(
        "/api/chat",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I'm getting a stack trace when I run this function, "
                        "help me debug the bug"
                    ),
                }
            ]
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task_type"] == "coding"
    assert "task=coding" in body["routing_reason"]
    assert "task: Classified as coding" in body["classification_reason"]


@pytest.mark.asyncio
async def test_email_in_query_forces_local_only_despite_balanced_default(
    client: AsyncClient,
) -> None:
    response = await client.post(
        "/api/chat",
        json={
            "messages": [
                {"role": "user", "content": "My email is jane.doe@example.com, can you help?"}
            ]
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["provider_name"] == "local"
    assert body["privacy_level"] == "private"
    assert "LOCAL_ONLY policy" in body["routing_reason"]
    assert "privacy:" in body["classification_reason"]


@pytest.mark.asyncio
async def test_explicit_model_id_bypasses_privacy_override(client: AsyncClient) -> None:
    response = await client.post(
        "/api/chat",
        json={
            "messages": [
                {"role": "user", "content": "My email is jane.doe@example.com, can you help?"}
            ],
            "model_id": "gpt-4o-mini",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["privacy_level"] == "private"
    assert body["provider_name"] == "openai"
    assert body["content"] == "cloud response"


@pytest.mark.asyncio
async def test_explicit_policy_bypasses_privacy_override(client: AsyncClient) -> None:
    response = await client.post(
        "/api/chat",
        json={
            "messages": [
                {"role": "user", "content": "My email is jane.doe@example.com, can you help?"}
            ],
            "policy": "MAX_QUALITY",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["privacy_level"] == "private"
    assert "MAX_QUALITY policy" in body["routing_reason"]
    assert "LOCAL_ONLY" not in body["routing_reason"]


@pytest.mark.asyncio
async def test_disabled_toggles_restore_phase_2_behavior(
    client_with_classification_disabled: AsyncClient,
) -> None:
    response = await client_with_classification_disabled.post(
        "/api/chat",
        json={
            "messages": [
                {"role": "user", "content": "My email is jane.doe@example.com, can you help?"}
            ]
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task_type"] is None
    assert body["privacy_level"] is None
    assert body["classification_reason"] is None
    assert "LOCAL_ONLY" not in body["routing_reason"]
