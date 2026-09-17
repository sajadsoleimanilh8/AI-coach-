from __future__ import annotations

import base64
import os
from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from nexus.api.main import create_app
from nexus.config.settings import get_settings
from nexus.core.embeddings import EmbeddingProvider
from nexus.core.provider_manager import ProviderManager
from nexus.core.providers import AIProvider
from nexus.core.router import ModelRouter
from nexus.core.types import GenerationChunk, GenerationResult, Message, ModelInfo, Usage
from nexus.memory.sqlite_vector_store import SqliteVectorStore
from nexus.memory.storage import create_async_db_engine
from nexus.models.registry import list_models
from nexus.rag.service import RagService


class FakeEmbeddingProvider(EmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]

    async def health_check(self) -> bool:
        return True


class FakeChatProvider(AIProvider):
    """Captures the messages it was called with, so tests can verify the
    synthetic RAG context message actually reached the provider."""

    name = "fake-local"

    def __init__(self) -> None:
        self.received_messages: list[Message] | None = None

    async def generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        self.received_messages = messages
        return GenerationResult(
            content="Arsenal are based in London.",
            model_used=model_id,
            provider_name=self.name,
            usage=Usage(prompt_tokens=5, completion_tokens=5),
        )

    async def stream_generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        self.received_messages = messages
        yield GenerationChunk(delta="Arsenal", done=False)
        yield GenerationChunk(delta="", done=True, usage=Usage(prompt_tokens=5, completion_tokens=5))

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> bool:
        return True

    def count_tokens(self, text: str, *, model_id: str) -> int:
        return len(text)


@pytest.fixture(scope="module")
def _memory_db_env(tmp_path_factory) -> str:
    db_path = str(tmp_path_factory.mktemp("nexus-rag") / "nexus.db")
    os.environ["NEXUS_MEMORY__DATABASE_PATH"] = db_path
    yield db_path
    os.environ.pop("NEXUS_MEMORY__DATABASE_PATH", None)


@pytest.fixture
async def client_and_provider(
    _memory_db_env: str,
) -> AsyncIterator[tuple[AsyncClient, FakeChatProvider]]:
    get_settings(refresh=True)
    app = create_app()

    async with app.router.lifespan_context(app):
        fake_provider = FakeChatProvider()
        provider_manager = ProviderManager({"local": fake_provider})
        app.state.services.provider_manager = provider_manager
        app.state.services.provider = fake_provider
        app.state.services.router = ModelRouter(provider_manager, list_models())

        # Swap in a fake embedding provider (real one would try to call a
        # local Ollama instance this test suite never starts) while
        # reusing the same on-disk SQLite file the rest of app.state
        # already points at.
        engine = create_async_db_engine(_memory_db_env)
        vector_store = SqliteVectorStore(engine)
        app.state.services.rag_service = RagService(
            engine, FakeEmbeddingProvider(), vector_store, chunk_size=500, chunk_overlap=50
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac, fake_provider


async def _ingest(client: AsyncClient, *, user_id: str = "u1") -> str:
    content = base64.b64encode(b"Arsenal are a football club based in London.").decode()
    response = await client.post(
        "/api/documents",
        json={
            "user_id": user_id,
            "source_name": "arsenal.txt",
            "source_type": "txt",
            "content_base64": content,
        },
    )
    assert response.status_code == 200
    return response.json()["doc_id"]


@pytest.mark.asyncio
async def test_use_rag_returns_citations_and_reaches_the_provider(
    client_and_provider: tuple[AsyncClient, FakeChatProvider],
) -> None:
    client, provider = client_and_provider
    doc_id = await _ingest(client)

    response = await client.post(
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "Where is Arsenal based?"}],
            "user_id": "u1",
            "use_rag": True,
            "model_id": "mistral:7b",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["citations"]
    assert body["citations"][0]["doc_id"] == doc_id
    assert "London" in body["citations"][0]["chunk_text"]

    assert provider.received_messages is not None
    system_messages = [m for m in provider.received_messages if m.role == "system"]
    assert any("Relevant context from your documents" in m.content for m in system_messages)
    assert any("Arsenal" in m.content for m in system_messages)


@pytest.mark.asyncio
async def test_use_rag_with_no_matching_documents_returns_empty_citations_not_an_error(
    client_and_provider: tuple[AsyncClient, FakeChatProvider],
) -> None:
    client, provider = client_and_provider

    response = await client.post(
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "Anything at all?"}],
            "user_id": "a-user-with-no-documents",
            "use_rag": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["citations"] == []
    assert not any(
        "Relevant context from your documents" in m.content for m in provider.received_messages
    )


@pytest.mark.asyncio
async def test_without_use_rag_citations_is_null_and_no_context_is_injected(
    client_and_provider: tuple[AsyncClient, FakeChatProvider],
) -> None:
    client, provider = client_and_provider
    await _ingest(client)

    response = await client.post(
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "Where is Arsenal based?"}],
            "user_id": "u1",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["citations"] is None
    assert not any(
        "Relevant context from your documents" in m.content for m in provider.received_messages
    )


@pytest.mark.asyncio
async def test_rag_streaming_final_event_includes_citations(
    client_and_provider: tuple[AsyncClient, FakeChatProvider],
) -> None:
    client, _provider = client_and_provider
    await _ingest(client)

    async with client.stream(
        "POST",
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "Where is Arsenal based?"}],
            "user_id": "u1",
            "use_rag": True,
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines() if line]

    assert any('"citations"' in line and "arsenal.txt" in line for line in lines)
