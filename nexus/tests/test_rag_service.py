from __future__ import annotations

import pytest

from nexus.core.embeddings import EmbeddingProvider
from nexus.memory.sqlite_vector_store import SqliteVectorStore
from nexus.memory.storage import create_async_db_engine
from nexus.rag.service import RagService


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic fake: embeds each text as [len(text), 1.0] so search
    behavior is predictable without a real embedding model."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]

    async def health_check(self) -> bool:
        return True


async def _make_service(tmp_path, **kwargs) -> RagService:
    engine = create_async_db_engine(str(tmp_path / "nexus.db"))
    vector_store = SqliteVectorStore(engine)
    await vector_store.init()
    service = RagService(engine, FakeEmbeddingProvider(), vector_store, **kwargs)
    await service.init()
    return service


@pytest.mark.asyncio
async def test_ingest_then_retrieve_round_trip(tmp_path) -> None:
    service = await _make_service(tmp_path, chunk_size=100, chunk_overlap=10)

    summary = await service.ingest(
        user_id="u1",
        source_name="notes.txt",
        source_type="txt",
        raw_bytes=b"Arsenal are a football club based in London.",
    )

    assert summary.chunk_count == 1
    assert summary.source_name == "notes.txt"

    results = await service.retrieve(user_id="u1", query="football club London", top_k=3)

    assert len(results) == 1
    assert results[0].doc_id == summary.doc_id
    assert results[0].source_name == "notes.txt"
    assert "Arsenal" in results[0].chunk_text


@pytest.mark.asyncio
async def test_retrieve_respects_top_k(tmp_path) -> None:
    service = await _make_service(tmp_path, chunk_size=10, chunk_overlap=2)
    await service.ingest(
        user_id="u1",
        source_name="long.txt",
        source_type="txt",
        raw_bytes=b"a" * 100,
    )

    results = await service.retrieve(user_id="u1", query="a", top_k=2)

    assert len(results) <= 2


@pytest.mark.asyncio
async def test_retrieve_is_scoped_by_user_id(tmp_path) -> None:
    service = await _make_service(tmp_path)
    await service.ingest(user_id="u1", source_name="a.txt", source_type="txt", raw_bytes=b"content for user one")

    results = await service.retrieve(user_id="u2", query="content", top_k=5)

    assert results == []


@pytest.mark.asyncio
async def test_list_documents_returns_ingested_summaries(tmp_path) -> None:
    service = await _make_service(tmp_path)
    await service.ingest(user_id="u1", source_name="a.txt", source_type="txt", raw_bytes=b"first")
    await service.ingest(user_id="u1", source_name="b.txt", source_type="txt", raw_bytes=b"second")

    docs = await service.list_documents("u1")

    assert {d.source_name for d in docs} == {"a.txt", "b.txt"}


@pytest.mark.asyncio
async def test_delete_document_removes_it_from_listing_and_retrieval(tmp_path) -> None:
    service = await _make_service(tmp_path)
    summary = await service.ingest(
        user_id="u1", source_name="a.txt", source_type="txt", raw_bytes=b"deletable content"
    )

    await service.delete_document("u1", summary.doc_id)

    assert await service.list_documents("u1") == []
    assert await service.retrieve(user_id="u1", query="deletable", top_k=5) == []


@pytest.mark.asyncio
async def test_delete_document_for_wrong_user_is_a_no_op(tmp_path) -> None:
    service = await _make_service(tmp_path)
    summary = await service.ingest(
        user_id="u1", source_name="a.txt", source_type="txt", raw_bytes=b"owned by u1"
    )

    await service.delete_document("u2", summary.doc_id)

    docs = await service.list_documents("u1")
    assert len(docs) == 1


@pytest.mark.asyncio
async def test_ingest_empty_document_yields_zero_chunks(tmp_path) -> None:
    service = await _make_service(tmp_path)
    summary = await service.ingest(user_id="u1", source_name="empty.txt", source_type="txt", raw_bytes=b"")

    assert summary.chunk_count == 0
    assert await service.retrieve(user_id="u1", query="anything", top_k=5) == []
