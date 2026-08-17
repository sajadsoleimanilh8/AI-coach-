from __future__ import annotations

import time

import pytest

from nexus.memory.sqlite_vector_store import SqliteVectorStore
from nexus.memory.storage import DocumentRecord, create_async_db_engine, make_session_factory


async def _make_store(tmp_path) -> SqliteVectorStore:
    engine = create_async_db_engine(str(tmp_path / "nexus.db"))
    store = SqliteVectorStore(engine)
    await store.init()
    return store


async def _add_document(engine, *, doc_id: str, user_id: str, source_name: str = "doc.txt") -> None:
    session_factory = make_session_factory(engine)
    async with session_factory() as db:
        db.add(
            DocumentRecord(
                id=doc_id,
                user_id=user_id,
                source_name=source_name,
                source_type="txt",
                chunk_count=0,
                ingested_at=time.time(),
            )
        )
        await db.commit()


@pytest.mark.asyncio
async def test_search_returns_nearest_by_cosine(tmp_path) -> None:
    store = await _make_store(tmp_path)
    await _add_document(store._engine, doc_id="doc1", user_id="u1")

    await store.add_chunks(
        doc_id="doc1",
        chunks=["exact match vector", "orthogonal vector"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )

    results = await store.search([1.0, 0.0, 0.0], top_k=2, user_id="u1")

    assert results[0].chunk_text == "exact match vector"
    assert results[0].cosine_score == pytest.approx(1.0)
    assert results[1].cosine_score == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_search_respects_top_k(tmp_path) -> None:
    store = await _make_store(tmp_path)
    await _add_document(store._engine, doc_id="doc1", user_id="u1")
    await store.add_chunks(
        doc_id="doc1",
        chunks=["a", "b", "c"],
        embeddings=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
    )

    results = await store.search([1.0, 0.0], top_k=2, user_id="u1")

    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_is_scoped_by_user_id(tmp_path) -> None:
    store = await _make_store(tmp_path)
    await _add_document(store._engine, doc_id="doc-u1", user_id="u1")
    await _add_document(store._engine, doc_id="doc-u2", user_id="u2")
    await store.add_chunks(doc_id="doc-u1", chunks=["user one content"], embeddings=[[1.0, 0.0]])
    await store.add_chunks(doc_id="doc-u2", chunks=["user two content"], embeddings=[[1.0, 0.0]])

    results = await store.search([1.0, 0.0], top_k=10, user_id="u1")

    assert len(results) == 1
    assert results[0].chunk_text == "user one content"


@pytest.mark.asyncio
async def test_search_with_no_chunks_returns_empty_list(tmp_path) -> None:
    store = await _make_store(tmp_path)
    results = await store.search([1.0, 0.0], top_k=5, user_id="nobody")
    assert results == []


@pytest.mark.asyncio
async def test_delete_document_removes_its_chunks(tmp_path) -> None:
    store = await _make_store(tmp_path)
    await _add_document(store._engine, doc_id="doc1", user_id="u1")
    await store.add_chunks(doc_id="doc1", chunks=["a", "b"], embeddings=[[1.0, 0.0], [0.0, 1.0]])

    await store.delete_document("doc1")

    results = await store.search([1.0, 0.0], top_k=10, user_id="u1")
    assert results == []


@pytest.mark.asyncio
async def test_zero_vector_scores_as_zero_cosine(tmp_path) -> None:
    store = await _make_store(tmp_path)
    await _add_document(store._engine, doc_id="doc1", user_id="u1")
    await store.add_chunks(doc_id="doc1", chunks=["zero vector chunk"], embeddings=[[0.0, 0.0]])

    results = await store.search([1.0, 0.0], top_k=1, user_id="u1")

    assert results[0].cosine_score == 0.0
