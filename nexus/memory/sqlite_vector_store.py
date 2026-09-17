from __future__ import annotations

import json
import time

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from nexus.core.vector_store import RetrievedChunk, VectorStore
from nexus.memory.storage import DocumentChunkRecord, DocumentRecord, init_db, make_session_factory


class SqliteVectorStore(VectorStore):
    """MVP implementation: brute-force cosine similarity over all chunks,
    scoped by joining document_chunks -> documents on user_id. Fine at
    personal/small-team scale (hundreds to low thousands of chunks);
    intentionally NOT an ANN index. Swap this implementation later (e.g.
    FAISS/pgvector) without touching VectorStore callers — that's the
    point of the abstraction.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._session_factory: async_sessionmaker = make_session_factory(engine)

    async def init(self) -> None:
        await init_db(self._engine)

    async def add_chunks(
        self, *, doc_id: str, chunks: list[str], embeddings: list[list[float]]
    ) -> None:
        now = time.time()
        async with self._session_factory() as db:
            for index, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                db.add(
                    DocumentChunkRecord(
                        doc_id=doc_id,
                        chunk_index=index,
                        chunk_text=chunk_text,
                        embedding_json=json.dumps(embedding),
                        created_at=now,
                    )
                )
            await db.commit()

    async def search(
        self, query_embedding: list[float], *, top_k: int, user_id: str
    ) -> list[RetrievedChunk]:
        async with self._session_factory() as db:
            result = await db.execute(
                select(DocumentChunkRecord, DocumentRecord.source_name)
                .join(DocumentRecord, DocumentChunkRecord.doc_id == DocumentRecord.id)
                .where(DocumentRecord.user_id == user_id)
            )
            rows = result.all()

        if not rows:
            return []

        query_vec = np.asarray(query_embedding, dtype=float)
        query_norm = float(np.linalg.norm(query_vec))

        scored: list[RetrievedChunk] = []
        for chunk_record, source_name in rows:
            chunk_vec = np.asarray(json.loads(chunk_record.embedding_json), dtype=float)
            chunk_norm = float(np.linalg.norm(chunk_vec))
            if query_norm == 0.0 or chunk_norm == 0.0 or chunk_vec.shape != query_vec.shape:
                cosine = 0.0
            else:
                cosine = float(np.dot(query_vec, chunk_vec) / (query_norm * chunk_norm))
            scored.append(
                RetrievedChunk(
                    doc_id=chunk_record.doc_id,
                    source_name=source_name,
                    chunk_text=chunk_record.chunk_text,
                    score=cosine,
                    cosine_score=cosine,
                )
            )

        scored.sort(key=lambda chunk: chunk.score, reverse=True)
        return scored[:top_k]

    async def delete_document(self, doc_id: str) -> None:
        async with self._session_factory() as db:
            await db.execute(delete(DocumentChunkRecord).where(DocumentChunkRecord.doc_id == doc_id))
            await db.commit()
