from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from nexus.core.embeddings import EmbeddingProvider
from nexus.core.vector_store import RetrievedChunk, VectorStore
from nexus.logging_setup.logger import get_logger
from nexus.memory.storage import DocumentRecord, init_db, make_session_factory
from nexus.rag.chunking import chunk_text
from nexus.rag.graph import EntityExtractor, GraphRetriever, GraphStore
from nexus.rag.parsers import parse_document
from nexus.rag.reranker import rerank

logger = get_logger("rag.service")

_DEFAULT_CHUNK_SIZE = 800
_DEFAULT_CHUNK_OVERLAP = 150


@dataclass
class DocumentSummary:
    doc_id: str
    source_name: str
    source_type: str
    chunk_count: int
    ingested_at: float


class RagService:
    """Owns the ingest/retrieve pipeline, composing EmbeddingProvider +
    VectorStore + storage.py's DocumentRecord/DocumentChunkRecord.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        *,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
        entity_extractor: EntityExtractor | None = None,
        graph_store: GraphStore | None = None,
        graph_retriever: GraphRetriever | None = None,
        graph_enabled: bool = False,
        graph_max_hops: int = 2,
        graph_max_nodes: int = 20,
    ) -> None:
        self._engine = engine
        self._session_factory: async_sessionmaker = make_session_factory(engine)
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._entity_extractor = entity_extractor
        self._graph_store = graph_store
        self._graph_retriever = graph_retriever
        self._graph_enabled = graph_enabled
        self._graph_max_hops = graph_max_hops
        self._graph_max_nodes = graph_max_nodes

    async def init(self) -> None:
        await init_db(self._engine)

    async def ingest(
        self, *, user_id: str, source_name: str, source_type: str, raw_bytes: bytes
    ) -> DocumentSummary:
        text = await parse_document(raw_bytes, source_type)
        chunks = chunk_text(text, chunk_size=self._chunk_size, overlap=self._chunk_overlap)
        embeddings = await self._embedding_provider.embed(chunks) if chunks else []

        doc_id = uuid.uuid4().hex
        now = time.time()
        async with self._session_factory() as db:
            db.add(
                DocumentRecord(
                    id=doc_id,
                    user_id=user_id,
                    source_name=source_name,
                    source_type=source_type,
                    chunk_count=len(chunks),
                    ingested_at=now,
                )
            )
            await db.commit()

        if chunks:
            await self._vector_store.add_chunks(doc_id=doc_id, chunks=chunks, embeddings=embeddings)
            await self._index_entities(user_id=user_id, doc_id=doc_id, chunks=chunks)

        return DocumentSummary(
            doc_id=doc_id,
            source_name=source_name,
            source_type=source_type,
            chunk_count=len(chunks),
            ingested_at=now,
        )

    async def _index_entities(self, *, user_id: str, doc_id: str, chunks: list[str]) -> None:
        if not self._graph_enabled or self._entity_extractor is None:
            return
        try:
            triples, _usage = await self._entity_extractor.extract(chunks)
        except Exception as exc:  # noqa: BLE001 - entity extraction must never fail an ingest
            logger.warning("entity extraction failed for doc=%s: %s", doc_id, exc)
            return
        if triples and self._graph_store is not None:
            await self._graph_store.add_triples(user_id=user_id, doc_id=doc_id, triples=triples)

    async def retrieve(
        self, *, user_id: str, query: str, top_k: int = 5
    ) -> list[RetrievedChunk]:
        """Vector retrieve -> optional graph expand -> rerank the combined
        set with the EXISTING reranker. The rerank runs over both sets
        together so a graph-found chunk has to earn its place against the
        vector hits rather than being appended to them."""
        [query_embedding] = await self._embedding_provider.embed([query])
        candidates = await self._vector_store.search(query_embedding, top_k=top_k, user_id=user_id)

        expanded = await self._graph_expand(user_id=user_id, seeds=candidates)
        return rerank(query, candidates + expanded)[:top_k]

    async def _graph_expand(
        self, *, user_id: str, seeds: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        if not self._graph_enabled or self._graph_retriever is None or not seeds:
            return []
        try:
            expansion = await self._graph_retriever.expand(
                seeds,
                user_id=user_id,
                max_hops=self._graph_max_hops,
                max_nodes=self._graph_max_nodes,
            )
        except Exception as exc:  # noqa: BLE001 - expansion is additive, never load-bearing
            logger.warning("graph expansion failed for user=%s: %s", user_id, exc)
            return []
        return expansion.chunks

    async def list_documents(self, user_id: str) -> list[DocumentSummary]:
        async with self._session_factory() as db:
            result = await db.execute(
                select(DocumentRecord).where(DocumentRecord.user_id == user_id)
            )
            rows = result.scalars().all()
        return [
            DocumentSummary(
                doc_id=row.id,
                source_name=row.source_name,
                source_type=row.source_type,
                chunk_count=row.chunk_count,
                ingested_at=row.ingested_at,
            )
            for row in rows
        ]

    async def delete_document(self, user_id: str, doc_id: str) -> None:
        async with self._session_factory() as db:
            result = await db.execute(
                select(DocumentRecord).where(
                    DocumentRecord.id == doc_id, DocumentRecord.user_id == user_id
                )
            )
            if result.scalar_one_or_none() is None:
                return
            await db.execute(delete(DocumentRecord).where(DocumentRecord.id == doc_id))
            await db.commit()
        await self._vector_store.delete_document(doc_id)
        if self._graph_store is not None:
            await self._graph_store.delete_document(doc_id)
