from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    doc_id: str
    source_name: str
    chunk_text: str
    score: float  # final blended score after reranking
    cosine_score: float  # raw similarity, kept for explainability


class VectorStore(ABC):
    """Common interface for chunk storage + nearest-neighbor search.

    Mirrors `MemoryStore`'s pattern: callers depend only on this ABC, so
    today's brute-force SQLite implementation can be swapped for a real
    vector DB (FAISS, pgvector, ...) later without touching RagService or
    chat.py.
    """

    async def init(self) -> None:  # noqa: B027 - deliberately optional, not abstract
        """Prepare storage (e.g. create tables) before first use.

        Part of the interface because nexus/api/services.py calls it on every
        store at startup. A no-op by default so a store that needs no setup,
        and the test fakes, do not have to implement it.
        """

    @abstractmethod
    async def add_chunks(
        self, *, doc_id: str, chunks: list[str], embeddings: list[list[float]]
    ) -> None:
        """Persist chunk texts + their embeddings for a document."""

    @abstractmethod
    async def search(
        self, query_embedding: list[float], *, top_k: int, user_id: str
    ) -> list[RetrievedChunk]:
        """Return the top_k chunks nearest to query_embedding, scoped to user_id."""

    @abstractmethod
    async def delete_document(self, doc_id: str) -> None:
        """Remove all chunks belonging to a document."""
