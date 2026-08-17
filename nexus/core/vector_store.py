from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    doc_id: str
    source_name: str
    chunk_text: str
    score: float
    cosine_score: float


class VectorStore(ABC):
    """Common interface for chunk storage + nearest-neighbor search."""

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
