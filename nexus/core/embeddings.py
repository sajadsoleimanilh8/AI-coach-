from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Common interface for text-embedding backends."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input text."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the backend is reachable and usable."""
