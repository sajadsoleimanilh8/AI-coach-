from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Common interface for text-embedding backends.

    Deliberately separate from `AIProvider` — not every chat provider
    offers embeddings (Anthropic doesn't), and forcing an abstractmethod
    onto `AIProvider` for that would break its substitutability for
    providers that only do chat.
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input text."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the backend is reachable and usable."""
