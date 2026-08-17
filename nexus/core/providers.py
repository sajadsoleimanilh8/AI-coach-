from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from nexus.core.types import ChunkStream, GenerationResult, Message, ModelInfo


class AIProvider(ABC):
    """Common interface every model backend (local or cloud) must implement."""

    name: str

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        *,
        model_id: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> GenerationResult:
        """Produce a single, complete response."""

    @abstractmethod
    def stream_generate(
        self,
        messages: list[Message],
        *,
        model_id: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChunkStream:
        """Produce a response as an async stream of chunks."""

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """List models currently available on this provider's backend."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the backend is reachable and usable."""

    @abstractmethod
    def count_tokens(self, text: str, *, model_id: str) -> int:
        """Estimate the token count of `text` for the given model."""
