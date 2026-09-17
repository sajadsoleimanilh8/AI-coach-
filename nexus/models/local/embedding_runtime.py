from __future__ import annotations

import httpx

from nexus.core.embeddings import EmbeddingProvider
from nexus.core.exceptions import ProviderUnavailableError
from nexus.logging_setup.logger import get_logger

logger = get_logger("models.local.ollama_embeddings")


class OllamaEmbeddingProvider(EmbeddingProvider):
    """EmbeddingProvider backed by a local Ollama instance's /api/embeddings.

    Ollama's classic embeddings endpoint takes one prompt per request (no
    batch input), so `embed()` issues one call per text — fine at RAG-chunk
    batch sizes (tens, not thousands), and keeps this adapter symmetric
    with `OllamaRuntime`'s httpx client construction pattern.
    """

    def __init__(
        self,
        base_url: str,
        *,
        model_id: str,
        timeout_seconds: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._timeout_seconds = timeout_seconds
        self._transport = transport  # test seam: inject httpx.MockTransport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout_seconds,
            transport=self._transport,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        try:
            async with self._client() as client:
                for text in texts:
                    response = await client.post(
                        "/api/embeddings", json={"model": self._model_id, "prompt": text}
                    )
                    response.raise_for_status()
                    data = response.json()
                    vectors.append(data.get("embedding", []))
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                f"Could not reach Ollama at {self._base_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(f"Ollama at {self._base_url} timed out: {exc}") from exc
        return vectors

    async def health_check(self) -> bool:
        try:
            async with self._client() as client:
                response = await client.get("/api/tags")
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
