from __future__ import annotations

import httpx

from nexus.core.embeddings import EmbeddingProvider
from nexus.core.exceptions import ProviderUnavailableError
from nexus.logging_setup.logger import get_logger

logger = get_logger("models.cloud.openai_embeddings")

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_MODEL = "text-embedding-3-small"


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """EmbeddingProvider backed by OpenAI's /v1/embeddings REST API."""

    def __init__(
        self,
        api_key: str | None,
        *,
        model_id: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_BASE_URL,
        timeout_seconds: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    def _require_api_key(self) -> str:
        if not self._api_key:
            raise ProviderUnavailableError("OPENAI_API_KEY is not configured.")
        return self._api_key

    def _client(self, api_key: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout_seconds,
            transport=self._transport,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        api_key = self._require_api_key()
        try:
            async with self._client(api_key) as client:
                response = await client.post(
                    "/embeddings", json={"model": self._model_id, "input": texts}
                )
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(f"Could not reach OpenAI API: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(f"OpenAI API timed out: {exc}") from exc

        if response.status_code == 401:
            raise ProviderUnavailableError("OpenAI API rejected the configured API key.")
        if response.status_code >= 500:
            raise ProviderUnavailableError(f"OpenAI API returned {response.status_code}.")
        response.raise_for_status()
        data = response.json()

        ordered = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
        return [item.get("embedding", []) for item in ordered]

    async def health_check(self) -> bool:
        if not self._api_key:
            return False
        try:
            async with self._client(self._api_key) as client:
                response = await client.get("/models")
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
