from __future__ import annotations

import time
from typing import Any, AsyncIterator

import httpx

from nexus.core.exceptions import (
    ContextLengthExceededError,
    ModelNotFoundError,
    ProviderUnavailableError,
)
from nexus.core.providers import AIProvider
from nexus.core.types import GenerationChunk, GenerationResult, Message, ModelInfo, Usage
from nexus.logging_setup.logger import get_logger
from nexus.models.registry import get_model

logger = get_logger("models.local.ollama")

_CHARS_PER_TOKEN_ESTIMATE = 4


class OllamaRuntime(AIProvider):
    """AIProvider backed by a local Ollama instance."""

    name = "ollama"

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout_seconds,
            transport=self._transport,
        )

    @staticmethod
    def _to_ollama_messages(messages: list[Message]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def _build_options(self, temperature: float, max_tokens: int | None) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        return options

    def _check_context_window(self, messages: list[Message], model_id: str) -> None:
        model_info = get_model(model_id)
        if model_info is None:
            return
        total_chars = sum(len(m.content) for m in messages)
        estimated_tokens = total_chars // _CHARS_PER_TOKEN_ESTIMATE
        if estimated_tokens > model_info.context_window:
            raise ContextLengthExceededError(
                f"Estimated {estimated_tokens} tokens exceeds {model_id}'s "
                f"context window of {model_info.context_window}."
            )

    async def generate(
        self,
        messages: list[Message],
        *,
        model_id: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> GenerationResult:
        self._check_context_window(messages, model_id)
        payload = {
            "model": model_id,
            "messages": self._to_ollama_messages(messages),
            "stream": False,
            "options": self._build_options(temperature, max_tokens),
        }

        start = time.monotonic()
        try:
            async with self._client() as client:
                response = await client.post("/api/chat", json=payload)
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                f"Could not reach Ollama at {self._base_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(
                f"Ollama at {self._base_url} timed out: {exc}"
            ) from exc
        latency = time.monotonic() - start

        if response.status_code == 404:
            raise ModelNotFoundError(f"Model '{model_id}' not found on Ollama backend.")
        response.raise_for_status()
        data = response.json()

        return GenerationResult(
            content=data.get("message", {}).get("content", ""),
            model_used=model_id,
            provider_name=self.name,
            usage=Usage(
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
            ),
            finish_reason="stop" if data.get("done") else None,
            latency_seconds=latency,
            raw=data,
        )

    async def stream_generate(
        self,
        messages: list[Message],
        *,
        model_id: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[GenerationChunk]:
        self._check_context_window(messages, model_id)
        payload = {
            "model": model_id,
            "messages": self._to_ollama_messages(messages),
            "stream": True,
            "options": self._build_options(temperature, max_tokens),
        }

        try:
            async with self._client() as client:
                async with client.stream("POST", "/api/chat", json=payload) as response:
                    if response.status_code == 404:
                        raise ModelNotFoundError(
                            f"Model '{model_id}' not found on Ollama backend."
                        )
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        chunk = _parse_ollama_stream_line(line)
                        if chunk is not None:
                            yield chunk
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                f"Could not reach Ollama at {self._base_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(
                f"Ollama at {self._base_url} timed out: {exc}"
            ) from exc

    async def list_models(self) -> list[ModelInfo]:
        try:
            async with self._client() as client:
                response = await client.get("/api/tags")
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                f"Could not reach Ollama at {self._base_url}: {exc}"
            ) from exc
        response.raise_for_status()
        data = response.json()

        models: list[ModelInfo] = []
        for entry in data.get("models", []):
            model_id = entry.get("name") or entry.get("model")
            if not model_id:
                continue
            registered = get_model(model_id)
            models.append(
                registered
                if registered is not None
                else ModelInfo(id=model_id, provider=self.name)
            )
        return models

    async def health_check(self) -> bool:
        try:
            async with self._client() as client:
                response = await client.get("/api/tags")
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def count_tokens(self, text: str, *, model_id: str) -> int:
        return max(1, len(text) // _CHARS_PER_TOKEN_ESTIMATE)


def _parse_ollama_stream_line(line: str) -> GenerationChunk | None:
    import json

    data = json.loads(line)
    if data.get("done"):
        return GenerationChunk(
            delta="",
            done=True,
            usage=Usage(
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
            ),
            finish_reason="stop",
        )
    delta = data.get("message", {}).get("content", "")
    if not delta:
        return None
    return GenerationChunk(delta=delta, done=False)
