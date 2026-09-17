from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from nexus.core.exceptions import (
    ContextLengthExceededError,
    ModelNotFoundError,
    ProviderUnavailableError,
)
from nexus.core.providers import AIProvider
from nexus.core.types import GenerationChunk, GenerationResult, Message, ModelInfo, ToolCall, Usage
from nexus.logging_setup.logger import get_logger
from nexus.models.registry import get_model
from nexus.models.registry import list_models as registry_list_models

logger = get_logger("models.cloud.anthropic")

_DEFAULT_BASE_URL = "https://api.anthropic.com"
_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 4096

# Anthropic doesn't ship a lightweight, dependency-free tokenizer, so token
# counts are estimated with the common ~4 chars/token heuristic (same
# approximation OllamaRuntime uses). This is a rough estimate, not an exact
# count — real usage figures always come from the API response's `usage`.
_CHARS_PER_TOKEN_ESTIMATE = 4


class AnthropicProvider(AIProvider):
    """AIProvider backed by the Anthropic Messages REST API."""

    name = "anthropic"

    def __init__(
        self,
        api_key: str | None,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        timeout_seconds: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._transport = transport  # test seam: inject httpx.MockTransport

    def _require_api_key(self) -> str:
        if not self._api_key:
            raise ProviderUnavailableError("ANTHROPIC_API_KEY is not configured.")
        return self._api_key

    def _client(self, api_key: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout_seconds,
            transport=self._transport,
            headers={
                "x-api-key": api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
        )

    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str | None, list[dict[str, str]]]:
        system_parts = [m.content for m in messages if m.role == "system"]
        chat_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        system_text = "\n".join(system_parts) if system_parts else None
        return system_text, chat_messages

    def _check_context_window(self, messages: list[Message], model_id: str) -> None:
        model_info = get_model(model_id)
        if model_info is None:
            return
        estimated_tokens = sum(self.count_tokens(m.content, model_id=model_id) for m in messages)
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
        api_key = self._require_api_key()
        self._check_context_window(messages, model_id)
        system_text, chat_messages = self._split_system(messages)
        payload: dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens or _DEFAULT_MAX_TOKENS,
            "temperature": temperature,
            "messages": chat_messages,
        }
        if system_text:
            payload["system"] = system_text
        if tools:
            payload["tools"] = _to_anthropic_tools(tools)

        start = time.monotonic()
        try:
            async with self._client(api_key) as client:
                response = await client.post("/v1/messages", json=payload)
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(f"Could not reach Anthropic API: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(f"Anthropic API timed out: {exc}") from exc
        latency = time.monotonic() - start

        if response.status_code == 404:
            raise ModelNotFoundError(f"Model '{model_id}' not found on Anthropic.")
        if response.status_code == 401:
            raise ProviderUnavailableError("Anthropic API rejected the configured API key.")
        if response.status_code >= 500:
            raise ProviderUnavailableError(f"Anthropic API returned {response.status_code}.")
        response.raise_for_status()
        data = response.json()

        content_blocks = data.get("content", [])
        text = "".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        )
        usage = data.get("usage", {})
        return GenerationResult(
            content=text,
            model_used=data.get("model", model_id),
            provider_name=self.name,
            usage=Usage(
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
            ),
            finish_reason=data.get("stop_reason"),
            latency_seconds=latency,
            raw=data,
            tool_calls=_parse_anthropic_tool_calls(content_blocks),
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
        # Phase 5 scope boundary: chat.py's tool-calling loop only ever
        # calls non-streaming generate() (see chat.py's docstring on why),
        # so streamed tool_use content-block deltas are accepted here for
        # API completeness but not accumulated — GenerationChunk has no
        # tool_calls field to carry them, only plain text deltas.
        api_key = self._require_api_key()
        self._check_context_window(messages, model_id)
        system_text, chat_messages = self._split_system(messages)
        payload: dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens or _DEFAULT_MAX_TOKENS,
            "temperature": temperature,
            "messages": chat_messages,
            "stream": True,
        }
        if system_text:
            payload["system"] = system_text
        if tools:
            payload["tools"] = _to_anthropic_tools(tools)

        try:
            async with self._client(api_key) as client:
                async with client.stream("POST", "/v1/messages", json=payload) as response:
                    if response.status_code == 404:
                        raise ModelNotFoundError(f"Model '{model_id}' not found on Anthropic.")
                    if response.status_code == 401:
                        raise ProviderUnavailableError(
                            "Anthropic API rejected the configured API key."
                        )
                    response.raise_for_status()

                    prompt_tokens = 0
                    current_event: str | None = None
                    async for line in response.aiter_lines():
                        if line.startswith("event:"):
                            current_event = line[len("event:") :].strip()
                            continue
                        if not line.startswith("data:"):
                            continue
                        data_str = line[len("data:") :].strip()
                        if not data_str:
                            continue
                        data = json.loads(data_str)

                        if current_event == "message_start":
                            prompt_tokens = (
                                data.get("message", {}).get("usage", {}).get("input_tokens", 0)
                            )
                            continue

                        chunk = _parse_anthropic_event(current_event, data, prompt_tokens)
                        if chunk is not None:
                            yield chunk
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(f"Could not reach Anthropic API: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(f"Anthropic API timed out: {exc}") from exc

    async def list_models(self) -> list[ModelInfo]:
        return [model for model in registry_list_models() if model.provider == self.name]

    async def health_check(self) -> bool:
        if not self._api_key:
            return False
        try:
            async with self._client(self._api_key) as client:
                response = await client.get("/v1/models")
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def count_tokens(self, text: str, *, model_id: str) -> int:
        return max(1, len(text) // _CHARS_PER_TOKEN_ESTIMATE)


def _to_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Anthropic's tool schema uses "input_schema" where the canonical
    # (OpenAI-style) shape uses "parameters" — everything else lines up.
    return [
        {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": tool.get("parameters", {}),
        }
        for tool in tools
    ]


def _parse_anthropic_tool_calls(content_blocks: list[dict[str, Any]]) -> list[ToolCall] | None:
    calls = [
        ToolCall(id=block.get("id", ""), name=block.get("name", ""), arguments=block.get("input") or {})
        for block in content_blocks
        if block.get("type") == "tool_use"
    ]
    return calls or None


def _parse_anthropic_event(
    event: str | None, data: dict[str, Any], prompt_tokens: int
) -> GenerationChunk | None:
    if event == "content_block_delta":
        text = data.get("delta", {}).get("text", "")
        if not text:
            return None
        return GenerationChunk(delta=text, done=False)
    if event == "message_delta":
        usage = data.get("usage", {})
        stop_reason = data.get("delta", {}).get("stop_reason")
        return GenerationChunk(
            delta="",
            done=True,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=usage.get("output_tokens", 0),
            ),
            finish_reason=stop_reason,
        )
    return None
