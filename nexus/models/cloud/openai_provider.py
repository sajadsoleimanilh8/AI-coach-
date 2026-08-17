from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

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

logger = get_logger("models.cloud.openai")

_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIProvider(AIProvider):
    """AIProvider backed by the OpenAI Chat Completions REST API."""

    name = "openai"

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

    @staticmethod
    def _to_openai_messages(messages: list[Message]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

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
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": self._to_openai_messages(messages),
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = _to_openai_tools(tools)

        start = time.monotonic()
        try:
            async with self._client(api_key) as client:
                response = await client.post("/chat/completions", json=payload)
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(f"Could not reach OpenAI API: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(f"OpenAI API timed out: {exc}") from exc
        latency = time.monotonic() - start

        if response.status_code == 404:
            raise ModelNotFoundError(f"Model '{model_id}' not found on OpenAI.")
        if response.status_code == 401:
            raise ProviderUnavailableError("OpenAI API rejected the configured API key.")
        if response.status_code >= 500:
            raise ProviderUnavailableError(f"OpenAI API returned {response.status_code}.")
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})
        return GenerationResult(
            content=choice["message"].get("content") or "",
            model_used=data.get("model", model_id),
            provider_name=self.name,
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            ),
            finish_reason=choice.get("finish_reason"),
            latency_seconds=latency,
            raw=data,
            tool_calls=_parse_openai_tool_calls(choice["message"].get("tool_calls")),
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
        api_key = self._require_api_key()
        self._check_context_window(messages, model_id)
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": self._to_openai_messages(messages),
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = _to_openai_tools(tools)

        try:
            async with self._client(api_key) as client:
                async with client.stream("POST", "/chat/completions", json=payload) as response:
                    if response.status_code == 404:
                        raise ModelNotFoundError(f"Model '{model_id}' not found on OpenAI.")
                    if response.status_code == 401:
                        raise ProviderUnavailableError(
                            "OpenAI API rejected the configured API key."
                        )
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        chunk = _parse_openai_stream_line(line)
                        if chunk is not None:
                            yield chunk
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(f"Could not reach OpenAI API: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(f"OpenAI API timed out: {exc}") from exc

    async def list_models(self) -> list[ModelInfo]:
        return [model for model in registry_list_models() if model.provider == self.name]

    async def health_check(self) -> bool:
        if not self._api_key:
            return False
        try:
            async with self._client(self._api_key) as client:
                response = await client.get("/models")
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def count_tokens(self, text: str, *, model_id: str) -> int:
        try:
            import tiktoken

            try:
                encoding = tiktoken.encoding_for_model(model_id)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except ImportError:
            return max(1, len(text) // 4)


def _to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            },
        }
        for tool in tools
    ]


def _parse_openai_tool_calls(
    raw_tool_calls: list[dict[str, Any]] | None,
) -> list[ToolCall] | None:
    if not raw_tool_calls:
        return None
    parsed: list[ToolCall] = []
    for call in raw_tool_calls:
        function = call.get("function", {})
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        parsed.append(
            ToolCall(id=call.get("id", ""), name=function.get("name", ""), arguments=arguments)
        )
    return parsed


def _parse_openai_stream_line(line: str) -> GenerationChunk | None:
    if not line.startswith("data:"):
        return None
    data_str = line[len("data:") :].strip()
    if not data_str or data_str == "[DONE]":
        return None

    data = json.loads(data_str)
    choices = data.get("choices") or []
    if not choices:
        usage = data.get("usage")
        if usage:
            return GenerationChunk(
                delta="",
                done=True,
                usage=Usage(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                ),
                finish_reason="stop",
            )
        return None

    delta = choices[0].get("delta", {}).get("content") or ""
    if not delta:
        return None
    return GenerationChunk(delta=delta, done=False)
