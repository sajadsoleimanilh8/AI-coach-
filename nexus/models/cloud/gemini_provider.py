from __future__ import annotations

import json
import time
import uuid
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

logger = get_logger("models.cloud.gemini")

_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Gemini ships no lightweight offline tokenizer, and its countTokens endpoint
# is a separate network round trip on a hot path — one extra call per request
# just to estimate. Same ~4 chars/token heuristic AnthropicProvider uses; real
# numbers always come back on the response in usageMetadata.
_CHARS_PER_TOKEN_ESTIMATE = 4


class GeminiProvider(AIProvider):
    """AIProvider backed by the Google Gemini generateContent REST API."""

    name = "gemini"

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
            raise ProviderUnavailableError("GEMINI_API_KEY is not configured.")
        return self._api_key

    def _client(self, api_key: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout_seconds,
            transport=self._transport,
            # Gemini also accepts ?key=, but a query string ends up in proxy
            # logs, crash dumps and error text; a header does not.
            headers={"x-goog-api-key": api_key, "content-type": "application/json"},
        )

    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Splits NEXUS messages into Gemini's systemInstruction + contents."""
        system_parts = [m.content for m in messages if m.role == "system"]
        contents = [
            # Gemini names the assistant turn "model"; every other provider in
            # NEXUS (and NEXUS's own Message.role) calls it "assistant". The
            # rename has to happen here so nothing outside this adapter knows.
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        # systemInstruction is a single Content, not a list, so multiple system
        # messages have to be flattened into one.
        system_instruction = (
            {"parts": [{"text": "\n".join(system_parts)}]} if system_parts else None
        )
        return system_instruction, contents

    def _build_payload(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        system_instruction, contents = self._split_system(messages)
        generation_config: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens

        payload: dict[str, Any] = {"contents": contents, "generationConfig": generation_config}
        if system_instruction is not None:
            payload["systemInstruction"] = system_instruction
        if tools:
            payload["tools"] = _to_gemini_tools(tools)
        return payload

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

    @staticmethod
    def _raise_for_status(status_code: int, model_id: str) -> None:
        if status_code == 404:
            raise ModelNotFoundError(f"Model '{model_id}' not found on Gemini.")
        if status_code in (401, 403):
            raise ProviderUnavailableError("Gemini API rejected the configured API key.")
        if status_code >= 500:
            raise ProviderUnavailableError(f"Gemini API returned {status_code}.")

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
        payload = self._build_payload(
            messages, temperature=temperature, max_tokens=max_tokens, tools=tools
        )

        start = time.monotonic()
        try:
            async with self._client(api_key) as client:
                response = await client.post(f"/models/{model_id}:generateContent", json=payload)
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(f"Could not reach Gemini API: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(f"Gemini API timed out: {exc}") from exc
        latency = time.monotonic() - start

        self._raise_for_status(response.status_code, model_id)
        response.raise_for_status()
        data = response.json()

        candidates = data.get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        text = "".join(part.get("text", "") for part in parts if "text" in part)
        usage = data.get("usageMetadata", {})
        return GenerationResult(
            content=text,
            model_used=data.get("modelVersion", model_id),
            provider_name=self.name,
            usage=Usage(
                prompt_tokens=usage.get("promptTokenCount", 0),
                completion_tokens=usage.get("candidatesTokenCount", 0),
            ),
            finish_reason=candidates[0].get("finishReason") if candidates else None,
            latency_seconds=latency,
            raw=data,
            tool_calls=_parse_gemini_tool_calls(parts),
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
        # Phase 5 scope boundary, same as the OpenAI/Anthropic adapters:
        # chat.py's tool-calling loop only ever calls non-streaming
        # generate(), so streamed functionCall parts are accepted here for
        # API completeness but not accumulated — GenerationChunk carries
        # only text deltas.
        api_key = self._require_api_key()
        self._check_context_window(messages, model_id)
        payload = self._build_payload(
            messages, temperature=temperature, max_tokens=max_tokens, tools=tools
        )

        try:
                    # Gemini marks the end with finishReason on the last
                    # candidate rather than with a sentinel event or [DONE].
            async with self._client(api_key) as client, client.stream(
                "POST",
                f"/models/{model_id}:streamGenerateContent",
                params={"alt": "sse"},
                json=payload,
            ) as response:
                self._raise_for_status(response.status_code, model_id)
                response.raise_for_status()

                async for line in response.aiter_lines():
                    chunk = _parse_gemini_stream_line(line)
                    if chunk is not None:
                        yield chunk
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(f"Could not reach Gemini API: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(f"Gemini API timed out: {exc}") from exc

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
        return max(1, len(text) // _CHARS_PER_TOKEN_ESTIMATE)


def _to_gemini_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Gemini nests every declaration under a single tools[0].functionDeclarations
    # array, where the canonical (OpenAI-style) shape is a flat list of tools.
    return [
        {
            "functionDeclarations": [
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                }
                for tool in tools
            ]
        }
    ]


def _parse_gemini_tool_calls(parts: list[dict[str, Any]]) -> list[ToolCall] | None:
    calls = []
    for part in parts:
        function_call = part.get("functionCall")
        if not function_call:
            continue
        calls.append(
            ToolCall(
                # Gemini returns no call id, but ToolCall.id is how the rest
                # of NEXUS correlates a result back to its request. Synthesize
                # one so that contract holds for every provider.
                id=uuid.uuid4().hex,
                name=function_call.get("name", ""),
                arguments=function_call.get("args") or {},
            )
        )
    return calls or None


def _parse_gemini_stream_line(line: str) -> GenerationChunk | None:
    if not line.startswith("data:"):
        return None
    data_str = line[len("data:") :].strip()
    if not data_str:
        return None
    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        return None

    candidates = data.get("candidates") or []
    if not candidates:
        return None
    candidate = candidates[0]
    parts = candidate.get("content", {}).get("parts", [])
    delta = "".join(part.get("text", "") for part in parts if "text" in part)

    finish_reason = candidate.get("finishReason")
    if finish_reason:
        usage = data.get("usageMetadata", {})
        return GenerationChunk(
            delta=delta,
            done=True,
            usage=Usage(
                prompt_tokens=usage.get("promptTokenCount", 0),
                completion_tokens=usage.get("candidatesTokenCount", 0),
            ),
            finish_reason=finish_reason,
        )
    if not delta:
        return None
    return GenerationChunk(delta=delta, done=False)
