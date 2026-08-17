from __future__ import annotations

import os
from typing import Any, AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from nexus.api.main import create_app
from nexus.config.settings import get_settings
from nexus.core.provider_manager import ProviderManager
from nexus.core.providers import AIProvider
from nexus.core.router import ModelRouter
from nexus.core.types import (
    GenerationChunk,
    GenerationResult,
    ModelInfo,
    ToolCall,
    Usage,
)
from nexus.models.registry import list_models
from nexus.tools.registry import Tool, ToolRegistry, ToolResult


class _FakeCalculatorTool(Tool):
    name = "calculator"
    description = "Adds two numbers."
    parameters_schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, output=str(arguments["a"] + arguments["b"]))


class FakeLocalProvider(AIProvider):
    """Stands in for the non-tool-capable local provider — never actually
    hit by these tests since they all pin model_id to a tool-capable model,
    but ProviderManager needs a "local" entry to construct cleanly."""

    name = "fake-local"

    async def generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        return GenerationResult(
            content="local", model_used=model_id, provider_name=self.name, usage=Usage()
        )

    async def stream_generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        yield GenerationChunk(delta="local", done=True, usage=Usage())

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> bool:
        return True

    def count_tokens(self, text: str, *, model_id: str) -> int:
        return len(text)


class FakeToolCallingProvider(AIProvider):
    """Returns a tool_call for `tool_call_rounds` calls, then a final
    (non-tool-call) answer — or, if tool_call_rounds is large enough,
    never stops, to exercise the max_iterations cap."""

    name = "fake-cloud"

    def __init__(self, *, tool_call_rounds: int = 1) -> None:
        self._tool_call_rounds = tool_call_rounds
        self.generate_call_count = 0
        self.received_tools_per_call: list[list[dict[str, Any]] | None] = []

    async def generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        self.generate_call_count += 1
        self.received_tools_per_call.append(tools)
        if self.generate_call_count <= self._tool_call_rounds:
            return GenerationResult(
                content="",
                model_used=model_id,
                provider_name=self.name,
                usage=Usage(prompt_tokens=10, completion_tokens=5),
                tool_calls=[
                    ToolCall(
                        id=f"call_{self.generate_call_count}",
                        name="calculator",
                        arguments={"a": 2, "b": 3},
                    )
                ],
            )
        return GenerationResult(
            content="The answer is 5.",
            model_used=model_id,
            provider_name=self.name,
            usage=Usage(prompt_tokens=20, completion_tokens=8),
        )

    async def stream_generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        yield GenerationChunk(delta="unused", done=False)
        yield GenerationChunk(delta="", done=True, usage=Usage())

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> bool:
        return True

    def count_tokens(self, text: str, *, model_id: str) -> int:
        return len(text)


@pytest.fixture(scope="module")
def _memory_db_env(tmp_path_factory) -> str:
    db_path = str(tmp_path_factory.mktemp("nexus-tools") / "nexus.db")
    os.environ["NEXUS_MEMORY__DATABASE_PATH"] = db_path
    yield db_path
    os.environ.pop("NEXUS_MEMORY__DATABASE_PATH", None)


async def _make_client(
    _memory_db_env: str, *, tool_call_rounds: int
) -> AsyncIterator[tuple[AsyncClient, FakeToolCallingProvider]]:
    get_settings(refresh=True)
    app = create_app()

    async with app.router.lifespan_context(app):
        fake_provider = FakeToolCallingProvider(tool_call_rounds=tool_call_rounds)
        provider_manager = ProviderManager(
            {"local": FakeLocalProvider(), "openai": fake_provider}
        )
        app.state.provider_manager = provider_manager
        app.state.router = ModelRouter(provider_manager, list_models())
        app.state.tool_registry = ToolRegistry({"calculator": _FakeCalculatorTool()})

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac, fake_provider


@pytest.fixture
async def client_one_round(_memory_db_env: str):
    async for pair in _make_client(_memory_db_env, tool_call_rounds=1):
        yield pair


@pytest.fixture
async def client_exceeds_cap(_memory_db_env: str):
    async for pair in _make_client(_memory_db_env, tool_call_rounds=10):
        yield pair


@pytest.mark.asyncio
async def test_tool_call_then_final_answer_populates_tool_calls_made(client_one_round) -> None:
    client, provider = client_one_round

    response = await client.post(
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "What is 2 + 3? Use the calculator."}],
            "use_tools": True,
            "model_id": "gpt-4o-mini",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["content"] == "The answer is 5."
    assert body["tool_calls_made"] == [
        {"name": "calculator", "arguments": {"a": 2, "b": 3}, "result_summary": "5"}
    ]
    assert body["usage"]["prompt_tokens"] == 30
    assert body["usage"]["completion_tokens"] == 13
    assert provider.generate_call_count == 2
    assert all(call for call in provider.received_tools_per_call)


@pytest.mark.asyncio
async def test_use_tools_forces_a_tool_capable_provider(client_one_round) -> None:
    client, _provider = client_one_round

    response = await client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "Add 2 and 3."}], "use_tools": True},
    )

    assert response.status_code == 200
    assert response.json()["provider_name"] == "openai"


@pytest.mark.asyncio
async def test_max_iterations_cap_is_respected(client_exceeds_cap) -> None:
    client, provider = client_exceeds_cap

    response = await client.post(
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "Keep calculating forever."}],
            "use_tools": True,
            "model_id": "gpt-4o-mini",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert provider.generate_call_count == 3
    assert len(body["tool_calls_made"]) == 3
    assert "max_iterations" in body["classification_reason"]


@pytest.mark.asyncio
async def test_streaming_with_tools_returns_a_single_final_chunk(client_one_round) -> None:
    client, _provider = client_one_round

    async with client.stream(
        "POST",
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "What is 2 + 3?"}],
            "use_tools": True,
            "model_id": "gpt-4o-mini",
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines() if line]

    content_lines = [line for line in lines if '"done": false' in line]
    assert len(content_lines) == 1
    assert "The answer is 5." in content_lines[0]
    assert any('"tool_calls_made"' in line for line in lines)


@pytest.mark.asyncio
async def test_use_tools_false_never_populates_tool_calls_made(client_one_round) -> None:
    client, provider = client_one_round

    response = await client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "Hello"}], "model_id": "gpt-4o-mini"},
    )

    assert response.status_code == 200
    assert response.json()["tool_calls_made"] is None
    assert provider.received_tools_per_call == [None]
