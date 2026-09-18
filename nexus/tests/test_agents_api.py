from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from nexus.agents.runtime import AgentRuntime
from nexus.api.main import create_app
from nexus.config.settings import get_settings
from nexus.core.provider_manager import ProviderManager
from nexus.core.providers import AIProvider
from nexus.core.router import ModelRouter
from nexus.core.types import GenerationChunk, GenerationResult, ModelInfo, ToolCall, Usage
from nexus.models.registry import list_models
from nexus.tools.registry import Tool, ToolRegistry, ToolResult


class _FakeFilesTool(Tool):
    name = "files"
    description = "Fake read-only file access."
    parameters_schema = {"type": "object", "properties": {}}

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, output="file contents")


class _FakeLocalProvider(AIProvider):
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


class _FakeAgentProvider(AIProvider):
    """Calls the "files" tool once, then gives a final answer."""

    name = "fake-cloud"

    def __init__(self) -> None:
        self.call_count = 0

    async def generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        self.call_count += 1
        if self.call_count == 1:
            return GenerationResult(
                content="Looking that up.",
                model_used=model_id,
                provider_name=self.name,
                usage=Usage(prompt_tokens=10, completion_tokens=5),
                tool_calls=[ToolCall(id="call_1", name="files", arguments={"path": "notes.txt"})],
            )
        return GenerationResult(
            content="Research complete.",
            model_used=model_id,
            provider_name=self.name,
            usage=Usage(prompt_tokens=20, completion_tokens=8),
        )

    async def stream_generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        yield GenerationChunk(delta="unused", done=True, usage=Usage())

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> bool:
        return True

    def count_tokens(self, text: str, *, model_id: str) -> int:
        return len(text)


@pytest.fixture
async def client_and_provider(
    _memory_db_env: str,
) -> AsyncIterator[tuple[AsyncClient, _FakeAgentProvider, Any]]:
    get_settings(refresh=True)
    app = create_app()

    async with app.router.lifespan_context(app):
        fake_provider = _FakeAgentProvider()
        provider_manager = ProviderManager(
            {"local": _FakeLocalProvider(), "openai": fake_provider}
        )
        app.state.services.provider_manager = provider_manager
        fake_router = ModelRouter(provider_manager, list_models())
        app.state.services.router = fake_router

        tool_registry = ToolRegistry({"files": _FakeFilesTool()})
        app.state.services.tool_registry = tool_registry
        # Lifespan already built an agent_runtime bound to the real local
        # router/tool_registry — swap it for one bound to the fakes above,
        # the same way other integration tests override app.state.services.router.
        app.state.services.agent_runtime = AgentRuntime(
            fake_router,
            tool_registry,
            max_iterations=app.state.services.settings.agents.max_iterations,
            cost_tracker=app.state.services.cost_tracker,
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac, fake_provider, app


@pytest.mark.asyncio
async def test_list_agents_returns_all_enabled_agents(
    client_and_provider: tuple[AsyncClient, _FakeAgentProvider, Any],
) -> None:
    client, _provider, _app = client_and_provider

    response = await client.get("/api/agents")

    assert response.status_code == 200
    names = {a["name"] for a in response.json()}
    assert names == {
        "research",
        "coding",
        "data",
        "planning",
        "health",
        "sports",
        "orchestrator",
        "autonomous_research",
    }
    for agent in response.json():
        assert agent["description"]
        assert isinstance(agent["allowed_tools"], list)


@pytest.mark.asyncio
async def test_list_agents_respects_the_enabled_allowlist(
    client_and_provider: tuple[AsyncClient, _FakeAgentProvider, Any],
) -> None:
    client, _provider, app = client_and_provider
    app.state.services.settings.agents.enabled = ["research"]

    response = await client.get("/api/agents")

    assert [a["name"] for a in response.json()] == ["research"]


@pytest.mark.asyncio
async def test_run_agent_returns_a_well_formed_response(
    client_and_provider: tuple[AsyncClient, _FakeAgentProvider, Any],
) -> None:
    client, provider, _app = client_and_provider

    # Uses "coding" rather than "research" deliberately — ResearchAgent's
    # prepare_context() calls the real rag_service (embeddings via Ollama),
    # which this test's fixture doesn't stub out; CodingAgent has no such
    # external dependency, so it isolates what this test actually checks:
    # the shape of a completed AgentRunResponse.
    response = await client.post(
        "/api/agents/coding/run",
        json={"goal": "What is the capital of France?", "user_id": "u1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["goal"] == "What is the capital of France?"
    assert body["final_answer"] == "Research complete."
    assert body["completed"] is True
    assert body["iterations_used"] == 2
    assert len(body["steps"]) == 1
    assert body["steps"][0]["tool_name"] == "files"
    assert body["steps"][0]["tool_output"] == "file contents"
    assert body["usage"]["prompt_tokens"] == 30
    assert body["usage"]["completion_tokens"] == 13
    assert body["model_used"] == "gpt-4o-mini"
    assert body["provider_name"] == "openai"
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_run_unknown_agent_returns_404(
    client_and_provider: tuple[AsyncClient, _FakeAgentProvider, Any],
) -> None:
    client, _provider, _app = client_and_provider

    response = await client.post(
        "/api/agents/does-not-exist/run", json={"goal": "anything", "user_id": "u1"}
    )

    assert response.status_code == 404
