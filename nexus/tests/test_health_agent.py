from __future__ import annotations

from typing import Any

import pytest

from nexus.agents.base import AgentContext
from nexus.agents.health import HealthAgent
from nexus.agents.runtime import AgentRuntime
from nexus.core.providers import AIProvider
from nexus.core.router import RoutingDecision
from nexus.core.types import GenerationChunk, GenerationResult, ModelInfo, RoutingPolicy, Usage
from nexus.health.safety import MEDICAL_DISCLAIMER
from nexus.tools.registry import Tool, ToolRegistry, ToolResult


class _FakeFilesTool(Tool):
    name = "files"
    description = "Fake."
    parameters_schema = {"type": "object", "properties": {}}

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, output="")


class _FakeProvider(AIProvider):
    name = "fake-local"

    def __init__(self, *, content: str) -> None:
        self._content = content

    async def generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        return GenerationResult(
            content=self._content, model_used=model_id, provider_name=self.name, usage=Usage()
        )

    async def stream_generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        yield GenerationChunk(delta="unused", done=True, usage=Usage())

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> bool:
        return True

    def count_tokens(self, text: str, *, model_id: str) -> int:
        return len(text)


class _FakeRouter:
    def __init__(self, provider: AIProvider) -> None:
        self._provider = provider
        self.received_kwargs: dict[str, Any] | None = None

    async def route_with_failover(self, **kwargs):
        self.received_kwargs = kwargs
        decision = RoutingDecision(
            provider_name=self._provider.name,
            model_id="fake-model",
            task_type=kwargs.get("task_type"),
            policy=kwargs.get("policy") or RoutingPolicy.BALANCED,
            reason="fake routing",
        )
        return decision, self._provider


def _context(extra: dict[str, Any] | None = None) -> AgentContext:
    return AgentContext(
        goal="How am I doing?",
        user_id="u1",
        session_id=None,
        rag_service=None,
        long_term_memory=None,
        personal_state=None,
        extra=extra or {},
    )


def _runtime(provider: AIProvider) -> tuple[AgentRuntime, _FakeRouter]:
    router = _FakeRouter(provider)
    tool_registry = ToolRegistry({"files": _FakeFilesTool()})
    return AgentRuntime(router, tool_registry, max_iterations=3), router


@pytest.mark.asyncio
async def test_diagnostic_output_is_rewritten_by_the_safety_filter() -> None:
    provider = _FakeProvider(content="You have an anxiety disorder based on these readings.")
    runtime, _router = _runtime(provider)

    result = await runtime.run(HealthAgent(), _context())

    assert result.final_answer != "You have an anxiety disorder based on these readings."
    assert MEDICAL_DISCLAIMER in result.final_answer


@pytest.mark.asyncio
async def test_medication_dosage_output_is_rewritten() -> None:
    provider = _FakeProvider(content="Take 500mg of ibuprofen every six hours.")
    runtime, _router = _runtime(provider)

    result = await runtime.run(HealthAgent(), _context())

    assert "500mg" not in result.final_answer
    assert MEDICAL_DISCLAIMER in result.final_answer


@pytest.mark.asyncio
async def test_safe_output_passes_through_unmodified() -> None:
    safe_text = "Your recovery has been below baseline for the last several days."
    provider = _FakeProvider(content=safe_text)
    runtime, _router = _runtime(provider)

    result = await runtime.run(HealthAgent(), _context())

    assert result.final_answer == safe_text


@pytest.mark.asyncio
async def test_health_agent_routes_local_only() -> None:
    provider = _FakeProvider(content="ok")
    runtime, router = _runtime(provider)

    await runtime.run(HealthAgent(), _context())

    assert router.received_kwargs["policy"] == RoutingPolicy.LOCAL_ONLY
    assert router.received_kwargs["require_tool_calling"] is True


@pytest.mark.asyncio
async def test_prepare_context_injects_the_structured_analysis() -> None:
    class _FakePattern:
        name = "recovery_debt"
        severity = "notable"
        dimensions_involved = ["physical.recovery"]
        confidence = 0.8
        sample_size = 6
        explanation = "Recovery has been low."

    class _FakeAnalysis:
        data_sufficiency = "adequate"
        patterns = [_FakePattern()]
        scorecard = {"physical": 0.4}

    class _FakeAnalyzer:
        async def analyze(self, user_id: str):
            return _FakeAnalysis()

    provider = _FakeProvider(content="ok")
    runtime, _router = _runtime(provider)

    context = _context({"health_analyzer": _FakeAnalyzer()})
    agent = HealthAgent()
    messages = await agent.prepare_context(context)

    assert len(messages) == 1
    assert "recovery_debt" in messages[0].content
    assert "physical.recovery" in messages[0].content


@pytest.mark.asyncio
async def test_prepare_context_returns_empty_without_an_analyzer() -> None:
    agent = HealthAgent()
    messages = await agent.prepare_context(_context())
    assert messages == []
