from __future__ import annotations

import pytest

from nexus.core.exceptions import ModelNotFoundError
from nexus.core.provider_manager import ProviderManager
from nexus.core.providers import AIProvider
from nexus.core.router import ModelRouter
from nexus.core.types import GenerationResult, ModelInfo, RoutingPolicy, TaskType, Usage


class _FakeProvider(AIProvider):
    def __init__(self, name: str) -> None:
        self.name = name

    async def generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        return GenerationResult(
            content="ok", model_used=model_id, provider_name=self.name, usage=Usage()
        )

    async def stream_generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        return
        yield  # pragma: no cover - makes this an async generator

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> bool:
        return True

    def count_tokens(self, text: str, *, model_id: str) -> int:
        return len(text)


def _fake_registry() -> list[ModelInfo]:
    return [
        ModelInfo(
            id="local-model",
            provider="local",
            kind="chat",
            context_window=8192,
            capabilities={"reasoning": 0.2},
        ),
        ModelInfo(
            id="cheap-cloud",
            provider="openai",
            kind="chat",
            context_window=128000,
            capabilities={"reasoning": 0.75},
            cost_per_1k_input_tokens=0.0004,
            cost_per_1k_output_tokens=0.0006,
        ),
        ModelInfo(
            id="quality-cloud",
            provider="anthropic",
            kind="chat",
            context_window=200000,
            capabilities={"reasoning": 0.95},
            cost_per_1k_input_tokens=0.008,
            cost_per_1k_output_tokens=0.012,
        ),
    ]


def _router() -> ModelRouter:
    provider_manager = ProviderManager(
        {
            "local": _FakeProvider("local"),
            "openai": _FakeProvider("openai"),
            "anthropic": _FakeProvider("anthropic"),
        }
    )
    return ModelRouter(provider_manager, _fake_registry())


def test_local_only_excludes_cloud_candidates() -> None:
    decision = _router().route(
        task_type=TaskType.COMPLEX_REASONING, policy=RoutingPolicy.LOCAL_ONLY
    )
    assert decision.model_id == "local-model"
    assert decision.provider_name == "local"


def test_max_quality_picks_the_highest_capability_model() -> None:
    decision = _router().route(
        task_type=TaskType.COMPLEX_REASONING, policy=RoutingPolicy.MAX_QUALITY
    )
    assert decision.model_id == "quality-cloud"


def test_low_cost_picks_the_free_local_model() -> None:
    decision = _router().route(task_type=TaskType.COMPLEX_REASONING, policy=RoutingPolicy.LOW_COST)
    assert decision.model_id == "local-model"


def test_low_latency_prefers_local_first() -> None:
    decision = _router().route(
        task_type=TaskType.COMPLEX_REASONING, policy=RoutingPolicy.LOW_LATENCY
    )
    assert decision.model_id == "local-model"


def test_balanced_weighs_capability_cost_and_locality() -> None:
    decision = _router().route(task_type=TaskType.COMPLEX_REASONING, policy=RoutingPolicy.BALANCED)
    assert decision.model_id == "cheap-cloud"
    assert "BALANCED policy" in decision.reason


def test_explicit_model_id_always_wins() -> None:
    decision = _router().route(requested_model_id="gpt-4o")
    assert decision.model_id == "gpt-4o"
    assert decision.provider_name == "openai"
    assert "Explicit model_id" in decision.reason


def _registry_with_mixed_tool_support() -> list[ModelInfo]:
    return [
        ModelInfo(
            id="strong-no-tools",
            provider="local",
            kind="chat",
            capabilities={"reasoning": 0.9},
            supports_tool_calling=False,
        ),
        ModelInfo(
            id="weaker-with-tools",
            provider="openai",
            kind="chat",
            capabilities={"reasoning": 0.5},
            supports_tool_calling=True,
        ),
    ]


def test_require_tool_calling_filters_out_non_capable_models() -> None:
    provider_manager = ProviderManager(
        {"local": _FakeProvider("local"), "openai": _FakeProvider("openai")}
    )
    router = ModelRouter(provider_manager, _registry_with_mixed_tool_support())

    decision = router.route(
        task_type=TaskType.COMPLEX_REASONING,
        policy=RoutingPolicy.MAX_QUALITY,
        require_tool_calling=True,
    )

    assert decision.model_id == "weaker-with-tools"


def test_require_tool_calling_false_is_unaffected() -> None:
    provider_manager = ProviderManager(
        {"local": _FakeProvider("local"), "openai": _FakeProvider("openai")}
    )
    router = ModelRouter(provider_manager, _registry_with_mixed_tool_support())

    decision = router.route(task_type=TaskType.COMPLEX_REASONING, policy=RoutingPolicy.MAX_QUALITY)

    assert decision.model_id == "strong-no-tools"


def test_require_tool_calling_raises_when_no_candidate_supports_tools() -> None:
    provider_manager = ProviderManager({"local": _FakeProvider("local")})
    registry = [
        ModelInfo(
            id="local-model",
            provider="local",
            kind="chat",
            capabilities={"reasoning": 0.9},
            supports_tool_calling=False,
        )
    ]
    router = ModelRouter(provider_manager, registry)

    with pytest.raises(ModelNotFoundError):
        router.route(task_type=TaskType.GENERAL, require_tool_calling=True)
