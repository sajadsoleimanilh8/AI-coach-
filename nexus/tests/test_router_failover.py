from __future__ import annotations

import pytest

from nexus.core.exceptions import ProviderUnavailableError
from nexus.core.provider_manager import ProviderManager
from nexus.core.providers import AIProvider
from nexus.core.router import ModelRouter
from nexus.core.types import GenerationResult, ModelInfo, RoutingPolicy, TaskType, Usage


class _FakeProvider(AIProvider):
    def __init__(self, name: str, *, healthy: bool = True, raises: bool = False) -> None:
        self.name = name
        self._healthy = healthy
        self._raises = raises

    async def generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        return GenerationResult(
            content="ok", model_used=model_id, provider_name=self.name, usage=Usage()
        )

    async def stream_generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        return
        yield  # pragma: no cover

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> bool:
        if self._raises:
            raise RuntimeError("backend exploded")
        return self._healthy

    def count_tokens(self, text: str, *, model_id: str) -> int:
        return len(text)


def _registry() -> list[ModelInfo]:
    return [
        ModelInfo(id="local-model", provider="local", kind="chat", capabilities={"reasoning": 0.3}),
        ModelInfo(
            id="cheap-cloud",
            provider="openai",
            kind="chat",
            capabilities={"reasoning": 0.7},
            cost_per_1k_input_tokens=0.0004,
            cost_per_1k_output_tokens=0.0006,
        ),
        ModelInfo(
            id="quality-cloud",
            provider="anthropic",
            kind="chat",
            capabilities={"reasoning": 0.95},
            cost_per_1k_input_tokens=0.008,
            cost_per_1k_output_tokens=0.012,
        ),
    ]


@pytest.mark.asyncio
async def test_failover_skips_unhealthy_top_candidate() -> None:
    provider_manager = ProviderManager(
        {
            "local": _FakeProvider("local", healthy=True),
            "openai": _FakeProvider("openai", healthy=True),
            "anthropic": _FakeProvider("anthropic", healthy=False),
        }
    )
    router = ModelRouter(provider_manager, _registry())

    decision, provider = await router.route_with_failover(
        task_type=TaskType.COMPLEX_REASONING, policy=RoutingPolicy.MAX_QUALITY
    )

    assert decision.model_id == "cheap-cloud"
    assert provider.name == "openai"


@pytest.mark.asyncio
async def test_failover_falls_back_to_local_when_all_cloud_candidates_fail() -> None:
    provider_manager = ProviderManager(
        {
            "local": _FakeProvider("local", healthy=True),
            "openai": _FakeProvider("openai", healthy=False),
            "anthropic": _FakeProvider("anthropic", raises=True),
        }
    )
    router = ModelRouter(provider_manager, _registry())

    decision, provider = await router.route_with_failover(
        task_type=TaskType.COMPLEX_REASONING, policy=RoutingPolicy.MAX_QUALITY
    )

    assert decision.model_id == "local-model"
    assert decision.provider_name == "local"
    assert provider.name == "local"


@pytest.mark.asyncio
async def test_failover_raises_when_local_is_also_unavailable() -> None:
    provider_manager = ProviderManager(
        {
            "local": _FakeProvider("local", healthy=False),
            "openai": _FakeProvider("openai", healthy=False),
            "anthropic": _FakeProvider("anthropic", healthy=False),
        }
    )
    router = ModelRouter(provider_manager, _registry())

    with pytest.raises(ProviderUnavailableError):
        await router.route_with_failover(
            task_type=TaskType.COMPLEX_REASONING, policy=RoutingPolicy.MAX_QUALITY
        )


@pytest.mark.asyncio
async def test_failover_falls_through_when_explicit_model_provider_is_unhealthy() -> None:
    provider_manager = ProviderManager(
        {
            "local": _FakeProvider("local", healthy=True),
            "openai": _FakeProvider("openai", healthy=True),
            "anthropic": _FakeProvider("anthropic", healthy=False),
        }
    )
    router = ModelRouter(provider_manager, _registry())

    decision, provider = await router.route_with_failover(requested_model_id="claude-sonnet-4-5")

    assert provider.name != "anthropic"


@pytest.mark.asyncio
async def test_require_tool_calling_final_fallback_raises_when_local_lacks_tool_support() -> None:
    provider_manager = ProviderManager(
        {
            "local": _FakeProvider("local", healthy=True),
            "openai": _FakeProvider("openai", healthy=False),
            "anthropic": _FakeProvider("anthropic", healthy=False),
        }
    )
    router = ModelRouter(provider_manager, _registry())

    with pytest.raises(ProviderUnavailableError):
        await router.route_with_failover(
            task_type=TaskType.COMPLEX_REASONING,
            policy=RoutingPolicy.MAX_QUALITY,
            require_tool_calling=True,
        )
