from __future__ import annotations

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
        yield  # pragma: no cover

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> bool:
        return True

    def count_tokens(self, text: str, *, model_id: str) -> int:
        return len(text)


class _StubMatrix:
    """Only the one synchronous method ModelRouter's CapabilityProvider
    Protocol requires — no DB, no eval store."""

    def __init__(self, learned: dict[str, dict[str, float]]) -> None:
        self._learned = learned

    def effective_capabilities(self, model_id: str) -> dict[str, float]:
        return self._learned.get(model_id, {})


def _registry() -> list[ModelInfo]:
    return [
        ModelInfo(
            id="model-a", provider="provider-a", kind="chat",
            capabilities={"reasoning": 0.60},
            cost_per_1k_input_tokens=0.005, cost_per_1k_output_tokens=0.005,
        ),
        ModelInfo(
            id="model-b", provider="provider-b", kind="chat",
            capabilities={"reasoning": 0.65},
            cost_per_1k_input_tokens=0.005, cost_per_1k_output_tokens=0.005,
        ),
    ]


def _manager() -> ProviderManager:
    return ProviderManager({"provider-a": _FakeProvider("provider-a"), "provider-b": _FakeProvider("provider-b")})


def _ranked(router: ModelRouter) -> list[str]:
    return [
        d.model_id
        for d in router.ranked_candidates(
            task_type=TaskType.COMPLEX_REASONING, policy=RoutingPolicy.MAX_QUALITY
        )
    ]


def test_no_matrix_ranks_on_static_capabilities() -> None:
    router = ModelRouter(_manager(), _registry())

    assert _ranked(router) == ["model-b", "model-a"]


def test_empty_matrix_is_identical_to_no_matrix() -> None:
    """Principle 4: with no measurements, routing must be exactly what it is
    today — not merely similar."""
    without = ModelRouter(_manager(), _registry())
    with_empty = ModelRouter(_manager(), _registry(), capability_matrix=_StubMatrix({}))

    assert _ranked(with_empty) == _ranked(without) == ["model-b", "model-a"]


def test_a_model_missing_from_the_matrix_keeps_its_static_score() -> None:
    matrix = _StubMatrix({"model-a": {"reasoning": 0.99}})
    router = ModelRouter(_manager(), _registry(), capability_matrix=matrix)

    assert _ranked(router) == ["model-a", "model-b"]


def test_learned_scores_flip_the_ranking() -> None:
    matrix = _StubMatrix({"model-a": {"reasoning": 0.90}, "model-b": {"reasoning": 0.40}})
    router = ModelRouter(_manager(), _registry(), capability_matrix=matrix)

    assert _ranked(router) == ["model-a", "model-b"]


def test_learned_scores_can_confirm_the_static_ordering() -> None:
    matrix = _StubMatrix({"model-a": {"reasoning": 0.30}, "model-b": {"reasoning": 0.95}})
    router = ModelRouter(_manager(), _registry(), capability_matrix=matrix)

    assert _ranked(router) == ["model-b", "model-a"]


def test_a_capability_absent_from_the_learned_dict_falls_back_to_static() -> None:
    matrix = _StubMatrix({"model-a": {"coding": 0.99}})
    router = ModelRouter(_manager(), _registry(), capability_matrix=matrix)

    assert _ranked(router) == ["model-b", "model-a"]


def test_route_uses_the_learned_ordering() -> None:
    matrix = _StubMatrix({"model-a": {"reasoning": 0.90}, "model-b": {"reasoning": 0.40}})
    router = ModelRouter(_manager(), _registry(), capability_matrix=matrix)

    decision = router.route(
        task_type=TaskType.COMPLEX_REASONING, policy=RoutingPolicy.MAX_QUALITY
    )

    assert decision.model_id == "model-a"


def test_balanced_policy_also_reads_learned_capabilities() -> None:
    matrix = _StubMatrix({"model-a": {"reasoning": 0.95}, "model-b": {"reasoning": 0.10}})
    router = ModelRouter(_manager(), _registry(), capability_matrix=matrix)

    ranked = [
        d.model_id
        for d in router.ranked_candidates(
            task_type=TaskType.COMPLEX_REASONING, policy=RoutingPolicy.BALANCED
        )
    ]

    assert ranked[0] == "model-a"


def test_low_cost_policy_is_unaffected_by_learned_capabilities() -> None:
    matrix = _StubMatrix({"model-a": {"reasoning": 0.99}})
    router = ModelRouter(_manager(), _registry(), capability_matrix=matrix)

    ranked_with = [
        d.model_id
        for d in router.ranked_candidates(
            task_type=TaskType.COMPLEX_REASONING, policy=RoutingPolicy.LOW_COST
        )
    ]
    ranked_without = [
        d.model_id
        for d in ModelRouter(_manager(), _registry()).ranked_candidates(
            task_type=TaskType.COMPLEX_REASONING, policy=RoutingPolicy.LOW_COST
        )
    ]

    assert ranked_with == ranked_without
