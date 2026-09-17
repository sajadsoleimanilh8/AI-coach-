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


class _FakeLatencyTracker:
    """Stub with controlled p50 values — no DB, no in-memory window, just
    the two synchronous reads ModelRouter actually calls."""

    def __init__(self, data: dict[tuple[str, str], tuple[float, int]]) -> None:
        self._data = data

    def p50(self, provider_name: str, model_id: str) -> float | None:
        entry = self._data.get((provider_name, model_id))
        return entry[0] if entry else None

    def sample_count(self, provider_name: str, model_id: str) -> int:
        entry = self._data.get((provider_name, model_id))
        return entry[1] if entry else 0


def _registry() -> list[ModelInfo]:
    # Equal cost on both cloud-ish candidates isolates the latency/capability
    # comparison — cost_norm is identical for both, so it can't be what
    # decides the ranking; only capability and the latency component can.
    return [
        ModelInfo(
            id="provider-a-model",
            provider="provider-a",
            kind="chat",
            capabilities={"reasoning": 0.6},
            cost_per_1k_input_tokens=0.005,
            cost_per_1k_output_tokens=0.005,
        ),
        ModelInfo(
            id="provider-b-model",
            provider="provider-b",
            kind="chat",
            capabilities={"reasoning": 0.65},
            cost_per_1k_input_tokens=0.005,
            cost_per_1k_output_tokens=0.005,
        ),
    ]


def _provider_manager() -> ProviderManager:
    return ProviderManager(
        {"provider-a": _FakeProvider("provider-a"), "provider-b": _FakeProvider("provider-b")}
    )


def test_low_latency_falls_back_to_cost_ordering_without_latency_data() -> None:
    router = ModelRouter(_provider_manager(), _registry())  # no latency_tracker

    decision = router.route(task_type=TaskType.COMPLEX_REASONING, policy=RoutingPolicy.LOW_LATENCY)

    # Neither candidate is "local", so the is_local fallback heuristic gives
    # both a component of 0.0 — equal cost ties them, broken by model.id.
    assert decision.model_id == "provider-a-model"
    assert "falling back to local-preference heuristic" in decision.reason


def test_low_latency_prefers_the_observed_faster_model() -> None:
    latency_tracker = _FakeLatencyTracker(
        {
            ("provider-a", "provider-a-model"): (0.1, 20),  # fast
            ("provider-b", "provider-b-model"): (2.0, 20),  # slow
        }
    )
    router = ModelRouter(_provider_manager(), _registry(), latency_tracker=latency_tracker)

    decision = router.route(task_type=TaskType.COMPLEX_REASONING, policy=RoutingPolicy.LOW_LATENCY)

    assert decision.model_id == "provider-a-model"
    assert "observed p50" in decision.reason
    assert "20 samples" in decision.reason


def test_low_latency_flips_when_observed_data_contradicts_cost_ordering() -> None:
    # provider-b is the "winner" by cost/id tiebreak without data (see the
    # no-data test above's sibling ordering); with data showing it's much
    # slower, provider-a must win instead.
    latency_tracker = _FakeLatencyTracker(
        {
            ("provider-a", "provider-a-model"): (0.2, 10),
            ("provider-b", "provider-b-model"): (5.0, 10),
        }
    )
    router = ModelRouter(_provider_manager(), _registry(), latency_tracker=latency_tracker)

    with_data = router.route(task_type=TaskType.COMPLEX_REASONING, policy=RoutingPolicy.LOW_LATENCY)
    without_data = ModelRouter(_provider_manager(), _registry()).route(
        task_type=TaskType.COMPLEX_REASONING, policy=RoutingPolicy.LOW_LATENCY
    )

    assert with_data.model_id == "provider-a-model"
    assert without_data.model_id == "provider-a-model"  # tie -> alphabetical, same winner here
    # The important assertion is *why*: heuristic vs. real observed data.
    assert "observed p50" in with_data.reason
    assert "falling back to local-preference heuristic" in without_data.reason


def test_balanced_uses_observed_latency_instead_of_local_bonus() -> None:
    # Without data: provider-b wins on capability alone (0.65 > 0.6, equal
    # cost_norm, equal locality-fallback term of 0.0 for both).
    without_data = ModelRouter(_provider_manager(), _registry()).route(
        task_type=TaskType.COMPLEX_REASONING, policy=RoutingPolicy.BALANCED
    )
    assert without_data.model_id == "provider-b-model"
    assert "falling back to local-preference heuristic" in without_data.reason

    # With data: provider-a is far faster, which is enough to flip BALANCED
    # despite provider-b's small capability edge.
    latency_tracker = _FakeLatencyTracker(
        {
            ("provider-a", "provider-a-model"): (0.1, 15),
            ("provider-b", "provider-b-model"): (2.0, 15),
        }
    )
    with_data = ModelRouter(_provider_manager(), _registry(), latency_tracker=latency_tracker).route(
        task_type=TaskType.COMPLEX_REASONING, policy=RoutingPolicy.BALANCED
    )
    assert with_data.model_id == "provider-a-model"
    assert "observed p50" in with_data.reason
    assert "15 samples" in with_data.reason


def test_missing_data_for_one_candidate_falls_back_only_for_that_candidate() -> None:
    # provider-a has data, provider-b doesn't — each must use its own path
    # independently rather than the whole scoring pass failing.
    latency_tracker = _FakeLatencyTracker({("provider-a", "provider-a-model"): (0.1, 10)})
    router = ModelRouter(_provider_manager(), _registry(), latency_tracker=latency_tracker)

    decisions = router._ranked_decisions(task_type=TaskType.COMPLEX_REASONING, policy=RoutingPolicy.BALANCED)
    reasons_by_model = {d.model_id: d.reason for d in decisions}

    assert "observed p50" in reasons_by_model["provider-a-model"]
    assert "falling back to local-preference heuristic" in reasons_by_model["provider-b-model"]
