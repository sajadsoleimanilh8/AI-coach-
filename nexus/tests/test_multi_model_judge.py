from __future__ import annotations

import json

import pytest

from nexus.core.exceptions import ProviderUnavailableError
from nexus.core.providers import AIProvider
from nexus.core.router import RoutingDecision
from nexus.core.types import GenerationChunk, GenerationResult, ModelInfo, RoutingPolicy, TaskType, Usage
from nexus.verification.judge import MultiModelJudge


class _FakeProvider(AIProvider):
    def __init__(self, name: str, *, healthy: bool = True, response: str = "") -> None:
        self.name = name
        self._healthy = healthy
        self._response = response
        self.generate_calls = 0

    async def generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        self.generate_calls += 1
        return GenerationResult(
            content=self._response, model_used=model_id, provider_name=self.name,
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        )

    async def stream_generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        yield GenerationChunk(delta="unused", done=True, usage=Usage())

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> bool:
        return self._healthy

    def count_tokens(self, text: str, *, model_id: str) -> int:
        return len(text)


class _FakeRouter:
    def __init__(self, candidates: list[RoutingDecision], providers: dict[str, AIProvider]) -> None:
        self._candidates = candidates
        self._providers = providers

    def ranked_candidates(self, **kwargs) -> list[RoutingDecision]:
        return self._candidates

    def get_provider(self, provider_name: str) -> AIProvider:
        try:
            return self._providers[provider_name]
        except KeyError as exc:
            raise ProviderUnavailableError(str(exc)) from exc


def _decision(provider_name: str, model_id: str) -> RoutingDecision:
    return RoutingDecision(
        provider_name=provider_name, model_id=model_id, task_type=TaskType.RESEARCH,
        policy=RoutingPolicy.BALANCED, reason="fake",
    )


@pytest.mark.asyncio
async def test_judge_picks_a_model_different_from_the_original() -> None:
    original_provider = _FakeProvider("openai")
    judge_provider = _FakeProvider("anthropic", response=json.dumps({"agrees": True}))
    router = _FakeRouter(
        candidates=[_decision("openai", "gpt-4o"), _decision("anthropic", "claude-sonnet-4-5")],
        providers={"openai": original_provider, "anthropic": judge_provider},
    )
    judge = MultiModelJudge(router)

    verdict = await judge.judge(
        question="What is 2+2?", answer="4", original_model_id="gpt-4o", task_type=TaskType.RESEARCH
    )

    assert verdict is not None
    assert verdict.judge_model == "claude-sonnet-4-5"
    assert original_provider.generate_calls == 0
    assert judge_provider.generate_calls == 1


@pytest.mark.asyncio
async def test_returns_none_not_a_pass_when_only_one_model_is_available() -> None:
    provider = _FakeProvider("local")
    router = _FakeRouter(
        candidates=[_decision("local", "mistral:7b")],
        providers={"local": provider},
    )
    judge = MultiModelJudge(router)

    verdict = await judge.judge(
        question="q", answer="a", original_model_id="mistral:7b", task_type=TaskType.RESEARCH
    )

    assert verdict is None
    assert provider.generate_calls == 0


@pytest.mark.asyncio
async def test_skips_unhealthy_distinct_candidates() -> None:
    unhealthy = _FakeProvider("openai", healthy=False)
    healthy = _FakeProvider("anthropic", healthy=True, response=json.dumps({"agrees": True}))
    router = _FakeRouter(
        candidates=[
            _decision("local", "mistral:7b"),
            _decision("openai", "gpt-4o"),
            _decision("anthropic", "claude-sonnet-4-5"),
        ],
        providers={"local": _FakeProvider("local"), "openai": unhealthy, "anthropic": healthy},
    )
    judge = MultiModelJudge(router)

    verdict = await judge.judge(
        question="q", answer="a", original_model_id="mistral:7b", task_type=TaskType.RESEARCH
    )

    assert verdict is not None
    assert verdict.judge_model == "claude-sonnet-4-5"


@pytest.mark.asyncio
async def test_agreeing_judge() -> None:
    judge_provider = _FakeProvider("anthropic", response=json.dumps({"agrees": True}))
    router = _FakeRouter(
        candidates=[_decision("local", "mistral:7b"), _decision("anthropic", "claude-sonnet-4-5")],
        providers={"local": _FakeProvider("local"), "anthropic": judge_provider},
    )
    judge = MultiModelJudge(router)

    verdict = await judge.judge(
        question="q", answer="a", original_model_id="mistral:7b", task_type=TaskType.RESEARCH
    )

    assert verdict.agrees is True
    assert verdict.disagreement_summary is None


@pytest.mark.asyncio
async def test_disagreeing_judge_carries_a_summary() -> None:
    response = json.dumps({"agrees": False, "disagreement_summary": "The date given is wrong."})
    judge_provider = _FakeProvider("anthropic", response=response)
    router = _FakeRouter(
        candidates=[_decision("local", "mistral:7b"), _decision("anthropic", "claude-sonnet-4-5")],
        providers={"local": _FakeProvider("local"), "anthropic": judge_provider},
    )
    judge = MultiModelJudge(router)

    verdict = await judge.judge(
        question="q", answer="a", original_model_id="mistral:7b", task_type=TaskType.RESEARCH
    )

    assert verdict.agrees is False
    assert verdict.disagreement_summary == "The date given is wrong."


@pytest.mark.asyncio
async def test_malformed_judge_response_is_not_silent_agreement() -> None:
    judge_provider = _FakeProvider("anthropic", response="not valid json at all")
    router = _FakeRouter(
        candidates=[_decision("local", "mistral:7b"), _decision("anthropic", "claude-sonnet-4-5")],
        providers={"local": _FakeProvider("local"), "anthropic": judge_provider},
    )
    judge = MultiModelJudge(router)

    verdict = await judge.judge(
        question="q", answer="a", original_model_id="mistral:7b", task_type=TaskType.RESEARCH
    )

    assert verdict is not None
    assert verdict.agrees is False
