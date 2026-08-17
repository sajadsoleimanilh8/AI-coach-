from __future__ import annotations

from typing import Any

import pytest

from nexus.core.providers import AIProvider
from nexus.core.router import RoutingDecision
from nexus.core.types import GenerationChunk, GenerationResult, ModelInfo, RoutingPolicy, Usage
from nexus.generation.planner import GenerationBrief
from nexus.generation.service import PersonalizedGenerator
from nexus.health.analyzer import HealthPattern
from nexus.health.safety import MEDICAL_DISCLAIMER
from nexus.personal.state import DimensionState, PersonalState
from nexus.personal.weakness import Weakness


class _FakeProvider(AIProvider):
    name = "fake"

    def __init__(self, *, content: str) -> None:
        self._content = content
        self.received_messages = None

    async def generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        self.received_messages = messages
        return GenerationResult(
            content=self._content,
            model_used=model_id,
            provider_name=self.name,
            usage=Usage(prompt_tokens=10, completion_tokens=20),
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
            policy=RoutingPolicy.BALANCED,
            reason="fake routing",
        )
        return decision, self._provider


class _FakeBriefBuilder:
    def __init__(self, brief: GenerationBrief) -> None:
        self._brief = brief
        self.received_kwargs: dict[str, Any] | None = None

    async def build(self, *, user_id: str, request_type: str, constraints: dict[str, Any]) -> GenerationBrief:
        self.received_kwargs = {
            "user_id": user_id, "request_type": request_type, "constraints": constraints
        }
        return self._brief


def _brief(
    *, request_type: str = "workout", with_signals: bool = True, health_patterns=None
) -> GenerationBrief:
    dimensions = (
        {"physical.energy": DimensionState("physical.energy", 0.6, 0.9, 5, 1.0)}
        if with_signals
        else {}
    )
    weaknesses = (
        [
            Weakness(
                dimension="physical.recovery", current=0.3, baseline=0.6, deviation=0.3,
                priority="HIGH", confidence=0.8, trend=None,
                explanation="physical.recovery is currently 0.30 vs. baseline 0.60.",
            )
        ]
        if with_signals
        else []
    )
    return GenerationBrief(
        user_id="u1",
        request_type=request_type,
        state=PersonalState(user_id="u1", dimensions=dimensions, computed_at=0.0),
        weaknesses=weaknesses,
        health_patterns=health_patterns or [],
        profile={},
        constraints={"time_minutes": 30},
        target_intensity=0.5 if with_signals else 0.7,
        rationale=["Starting from the default intensity of 0.70.", "Reduced intensity by 0.12 due to low recovery."],
    )


@pytest.mark.asyncio
async def test_brief_reaches_the_model() -> None:
    provider = _FakeProvider(content="Here is your plan.")
    router = _FakeRouter(provider)
    brief_builder = _FakeBriefBuilder(_brief())
    generator = PersonalizedGenerator(router, brief_builder)

    await generator.generate(user_id="u1", request_type="workout", constraints={"time_minutes": 30})

    assert provider.received_messages is not None
    user_message = next(m for m in provider.received_messages if m.role == "user")
    assert "Target intensity: 0.50" in user_message.content
    assert "physical.recovery" in user_message.content


@pytest.mark.asyncio
async def test_plan_surfaces_rationale_and_addressed_weaknesses() -> None:
    provider = _FakeProvider(content="Here is your plan.")
    router = _FakeRouter(provider)
    brief = _brief()
    brief_builder = _FakeBriefBuilder(brief)
    generator = PersonalizedGenerator(router, brief_builder)

    plan = await generator.generate(user_id="u1", request_type="workout", constraints={})

    assert plan.brief_rationale == brief.rationale
    assert plan.addressed_weaknesses == ["physical.recovery"]
    assert plan.target_intensity == 0.5
    assert plan.content == "Here is your plan."
    assert plan.model_used == "fake-model"


@pytest.mark.asyncio
async def test_personalized_is_true_when_signals_exist() -> None:
    provider = _FakeProvider(content="plan")
    router = _FakeRouter(provider)
    brief_builder = _FakeBriefBuilder(_brief(with_signals=True))
    generator = PersonalizedGenerator(router, brief_builder)

    plan = await generator.generate(user_id="u1", request_type="workout", constraints={})

    assert plan.personalized is True


@pytest.mark.asyncio
async def test_personalized_is_false_with_no_signals() -> None:
    provider = _FakeProvider(content="plan")
    router = _FakeRouter(provider)
    brief_builder = _FakeBriefBuilder(_brief(with_signals=False))
    generator = PersonalizedGenerator(router, brief_builder)

    plan = await generator.generate(user_id="u1", request_type="workout", constraints={})

    assert plan.personalized is False


@pytest.mark.asyncio
async def test_workout_with_significant_health_pattern_is_safety_gated() -> None:
    provider = _FakeProvider(content="You have a stress disorder, push through it.")
    router = _FakeRouter(provider)
    pattern = HealthPattern(
        name="high_load_low_recovery", dimensions_involved=["physical.recovery"],
        severity="significant", confidence=0.9, sample_size=6, explanation="...",
    )
    brief_builder = _FakeBriefBuilder(_brief(request_type="workout", health_patterns=[pattern]))
    generator = PersonalizedGenerator(router, brief_builder)

    plan = await generator.generate(user_id="u1", request_type="workout", constraints={})

    assert plan.content != "You have a stress disorder, push through it."
    assert MEDICAL_DISCLAIMER in plan.content


@pytest.mark.asyncio
async def test_study_request_is_never_safety_gated_even_with_significant_pattern() -> None:
    provider = _FakeProvider(content="You have a stress disorder.")
    router = _FakeRouter(provider)
    pattern = HealthPattern(
        name="stress_accumulation", dimensions_involved=["mental.stress"],
        severity="significant", confidence=0.9, sample_size=6, explanation="...",
    )
    brief_builder = _FakeBriefBuilder(_brief(request_type="study", health_patterns=[pattern]))
    generator = PersonalizedGenerator(router, brief_builder)

    plan = await generator.generate(user_id="u1", request_type="study", constraints={})

    assert plan.content == "You have a stress disorder."


@pytest.mark.asyncio
async def test_workout_without_significant_pattern_is_not_gated() -> None:
    provider = _FakeProvider(content="You have a stress disorder.")
    router = _FakeRouter(provider)
    brief_builder = _FakeBriefBuilder(_brief(request_type="workout", health_patterns=[]))
    generator = PersonalizedGenerator(router, brief_builder)

    plan = await generator.generate(user_id="u1", request_type="workout", constraints={})

    assert plan.content == "You have a stress disorder."
