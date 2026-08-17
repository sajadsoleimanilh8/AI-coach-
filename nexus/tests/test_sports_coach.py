from __future__ import annotations

import pytest

from nexus.core.providers import AIProvider
from nexus.core.router import RoutingDecision
from nexus.core.types import GenerationChunk, GenerationResult, ModelInfo, RoutingPolicy, Usage
from nexus.sports.adapter import MatchAnalysis, SportsDataAdapter, SportsMetric
from nexus.sports.coach import CoachAssistant, unavailable_reason


def _metric(name: str, value, confidence: str = "normal", sample_size: int = 10) -> SportsMetric:
    return SportsMetric(
        metric_name=name, value=value, method="deterministic", confidence=confidence,
        sample_size=sample_size, sub_scores={},
    )


class _FakeAdapter(SportsDataAdapter):
    def __init__(self, analysis: MatchAnalysis) -> None:
        self._analysis = analysis
        self.match_calls: list[str] = []
        self.player_calls: list[tuple[str, int]] = []

    async def get_match_analysis(self, match_id: str) -> MatchAnalysis:
        self.match_calls.append(match_id)
        return self._analysis

    async def get_player_analysis(self, match_id: str, player_id: int) -> MatchAnalysis:
        self.player_calls.append((match_id, player_id))
        return self._analysis


class _FakeProvider(AIProvider):
    name = "fake"

    def __init__(self) -> None:
        self.received_messages = None

    async def generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        self.received_messages = messages
        return GenerationResult(
            content="Solid defensive shape, but pressing data is missing.",
            model_used=model_id, provider_name=self.name, usage=Usage(),
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

    async def route_with_failover(self, **kwargs):
        decision = RoutingDecision(
            provider_name=self._provider.name, model_id="fake-model",
            task_type=kwargs.get("task_type"), policy=RoutingPolicy.BALANCED, reason="fake",
        )
        return decision, self._provider


def _analysis() -> MatchAnalysis:
    available = _metric("compactness_score", 80.0)
    unavailable = _metric("pressing_intensity_score", None, confidence="low_upstream_confidence", sample_size=0)
    return MatchAnalysis(
        match_id="m1", team_metrics=[available, unavailable], player_metrics=[],
        unavailable=[unavailable], coverage=0.5,
    )


@pytest.mark.asyncio
async def test_match_report_uses_match_scoped_analysis() -> None:
    adapter = _FakeAdapter(_analysis())
    coach = CoachAssistant(_FakeRouter(_FakeProvider()), adapter)

    await coach.build_report("m1")

    assert adapter.match_calls == ["m1"]
    assert adapter.player_calls == []


@pytest.mark.asyncio
async def test_player_report_uses_player_scoped_analysis() -> None:
    adapter = _FakeAdapter(_analysis())
    coach = CoachAssistant(_FakeRouter(_FakeProvider()), adapter)

    await coach.build_report("m1", 7)

    assert adapter.player_calls == [("m1", 7)]
    assert adapter.match_calls == []


@pytest.mark.asyncio
async def test_unavailable_metrics_are_populated_and_reach_the_prompt() -> None:
    adapter = _FakeAdapter(_analysis())
    provider = _FakeProvider()
    coach = CoachAssistant(_FakeRouter(provider), adapter)

    report = await coach.build_report("m1")

    assert len(report.unavailable_metrics) == 1
    assert "pressing_intensity_score" in report.unavailable_metrics[0]

    user_message = next(m for m in provider.received_messages if m.role == "user")
    assert "pressing_intensity_score" in user_message.content
    assert "Unavailable metrics" in user_message.content


@pytest.mark.asyncio
async def test_findings_reach_the_prompt() -> None:
    adapter = _FakeAdapter(_analysis())
    provider = _FakeProvider()
    coach = CoachAssistant(_FakeRouter(provider), adapter)

    await coach.build_report("m1")

    user_message = next(m for m in provider.received_messages if m.role == "user")
    assert "defensive_compactness" in user_message.content


@pytest.mark.asyncio
async def test_narrative_is_exactly_the_llm_output_nothing_else_is_llm_generated() -> None:
    adapter = _FakeAdapter(_analysis())
    provider = _FakeProvider()
    coach = CoachAssistant(_FakeRouter(provider), adapter)

    report = await coach.build_report("m1")

    assert report.narrative == "Solid defensive shape, but pressing data is missing."
    assert report.coverage == 0.5
    assert report.model_used == "fake-model"
    assert len(report.findings) == 1
    assert report.findings[0].area == "defensive_compactness"


def test_unavailable_reason_distinguishes_upstream_from_not_yet_computed() -> None:
    low_upstream = _metric("x", None, confidence="low_upstream_confidence")
    not_computed = _metric("y", None, confidence="normal", sample_size=0)

    assert "upstream tracking confidence" in unavailable_reason(low_upstream)
    assert "not yet computed" in unavailable_reason(not_computed)
