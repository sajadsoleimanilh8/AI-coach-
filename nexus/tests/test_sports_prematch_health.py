"""
Tests for the pre-match readiness integration in nexus/sports/.

Two things are being protected here:
  1. PreMatchHealthClient preserves the football backend's own distinctions
     -- "not submitted yet" (404 -> None) must never be confused with
     "backend is down" (connection error / 5xx -> ProviderUnavailableError).
  2. CoachAssistant.build_prematch_report narrates and never calculates: the
     numbers on the returned report are byte-identical to what the backend
     supplied, and the LLM is told in its system prompt that it may not
     recompute them.

Uses httpx.MockTransport against the client, the same seam
test_sports_adapter.py uses, so no real backend is needed.
"""

from __future__ import annotations

import httpx
import pytest

from nexus.core.exceptions import ProviderUnavailableError
from nexus.core.providers import AIProvider
from nexus.core.router import RoutingDecision
from nexus.core.types import (
    GenerationChunk,
    GenerationResult,
    ModelInfo,
    RoutingPolicy,
    Usage,
)
from nexus.sports.adapter import MatchAnalysis, SportsDataAdapter
from nexus.sports.coach import CoachAssistant
from nexus.sports.prematch_health import PreMatchAssessment, PreMatchHealthClient

_ASSESSMENT = {
    "player_id": "p123",
    "match_id": "m456",
    "physical_readiness": 78.0,
    "fatigue_score": 32.0,
    "recovery_score": 81.0,
    "performance_risk": "low",
    "workload_risk": "moderate",
    "key_positive_factors": ["good sleep quality and duration", "good hydration"],
    "key_negative_factors": ["high recent training load"],
    "method": "heuristic_proxy",
    "schema_version": "v1",
    "computed_at": "2026-08-10T09:00:00",
    "assessment_id": "a1",
    "questionnaire_id": "q1",
    "submission_index": 1,
    "factors": [],
    "data_source": "self_reported",
    "notes": None,
    "disclaimer": "Performance-readiness estimate ... Not a medical assessment.",
}


def _handler(*, status: int = 200, unreachable: bool = False):
    def handler(request: httpx.Request) -> httpx.Response:
        if unreachable:
            raise httpx.ConnectError("connection refused", request=request)
        if status != 200:
            return httpx.Response(status, json={"detail": "boom"})
        return httpx.Response(200, json=_ASSESSMENT)

    return handler


def _client(**kwargs) -> PreMatchHealthClient:
    return PreMatchHealthClient(
        "http://football-backend:8000", transport=httpx.MockTransport(_handler(**kwargs))
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_latest_maps_the_backend_contract_verbatim() -> None:
    assessment = await _client().get_latest("p123")

    assert assessment is not None
    assert assessment.player_id == "p123"
    assert assessment.match_id == "m456"
    assert assessment.physical_readiness == 78.0
    assert assessment.fatigue_score == 32.0
    assert assessment.recovery_score == 81.0
    assert assessment.performance_risk == "low"
    assert assessment.workload_risk == "moderate"
    assert assessment.key_positive_factors == [
        "good sleep quality and duration",
        "good hydration",
    ]
    assert assessment.key_negative_factors == ["high recent training load"]
    assert assessment.method == "heuristic_proxy"
    assert assessment.schema_version == "v1"
    assert assessment.data_source == "self_reported"
    # The untouched payload is kept so nothing is lost in translation.
    assert assessment.raw == _ASSESSMENT


@pytest.mark.asyncio
async def test_get_assessment_fetches_a_specific_id() -> None:
    assessment = await _client().get_assessment("p123", "a1")
    assert assessment is not None
    assert assessment.player_id == "p123"


@pytest.mark.asyncio
async def test_404_means_not_submitted_yet_not_an_outage() -> None:
    assert await _client(status=404).get_latest("nobody") is None
    assert await _client(status=404).get_assessment("nobody", "a1") is None


@pytest.mark.asyncio
async def test_connection_failure_raises_rather_than_guessing() -> None:
    with pytest.raises(ProviderUnavailableError):
        await _client(unreachable=True).get_latest("p123")


@pytest.mark.asyncio
async def test_server_error_raises_rather_than_guessing() -> None:
    with pytest.raises(ProviderUnavailableError):
        await _client(status=500).get_latest("p123")


def test_prompt_context_contains_every_number_and_nothing_invented() -> None:
    context = PreMatchAssessment.from_json(_ASSESSMENT).as_prompt_context()
    for fragment in (
        "78.0",
        "32.0",
        "81.0",
        "low",
        "moderate",
        "good hydration",
        "high recent training load",
        "heuristic_proxy",
        "self_reported",
    ):
        assert fragment in context


def test_prompt_context_labels_free_text_as_unscored() -> None:
    context = PreMatchAssessment.from_json(
        {**_ASSESSMENT, "notes": "200mg caffeine"}
    ).as_prompt_context()
    assert "200mg caffeine" in context
    assert "not scored" in context


def test_prompt_context_states_when_no_match_is_linked() -> None:
    context = PreMatchAssessment.from_json(
        {**_ASSESSMENT, "match_id": None}
    ).as_prompt_context()
    assert "not linked to a match yet" in context


# ---------------------------------------------------------------------------
# CoachAssistant -- narrate only
# ---------------------------------------------------------------------------


class _FakeProvider(AIProvider):
    name = "fake"

    def __init__(self) -> None:
        self.received_messages = None

    async def generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        self.received_messages = messages
        return GenerationResult(
            content="Fit to start; manage minutes given the recent load.",
            model_used=model_id,
            provider_name=self.name,
            usage=Usage(),
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
            provider_name=self._provider.name,
            model_id="fake-model",
            task_type=kwargs.get("task_type"),
            policy=RoutingPolicy.BALANCED,
            reason="fake",
        )
        return decision, self._provider


class _UnusedAdapter(SportsDataAdapter):
    """The pre-match path must not touch the tactical adapter at all."""

    async def get_match_analysis(self, match_id: str) -> MatchAnalysis:
        raise AssertionError("tactical adapter must not be used for a pre-match report")

    async def get_player_analysis(self, match_id: str, player_id: int) -> MatchAnalysis:
        raise AssertionError("tactical adapter must not be used for a pre-match report")


def _coach(provider: _FakeProvider, **client_kwargs) -> CoachAssistant:
    return CoachAssistant(_FakeRouter(provider), _UnusedAdapter(), _client(**client_kwargs))


@pytest.mark.asyncio
async def test_report_carries_the_backend_numbers_through_unchanged() -> None:
    provider = _FakeProvider()
    report = await _coach(provider).build_prematch_report("p123", "m456")

    assert report is not None
    assert report.player_id == "p123"
    assert report.match_id == "m456"
    assert report.assessment.physical_readiness == 78.0
    assert report.assessment.fatigue_score == 32.0
    assert report.assessment.recovery_score == 81.0
    assert report.assessment.performance_risk == "low"
    assert report.assessment.workload_risk == "moderate"
    # The narrative is the ONLY model-generated field.
    assert report.narrative == "Fit to start; manage minutes given the recent load."
    assert report.model_used == "fake-model"


@pytest.mark.asyncio
async def test_llm_is_instructed_not_to_compute_anything() -> None:
    provider = _FakeProvider()
    await _coach(provider).build_prematch_report("p123", "m456")

    system_prompt = provider.received_messages[0].content
    assert provider.received_messages[0].role == "system"
    assert "ALREADY been computed" in system_prompt
    assert "Never calculate" in system_prompt
    assert "not a medical assessment" in system_prompt.lower()


@pytest.mark.asyncio
async def test_llm_receives_the_computed_values_as_context() -> None:
    provider = _FakeProvider()
    await _coach(provider).build_prematch_report("p123", "m456")

    user_content = provider.received_messages[1].content
    assert provider.received_messages[1].role == "user"
    assert "78.0" in user_content
    assert "high recent training load" in user_content
    assert "do not" in user_content.lower()


@pytest.mark.asyncio
async def test_exactly_one_llm_call_is_made() -> None:
    provider = _FakeProvider()
    await _coach(provider).build_prematch_report("p123", "m456")
    assert len(provider.received_messages) == 2  # one system + one user, single call


@pytest.mark.asyncio
async def test_no_assessment_yet_returns_none_rather_than_a_blank_report() -> None:
    provider = _FakeProvider()
    report = await _coach(provider, status=404).build_prematch_report("nobody", "m456")
    assert report is None
    # No LLM call is made when there is nothing to narrate.
    assert provider.received_messages is None


@pytest.mark.asyncio
async def test_backend_outage_propagates() -> None:
    with pytest.raises(ProviderUnavailableError):
        await _coach(_FakeProvider(), unreachable=True).build_prematch_report("p123", "m456")


@pytest.mark.asyncio
async def test_unconfigured_client_fails_loudly() -> None:
    """A CoachAssistant built without a pre-match client must raise, not
    quietly return an empty report."""
    coach = CoachAssistant(_FakeRouter(_FakeProvider()), _UnusedAdapter())
    with pytest.raises(ProviderUnavailableError):
        await coach.build_prematch_report("p123")


@pytest.mark.asyncio
async def test_assessment_match_id_wins_over_the_path_match_id() -> None:
    """The questionnaire's own match link is authoritative; the path value is
    only a fallback label for an unlinked submission."""
    report = await _coach(_FakeProvider()).build_prematch_report("p123", "some-other-match")
    assert report is not None
    assert report.match_id == "m456"
