"""
Tests for the mental-readiness integration in nexus/sports/.
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
from nexus.sports.psychology import (
    build_prompt_context,
    derive_psychology_factors,
)
from nexus.sports.psychology_adapter import PsychologyAssessment, PsychologyClient

_ASSESSMENT = {
    "player_id": "p123",
    "match_id": "m456",
    "cv_player_id": None,
    "mental_readiness": 82,
    "focus": 88,
    "confidence": 76,
    "stress": 54,
    "pressure_risk": "moderate",
    "mental_performance_risk": "low",
    "factors": {
        "focus": "positive",
        "confidence": "positive",
        "stress": "negative",
        "error_recovery": "positive",
        "motivation": "positive",
        "pressure_response": "neutral",
    },
    "key_positive_factors": ["high focus", "strong confidence"],
    "key_negative_factors": ["elevated pre-match stress"],
    "method": "heuristic_proxy",
    "confidence_level": "normal",
    "sample_size": 10,
    "schema_version": "v1",
    "submitted_at": "2026-08-10T09:00:00",
    "computed_at": "2026-08-10T09:00:00",
    "assessment_id": "a1",
    "submission_index": 1,
    "data_source": "self_reported",
    "historical_context": [],
    "disclaimer": "Mental-readiness estimate ... Not a diagnosis.",
}


def _handler(*, status: int = 200, unreachable: bool = False, payload: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if unreachable:
            raise httpx.ConnectError("connection refused", request=request)
        if status != 200:
            return httpx.Response(status, json={"detail": "boom"})
        return httpx.Response(200, json=payload or _ASSESSMENT)

    return handler


def _client(**kwargs) -> PsychologyClient:
    return PsychologyClient(
        "http://football-backend:8000", transport=httpx.MockTransport(_handler(**kwargs))
    )




@pytest.mark.asyncio
async def test_get_latest_maps_the_backend_contract_verbatim() -> None:
    assessment = await _client().get_latest("p123")

    assert assessment is not None
    assert assessment.player_id == "p123"
    assert assessment.match_id == "m456"
    assert assessment.mental_readiness == 82
    assert assessment.focus == 88
    assert assessment.confidence == 76
    assert assessment.stress == 54
    assert assessment.pressure_risk == "moderate"
    assert assessment.mental_performance_risk == "low"
    assert assessment.method == "heuristic_proxy"
    assert assessment.confidence_level == "normal"
    assert assessment.schema_version == "v1"
    assert assessment.data_source == "self_reported"
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




def _assessment(**overrides) -> PsychologyAssessment:
    return PsychologyAssessment.from_json({**_ASSESSMENT, **overrides})


def test_factors_are_split_into_positive_and_negative_phrases() -> None:
    findings = derive_psychology_factors(_assessment())

    assert "high focus" in findings.key_positive_factors
    assert "strong confidence" in findings.key_positive_factors
    assert "good error recovery" in findings.key_positive_factors
    assert "strong motivation" in findings.key_positive_factors
    assert findings.key_negative_factors == ["elevated pre-match stress"]
    assert findings.neutral_factors == ["neutral pressure response"]


def test_a_positive_stress_factor_reads_as_low_stress() -> None:
    """The label is on a goodness scale, so "positive" stress means LOW stress.
    The phrase has to say that or it is actively misleading."""
    findings = derive_psychology_factors(
        _assessment(factors={**_ASSESSMENT["factors"], "stress": "positive"})
    )
    assert "low pre-match stress" in findings.key_positive_factors


def test_derivation_is_deterministic() -> None:
    first = derive_psychology_factors(_assessment())
    second = derive_psychology_factors(_assessment())
    assert first == second
    assert first.key_positive_factors == second.key_positive_factors


def test_derivation_is_pure_and_does_not_mutate_the_assessment() -> None:
    assessment = _assessment()
    before = dict(assessment.factors)
    derive_psychology_factors(assessment)
    assert assessment.factors == before


def test_unknown_dimensions_produce_no_phrase_rather_than_a_guess() -> None:
    findings = derive_psychology_factors(
        _assessment(factors={"focus": "positive", "telepathy": "positive"})
    )
    assert findings.key_positive_factors == ["high focus"]


def test_unknown_label_produces_no_phrase() -> None:
    findings = derive_psychology_factors(_assessment(factors={"focus": "excellent"}))
    assert findings.key_positive_factors == []
    assert findings.key_negative_factors == []


def test_empty_factors_yield_empty_lists_not_an_error() -> None:
    findings = derive_psychology_factors(_assessment(factors={}))
    assert findings.key_positive_factors == []
    assert findings.key_negative_factors == []


def test_prompt_context_contains_every_number_and_nothing_invented() -> None:
    assessment = _assessment()
    context = build_prompt_context(assessment, derive_psychology_factors(assessment))
    for fragment in (
        "82",
        "88",
        "76",
        "54",
        "moderate",
        "low",
        "high focus",
        "elevated pre-match stress",
        "heuristic_proxy",
        "self_reported",
    ):
        assert fragment in context


def test_prompt_context_states_when_no_match_is_linked() -> None:
    assessment = _assessment(match_id=None)
    context = build_prompt_context(assessment, derive_psychology_factors(assessment))
    assert "not linked to a match yet" in context


def test_prompt_context_fences_off_observed_performance_proxies() -> None:
    assessment = _assessment(
        historical_context=["focus proxy (observed-performance proxy): 71.0/100"]
    )
    context = build_prompt_context(assessment, derive_psychology_factors(assessment))
    assert "NOT self-reported" in context
    assert "NOT measures of a mental state" in context




class _FakeProvider(AIProvider):
    name = "fake"

    def __init__(self) -> None:
        self.received_messages = None
        self.call_count = 0

    async def generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        self.received_messages = messages
        self.call_count += 1
        return GenerationResult(
            content="Cleared to start; manage the pressure moments early.",
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
        self.task_types: list = []

    async def route_with_failover(self, **kwargs):
        self.task_types.append(kwargs.get("task_type"))
        decision = RoutingDecision(
            provider_name=self._provider.name,
            model_id="fake-model",
            task_type=kwargs.get("task_type"),
            policy=RoutingPolicy.BALANCED,
            reason="fake",
        )
        return decision, self._provider


class _UnusedAdapter(SportsDataAdapter):
    """The psychology path must not touch the tactical adapter at all."""

    async def get_match_analysis(self, match_id: str) -> MatchAnalysis:
        raise AssertionError("tactical adapter must not be used for a psychology report")

    async def get_player_analysis(self, match_id: str, player_id: int) -> MatchAnalysis:
        raise AssertionError("tactical adapter must not be used for a psychology report")


def _coach(provider: _FakeProvider, router=None, **client_kwargs) -> CoachAssistant:
    return CoachAssistant(
        router or _FakeRouter(provider),
        _UnusedAdapter(),
        None,
        _client(**client_kwargs),
    )


@pytest.mark.asyncio
async def test_report_carries_the_backend_numbers_through_unchanged() -> None:
    provider = _FakeProvider()
    report = await _coach(provider).build_psychology_report("p123", "m456")

    assert report is not None
    assert report.player_id == "p123"
    assert report.match_id == "m456"
    assert report.assessment.mental_readiness == 82
    assert report.assessment.focus == 88
    assert report.assessment.confidence == 76
    assert report.assessment.stress == 54
    assert report.assessment.pressure_risk == "moderate"
    assert report.assessment.mental_performance_risk == "low"
    assert report.narrative == "Cleared to start; manage the pressure moments early."
    assert report.model_used == "fake-model"


@pytest.mark.asyncio
async def test_llm_is_instructed_not_to_compute_anything() -> None:
    provider = _FakeProvider()
    await _coach(provider).build_psychology_report("p123", "m456")

    system_prompt = provider.received_messages[0].content
    assert provider.received_messages[0].role == "system"
    assert "ALREADY been computed" in system_prompt
    assert "Never calculate" in system_prompt


@pytest.mark.asyncio
async def test_llm_is_forbidden_from_clinical_or_emotion_claims() -> None:
    provider = _FakeProvider()
    await _coach(provider).build_psychology_report("p123", "m456")

    system_prompt = provider.received_messages[0].content.lower()
    assert "not emotion detection" in system_prompt
    assert "not a diagnosis" in system_prompt
    assert "do not diagnose" in system_prompt
    assert "clinical language" in system_prompt


@pytest.mark.asyncio
async def test_llm_receives_only_precomputed_values_and_phrases() -> None:
    provider = _FakeProvider()
    await _coach(provider).build_psychology_report("p123", "m456")

    user_content = provider.received_messages[1].content
    assert provider.received_messages[1].role == "user"
    assert "82" in user_content
    assert "88" in user_content
    assert "76" in user_content
    assert "54" in user_content
    assert "high focus" in user_content
    assert "elevated pre-match stress" in user_content
    assert "do not" in user_content.lower()


@pytest.mark.asyncio
async def test_llm_is_never_asked_to_produce_a_score() -> None:
    """The user turn must ask for prose about given numbers, never for the
    model to rate, score, or estimate anything itself."""
    provider = _FakeProvider()
    await _coach(provider).build_psychology_report("p123", "m456")

    user_content = provider.received_messages[1].content.lower()
    for forbidden in (
        "calculate the",
        "compute the",
        "estimate the",
        "score the player",
        "rate the player",
        "what score",
    ):
        assert forbidden not in user_content


@pytest.mark.asyncio
async def test_exactly_one_llm_call_is_made() -> None:
    provider = _FakeProvider()
    await _coach(provider).build_psychology_report("p123", "m456")
    assert provider.call_count == 1
    assert len(provider.received_messages) == 2


@pytest.mark.asyncio
async def test_the_existing_sports_task_type_is_reused() -> None:
    from nexus.core.types import TaskType

    provider = _FakeProvider()
    router = _FakeRouter(provider)
    await _coach(provider, router=router).build_psychology_report("p123")
    assert router.task_types == [TaskType.SPORTS]


@pytest.mark.asyncio
async def test_findings_on_the_report_match_the_deterministic_derivation() -> None:
    provider = _FakeProvider()
    report = await _coach(provider).build_psychology_report("p123", "m456")
    assert report is not None
    assert report.findings == derive_psychology_factors(_assessment())


@pytest.mark.asyncio
async def test_no_assessment_yet_returns_none_rather_than_a_blank_report() -> None:
    provider = _FakeProvider()
    report = await _coach(provider, status=404).build_psychology_report("nobody")
    assert report is None
    assert provider.received_messages is None


@pytest.mark.asyncio
async def test_backend_outage_propagates() -> None:
    with pytest.raises(ProviderUnavailableError):
        await _coach(_FakeProvider(), unreachable=True).build_psychology_report("p123")


@pytest.mark.asyncio
async def test_unconfigured_client_fails_loudly() -> None:
    """A CoachAssistant built without a psychology client must raise, not
    quietly return an empty report."""
    coach = CoachAssistant(_FakeRouter(_FakeProvider()), _UnusedAdapter())
    with pytest.raises(ProviderUnavailableError):
        await coach.build_psychology_report("p123")


@pytest.mark.asyncio
async def test_assessment_match_id_wins_over_the_path_match_id() -> None:
    """The questionnaire's own match link is authoritative; the path value is
    only a fallback label for an unlinked submission."""
    report = await _coach(_FakeProvider()).build_psychology_report("p123", "some-other-match")
    assert report is not None
    assert report.match_id == "m456"




class _FakeCoach:
    """Stands in for CoachAssistant on app.state, the same way
    test_group_c_apis.py fakes the other services."""

    def __init__(self, report=None, error: Exception | None = None) -> None:
        self._report = report
        self._error = error
        self.calls: list[tuple] = []

    async def build_psychology_report(self, player_id, match_id=None):
        self.calls.append((player_id, match_id))
        if self._error is not None:
            raise self._error
        return self._report


async def _route_client(coach: _FakeCoach, tmp_path):
    """Builds the real app and swaps in the fake coach, so the route, the
    response schema and the registration in nexus/api/main.py are all
    exercised rather than assumed."""
    import os

    from httpx import ASGITransport, AsyncClient

    from nexus.api.main import create_app
    from nexus.config.settings import get_settings

    os.environ["NEXUS_MEMORY__DATABASE_PATH"] = str(tmp_path / "nexus.db")
    get_settings(refresh=True)
    app = create_app()
    return app, AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_route_is_registered_and_returns_the_contract(tmp_path) -> None:
    provider = _FakeProvider()
    report = await _coach(provider).build_psychology_report("p123", "m456")
    coach = _FakeCoach(report=report)
    app, AsyncClient, ASGITransport = await _route_client(coach, tmp_path)

    async with app.router.lifespan_context(app):
        app.state.coach_assistant = coach
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/sports/psychology/p123/report")

    assert response.status_code == 200
    body = response.json()
    assert body["mental_readiness"] == 82
    assert body["focus"] == 88
    assert body["confidence"] == 76
    assert body["stress"] == 54
    assert body["pressure_risk"] == "moderate"
    assert body["mental_performance_risk"] == "low"
    assert "high focus" in body["key_positive_factors"]
    assert body["key_negative_factors"] == ["elevated pre-match stress"]
    assert body["narrative"] == "Cleared to start; manage the pressure moments early."


@pytest.mark.asyncio
async def test_route_passes_the_match_id_query_through(tmp_path) -> None:
    provider = _FakeProvider()
    report = await _coach(provider).build_psychology_report("p123", "m456")
    coach = _FakeCoach(report=report)
    app, AsyncClient, ASGITransport = await _route_client(coach, tmp_path)

    async with app.router.lifespan_context(app):
        app.state.coach_assistant = coach
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.get("/api/sports/psychology/p123/report?match_id=m999")

    assert coach.calls == [("p123", "m999")]


@pytest.mark.asyncio
async def test_route_404s_when_nothing_has_been_submitted(tmp_path) -> None:
    coach = _FakeCoach(report=None)
    app, AsyncClient, ASGITransport = await _route_client(coach, tmp_path)

    async with app.router.lifespan_context(app):
        app.state.coach_assistant = coach
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/sports/psychology/nobody/report")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_route_503s_on_a_backend_outage(tmp_path) -> None:
    """Distinct from the 404 above, so a caller can tell "nothing to report
    yet" from "we could not find out"."""
    coach = _FakeCoach(error=ProviderUnavailableError("football backend down"))
    app, AsyncClient, ASGITransport = await _route_client(coach, tmp_path)

    async with app.router.lifespan_context(app):
        app.state.coach_assistant = coach
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/sports/psychology/p123/report")

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_the_prematch_health_path_still_works_alongside_this_one() -> None:
    """Regression guard: adding the psychology client to CoachAssistant must
    not disturb the existing pre-match wiring."""
    from nexus.sports.prematch_health import PreMatchHealthClient

    coach = CoachAssistant(
        _FakeRouter(_FakeProvider()),
        _UnusedAdapter(),
        PreMatchHealthClient(
            "http://football-backend:8000",
            transport=httpx.MockTransport(_handler(status=404)),
        ),
        _client(status=404),
    )
    assert await coach.build_prematch_report("p123") is None
    assert await coach.build_psychology_report("p123") is None
