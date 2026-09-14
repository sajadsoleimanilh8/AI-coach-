from __future__ import annotations

import os
from typing import Any, AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from nexus.api.main import create_app
from nexus.config.settings import get_settings
from nexus.core.types import Usage
from nexus.generation.service import PersonalizedPlan
from nexus.health.analyzer import HealthAnalysis, HealthPattern
from nexus.sports.adapter import MatchAnalysis, SportsDataAdapter, SportsMetric
from nexus.sports.coach import CoachReport
from nexus.sports.game_plan import build_game_plan
from nexus.sports.tactical import TacticalFinding


def _metric(name: str, value, confidence: str = "normal", player_id=None) -> SportsMetric:
    return SportsMetric(
        metric_name=name, value=value, method="deterministic", confidence=confidence,
        sample_size=10, sub_scores={}, player_id=player_id,
    )


class _FakeHealthAnalyzer:
    async def analyze(self, user_id: str) -> HealthAnalysis:
        return HealthAnalysis(
            user_id=user_id,
            patterns=[
                HealthPattern(
                    name="recovery_debt", dimensions_involved=["physical.recovery"],
                    severity="notable", confidence=0.8, sample_size=6,
                    explanation="Recovery has been below baseline.",
                )
            ],
            scorecard={"physical": 0.4},
            data_sufficiency="adequate",
            computed_at=123.0,
        )


class _FakeGenerator:
    async def generate(self, *, user_id: str, request_type: str, constraints: dict[str, Any]) -> PersonalizedPlan:
        return PersonalizedPlan(
            request_type=request_type,
            content="Here is your personalized plan.",
            brief_rationale=["Starting from the default intensity of 0.70."],
            addressed_weaknesses=["physical.recovery"],
            target_intensity=0.5,
            personalized=True,
            model_used="fake-model",
            usage=Usage(prompt_tokens=10, completion_tokens=20),
        )


def _game_plan_analysis() -> MatchAnalysis:
    """Enough real metrics for build_game_plan to produce a full plan, so
    the route's game-plan serialisation is exercised against a real
    GamePlan rather than a hand-written stand-in."""
    team = [
        SportsMetric("compactness_score", 80.0, "deterministic", "normal", 50, {}),
        SportsMetric("formation_stability_score", 22.0, "deterministic", "normal", 40, {}),
    ]
    return MatchAnalysis(
        match_id="m1", team_metrics=team, player_metrics=[], unavailable=[], coverage=0.5
    )


class _FakeCoachAssistant:
    async def build_report(self, match_id: str, player_id: int | None = None) -> CoachReport:
        return CoachReport(
            game_plan=build_game_plan(_game_plan_analysis()),
            match_id=match_id,
            findings=[
                TacticalFinding(
                    area="defensive_compactness", assessment="strength",
                    supporting_metrics=["compactness_score"], confidence="normal",
                    explanation="compactness_score=80.0 (strength).",
                )
            ],
            unavailable_metrics=["pressing_intensity_score: unavailable — upstream tracking confidence too low to compute"],
            coverage=0.5,
            narrative="Solid defensive shape; pressing data unavailable this match.",
            model_used="fake-model",
        )


class _FakeSportsAdapter(SportsDataAdapter):
    async def get_match_analysis(self, match_id: str) -> MatchAnalysis:
        return self._analysis()

    async def get_player_analysis(self, match_id: str, player_id: int) -> MatchAnalysis:
        return self._analysis()

    @staticmethod
    def _analysis() -> MatchAnalysis:
        available = _metric("decision_making_score", 75.0, player_id=7)
        return MatchAnalysis(
            match_id="m1", team_metrics=[], player_metrics=[available], unavailable=[], coverage=1.0,
        )


@pytest.fixture(scope="module")
def _memory_db_env(tmp_path_factory) -> str:
    db_path = str(tmp_path_factory.mktemp("nexus-group-c") / "nexus.db")
    os.environ["NEXUS_MEMORY__DATABASE_PATH"] = db_path
    yield db_path
    os.environ.pop("NEXUS_MEMORY__DATABASE_PATH", None)


@pytest.fixture
async def client(_memory_db_env: str) -> AsyncIterator[AsyncClient]:
    get_settings(refresh=True)
    app = create_app()
    async with app.router.lifespan_context(app):
        app.state.health_analyzer = _FakeHealthAnalyzer()
        app.state.personalized_generator = _FakeGenerator()
        app.state.coach_assistant = _FakeCoachAssistant()
        app.state.sports_adapter = _FakeSportsAdapter()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest.mark.asyncio
async def test_health_analysis_endpoint_is_well_formed(client: AsyncClient) -> None:
    response = await client.get("/api/health-intel/u1/analysis")

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "u1"
    assert body["data_sufficiency"] == "adequate"
    assert len(body["patterns"]) == 1
    assert body["patterns"][0]["name"] == "recovery_debt"
    assert body["patterns"][0]["confidence"] == 0.8
    assert "not medical advice" in body["disclaimer"].lower()


@pytest.mark.asyncio
async def test_health_scorecard_endpoint_is_well_formed(client: AsyncClient) -> None:
    response = await client.get("/api/health-intel/u1/scorecard")

    assert response.status_code == 200
    body = response.json()
    assert body["scorecard"] == {"physical": 0.4}
    assert body["disclaimer"]


@pytest.mark.asyncio
async def test_generate_endpoint_is_well_formed(client: AsyncClient) -> None:
    response = await client.post(
        "/api/generate/workout", json={"user_id": "u1", "constraints": {"time_minutes": 30}}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["request_type"] == "workout"
    assert body["personalized"] is True
    assert body["target_intensity"] == 0.5
    assert body["addressed_weaknesses"] == ["physical.recovery"]
    assert body["usage"]["total_tokens"] == 30


@pytest.mark.asyncio
async def test_generate_endpoint_rejects_unknown_request_type(client: AsyncClient) -> None:
    response = await client.post("/api/generate/not-a-real-type", json={"user_id": "u1"})
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_sports_match_report_endpoint_is_well_formed(client: AsyncClient) -> None:
    response = await client.get("/api/sports/m1/report")

    assert response.status_code == 200
    body = response.json()
    assert body["match_id"] == "m1"
    assert len(body["findings"]) == 1
    assert body["unavailable_metrics"]
    assert "pressing_intensity_score" in body["unavailable_metrics"][0]
    assert body["coverage"] == 0.5
    assert body["narrative"]

    # The game plan is the part a coach acts on, so the route must serialise
    # it whole -- shape, principles, and adjustments still tied to a named
    # weakness once they have crossed the wire.
    plan = body["game_plan"]
    assert plan is not None
    assert plan["proposed_shape"]["shape"] == "3-4-3"
    assert plan["opponent_weaknesses"][0]["key"] == "formation_stability_score"
    assert plan["opponent_strengths"][0]["key"] == "compactness_score"
    assert plan["principles"] and plan["adjustments"]
    weakness_labels = {w["label"] for w in plan["opponent_weaknesses"]}
    for adjustment in plan["adjustments"]:
        assert adjustment["targets_weakness"] in weakness_labels
        assert adjustment["supporting_metrics"]

    # A missing Phase 4 timeline must be reported as missing, not omitted.
    assert body["timeline_sections"]["available"] == []
    assert set(body["timeline_sections"]["missing"]) == {
        "phases", "pressing", "transitions", "territory"
    }
    assert body["is_partial"] is True


@pytest.mark.asyncio
async def test_sports_player_report_endpoint_is_well_formed(client: AsyncClient) -> None:
    response = await client.get("/api/sports/m1/player/7")

    assert response.status_code == 200
    assert response.json()["match_id"] == "m1"


@pytest.mark.asyncio
async def test_sports_ingest_endpoint_records_signals(client: AsyncClient) -> None:
    response = await client.post(
        "/api/sports/m1/ingest", json={"user_id": "ingest-user", "player_id": 7}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["match_id"] == "m1"
    assert body["user_id"] == "ingest-user"
    assert body["signals_recorded"] == 1

    state_response = await client.get("/api/personal/ingest-user/state")
    assert "sports.decision_making" in state_response.json()["dimensions"]
