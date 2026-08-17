from __future__ import annotations

import pytest

from nexus.agents.base import AgentContext
from nexus.agents.sports import SportsAgent
from nexus.sports.adapter import MatchAnalysis, SportsDataAdapter, SportsMetric


def _metric(name: str, value, confidence: str = "normal", player_id=None) -> SportsMetric:
    return SportsMetric(
        metric_name=name, value=value, method="deterministic", confidence=confidence,
        sample_size=10, sub_scores={}, player_id=player_id,
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


def _analysis() -> MatchAnalysis:
    available = _metric("compactness_score", 80.0)
    unavailable = _metric("pressing_intensity_score", None, confidence="low_upstream_confidence")
    return MatchAnalysis(
        match_id="m1", team_metrics=[available, unavailable], player_metrics=[],
        unavailable=[unavailable], coverage=0.5,
    )


def _context(extra: dict) -> AgentContext:
    return AgentContext(
        goal="Analyze this match", user_id="u1", session_id=None,
        rag_service=None, long_term_memory=None, personal_state=None, extra=extra,
    )


@pytest.mark.asyncio
async def test_prepare_context_injects_findings_and_unavailable_list() -> None:
    adapter = _FakeAdapter(_analysis())
    agent = SportsAgent()

    messages = await agent.prepare_context(_context({"sports_adapter": adapter, "match_id": "m1"}))

    assert len(messages) == 1
    content = messages[0].content
    assert "defensive_compactness" in content
    assert "pressing_intensity_score" in content
    assert "Unavailable metrics" in content


@pytest.mark.asyncio
async def test_prepare_context_uses_match_analysis_without_a_player_id() -> None:
    adapter = _FakeAdapter(_analysis())

    await SportsAgent().prepare_context(_context({"sports_adapter": adapter, "match_id": "m1"}))

    assert adapter.match_calls == ["m1"]
    assert adapter.player_calls == []


@pytest.mark.asyncio
async def test_prepare_context_uses_player_analysis_when_player_id_given() -> None:
    adapter = _FakeAdapter(_analysis())

    await SportsAgent().prepare_context(
        _context({"sports_adapter": adapter, "match_id": "m1", "player_id": 9})
    )

    assert adapter.player_calls == [("m1", 9)]
    assert adapter.match_calls == []


@pytest.mark.asyncio
async def test_prepare_context_returns_empty_without_a_match_id() -> None:
    adapter = _FakeAdapter(_analysis())
    messages = await SportsAgent().prepare_context(_context({"sports_adapter": adapter}))
    assert messages == []


@pytest.mark.asyncio
async def test_prepare_context_returns_empty_without_an_adapter() -> None:
    messages = await SportsAgent().prepare_context(_context({"match_id": "m1"}))
    assert messages == []
