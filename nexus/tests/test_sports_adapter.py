from __future__ import annotations

import httpx
import pytest

from nexus.core.exceptions import ProviderUnavailableError
from nexus.sports.adapter import HttpSportsDataAdapter

_FORMATION = {
    "metric_id": "f1", "match_id": "m1", "team_id": "unassigned", "metric_name": "formation",
    "value": "4-3-3", "method": "heuristic_proxy", "confidence": "normal", "confidence_score": 0.8,
    "sample_size": 30, "sub_scores": {}, "computed_at": "2026-01-01T00:00:00", "schema_version": "v3",
}

_TEAM_SHAPE = [
    {
        "metric_id": "t1", "match_id": "m1", "team_id": "unassigned",
        "metric_name": "compactness_score", "value": 72.5, "method": "deterministic",
        "confidence": "normal", "confidence_score": 0.9, "sample_size": 50, "sub_scores": {},
        "computed_at": "2026-01-01T00:00:00", "schema_version": "v3",
    },
    {
        "metric_id": "t2", "match_id": "m1", "team_id": "unassigned",
        "metric_name": "pressing_intensity_score", "value": None, "method": "deterministic",
        "confidence": "low_upstream_confidence", "confidence_score": None, "sample_size": 0,
        "sub_scores": {}, "computed_at": "2026-01-01T00:00:00", "schema_version": "v3",
    },
]

_PLAYERS = [
    {
        "player_id": 7, "player_name": "Player #7",
        "metrics": [
            {
                "metric_id": "p1", "match_id": "m1", "player_id": 7,
                "metric_name": "decision_making_score", "value": 68.0, "method": "heuristic_proxy",
                "confidence": "normal", "sample_size": 12, "sub_scores": {},
                "computed_at": "2026-01-01T00:00:00", "schema_version": "v3",
            },
            {
                "metric_id": "p2", "match_id": "m1", "player_id": 7,
                "metric_name": "press_resistance_score", "value": None, "method": "heuristic_proxy",
                "confidence": "low_upstream_confidence", "sample_size": 0, "sub_scores": {},
                "computed_at": "2026-01-01T00:00:00", "schema_version": "v3",
            },
            {
                "metric_id": "p3", "match_id": "m1", "player_id": 7,
                "metric_name": "first_touch_score", "value": 55.0, "method": "heuristic_proxy",
                "confidence": "low_sample", "sample_size": 2, "sub_scores": {},
                "computed_at": "2026-01-01T00:00:00", "schema_version": "v3",
            },
        ],
    },
]

_PLAYER_SCOPED = [_PLAYERS[0]["metrics"][0]]


def _handler(*, formation_status: int = 200, unreachable_paths: frozenset[str] = frozenset()):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path in unreachable_paths:
            raise httpx.ConnectError("connection refused", request=request)
        if path.startswith("/api/tactical/formation/"):
            if formation_status == 404:
                return httpx.Response(404, json={"detail": "not found"})
            return httpx.Response(200, json=_FORMATION)
        if path.startswith("/api/tactical/team_shape/"):
            return httpx.Response(200, json=_TEAM_SHAPE)
        if path.count("/") == 4:
            return httpx.Response(200, json=_PLAYER_SCOPED)
        if path.startswith("/api/player_intelligence/"):
            return httpx.Response(200, json=_PLAYERS)
        return httpx.Response(404)

    return handler


def _adapter(**handler_kwargs) -> HttpSportsDataAdapter:
    transport = httpx.MockTransport(_handler(**handler_kwargs))
    return HttpSportsDataAdapter("http://football-backend.test", transport=transport)


@pytest.mark.asyncio
async def test_low_upstream_confidence_metrics_land_in_unavailable_not_coerced() -> None:
    adapter = _adapter()
    analysis = await adapter.get_match_analysis("m1")

    unavailable_names = {m.metric_name for m in analysis.unavailable}
    assert "pressing_intensity_score" in unavailable_names
    assert "press_resistance_score" in unavailable_names

    pressing = next(m for m in analysis.unavailable if m.metric_name == "pressing_intensity_score")
    assert pressing.value is None
    assert pressing.is_available is False


@pytest.mark.asyncio
async def test_low_sample_confidence_with_a_real_value_is_still_available() -> None:
    adapter = _adapter()
    analysis = await adapter.get_match_analysis("m1")

    first_touch = next(m for m in analysis.player_metrics if m.metric_name == "first_touch_score")
    assert first_touch.confidence == "low_sample"
    assert first_touch.value == 55.0
    assert first_touch.is_available is True
    assert first_touch not in analysis.unavailable


@pytest.mark.asyncio
async def test_normal_metrics_are_available() -> None:
    adapter = _adapter()
    analysis = await adapter.get_match_analysis("m1")

    compactness = next(m for m in analysis.team_metrics if m.metric_name == "compactness_score")
    assert compactness.is_available is True
    assert compactness.value == 72.5


@pytest.mark.asyncio
async def test_formation_string_value_is_preserved_not_dropped() -> None:
    adapter = _adapter()
    analysis = await adapter.get_match_analysis("m1")

    formation = next(m for m in analysis.team_metrics if m.metric_name == "formation")
    assert formation.value == "4-3-3"
    assert formation.is_available is True


@pytest.mark.asyncio
async def test_formation_404_yields_no_formation_metric_not_an_error() -> None:
    adapter = _adapter(formation_status=404)
    analysis = await adapter.get_match_analysis("m1")

    assert not any(m.metric_name == "formation" for m in analysis.team_metrics)
    assert not any(m.metric_name == "formation" for m in analysis.unavailable)


@pytest.mark.asyncio
async def test_coverage_math() -> None:
    adapter = _adapter()
    analysis = await adapter.get_match_analysis("m1")

    total = len(analysis.team_metrics) + len(analysis.player_metrics)
    available = total - len(analysis.unavailable)
    assert analysis.coverage == pytest.approx(available / total)
    assert total == 6
    assert len(analysis.unavailable) == 2
    assert analysis.coverage == pytest.approx(4 / 6)


@pytest.mark.asyncio
async def test_backend_unreachable_raises_provider_unavailable_on_match_analysis() -> None:
    adapter = _adapter(unreachable_paths=frozenset({"/api/tactical/formation/m1"}))
    with pytest.raises(ProviderUnavailableError):
        await adapter.get_match_analysis("m1")


@pytest.mark.asyncio
async def test_backend_unreachable_on_player_intelligence_raises() -> None:
    adapter = _adapter(unreachable_paths=frozenset({"/api/player_intelligence/m1"}))
    with pytest.raises(ProviderUnavailableError):
        await adapter.get_match_analysis("m1")


@pytest.mark.asyncio
async def test_get_player_analysis_scopes_to_the_requested_player() -> None:
    adapter = _adapter()
    analysis = await adapter.get_player_analysis("m1", 7)

    assert len(analysis.player_metrics) == 1
    assert analysis.player_metrics[0].metric_name == "decision_making_score"
    assert analysis.player_metrics[0].player_id == 7
    assert any(m.metric_name == "compactness_score" for m in analysis.team_metrics)
