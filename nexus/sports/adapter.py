from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from nexus.core.exceptions import ProviderUnavailableError


@dataclass
class SportsMetric:
    metric_name: str
    value: float | str | None
    method: str
    confidence: str
    sample_size: int
    sub_scores: dict[str, Any]
    player_id: int | None = None

    @property
    def is_available(self) -> bool:
        return self.value is not None and self.confidence != "low_upstream_confidence"


@dataclass
class MatchAnalysis:
    match_id: str
    team_metrics: list[SportsMetric]
    player_metrics: list[SportsMetric]
    unavailable: list[SportsMetric] = field(default_factory=list)
    coverage: float = 0.0


def _partition(metrics: list[SportsMetric]) -> tuple[list[SportsMetric], list[SportsMetric]]:
    available = [m for m in metrics if m.is_available]
    unavailable = [m for m in metrics if not m.is_available]
    return available, unavailable


def _coverage(total: int, available: int) -> float:
    return (available / total) if total > 0 else 0.0


class SportsDataAdapter(ABC):
    @abstractmethod
    async def get_match_analysis(self, match_id: str) -> MatchAnalysis: ...

    @abstractmethod
    async def get_player_analysis(self, match_id: str, player_id: int) -> MatchAnalysis: ...


def _to_metric(raw: dict[str, Any]) -> SportsMetric:
    return SportsMetric(
        metric_name=raw["metric_name"],
        value=raw.get("value"),
        method=raw["method"],
        confidence=raw["confidence"],
        sample_size=raw["sample_size"],
        sub_scores=raw.get("sub_scores") or {},
        player_id=raw.get("player_id"),
    )


class HttpSportsDataAdapter(SportsDataAdapter):
    """Consumes the EXISTING football computer-vision backend
    (backend/api/*) over HTTP rather than importing its models or querying
    its database directly. This is the one integration seam nexus/ has
    into that system — HTTP keeps the two codebases decoupled (NEXUS never
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url, timeout=self._timeout_seconds, transport=self._transport
        )

    async def _get_json(self, client: httpx.AsyncClient, path: str) -> Any:
        try:
            response = await client.get(path)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                f"Football backend at {self._base_url} unreachable or errored for {path}: {exc}"
            ) from exc
        return response.json()

    async def _get_formation(self, client: httpx.AsyncClient, match_id: str) -> SportsMetric | None:
        try:
            response = await client.get(f"/api/tactical/formation/{match_id}")
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                f"Football backend at {self._base_url} unreachable for formation/{match_id}: {exc}"
            ) from exc
        if response.status_code == 404:
            return None
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderUnavailableError(
                f"Football backend at {self._base_url} errored for formation/{match_id}: {exc}"
            ) from exc
        return _to_metric(response.json())

    async def _team_metrics(self, client: httpx.AsyncClient, match_id: str) -> list[SportsMetric]:
        formation = await self._get_formation(client, match_id)
        team_shape_raw = await self._get_json(client, f"/api/tactical/team_shape/{match_id}")
        metrics = [_to_metric(item) for item in team_shape_raw]
        if formation is not None:
            metrics.insert(0, formation)
        return metrics

    async def get_match_analysis(self, match_id: str) -> MatchAnalysis:
        async with self._client() as client:
            team_metrics = await self._team_metrics(client, match_id)
            players_raw = await self._get_json(client, f"/api/player_intelligence/{match_id}")

        player_metrics = [
            _to_metric({**metric, "player_id": player["player_id"]})
            for player in players_raw
            for metric in player["metrics"]
        ]

        available, unavailable = _partition(team_metrics + player_metrics)
        return MatchAnalysis(
            match_id=match_id,
            team_metrics=team_metrics,
            player_metrics=player_metrics,
            unavailable=unavailable,
            coverage=_coverage(len(team_metrics) + len(player_metrics), len(available)),
        )

    async def get_player_analysis(self, match_id: str, player_id: int) -> MatchAnalysis:
        async with self._client() as client:
            team_metrics = await self._team_metrics(client, match_id)
            player_raw = await self._get_json(
                client, f"/api/player_intelligence/{match_id}/{player_id}"
            )

        player_metrics = [_to_metric({**item, "player_id": player_id}) for item in player_raw]

        available, unavailable = _partition(team_metrics + player_metrics)
        return MatchAnalysis(
            match_id=match_id,
            team_metrics=team_metrics,
            player_metrics=player_metrics,
            unavailable=unavailable,
            coverage=_coverage(len(team_metrics) + len(player_metrics), len(available)),
        )
