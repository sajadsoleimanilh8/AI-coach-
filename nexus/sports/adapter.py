from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from nexus.core.exceptions import ProviderUnavailableError
from nexus.sports.timeline import TIMELINE_PATH, TacticalTimeline, parse_timeline


@dataclass
class SportsMetric:
    metric_name: str
    value: float | str | None
    method: str
    confidence: str
    sample_size: int
    sub_scores: dict[str, Any]
    player_id: int | None = None
    team_id: str | None = None

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
    timeline: TacticalTimeline | None = None
    team_ids: list[str] = field(default_factory=list)


def _partition(metrics: list[SportsMetric]) -> tuple[list[SportsMetric], list[SportsMetric]]:
    available = [m for m in metrics if m.is_available]
    unavailable = [m for m in metrics if not m.is_available]
    return available, unavailable


def _coverage(total: int, available: int) -> float:
    return (available / total) if total > 0 else 0.0


def _build_analysis(
    match_id: str,
    team_metrics: list[SportsMetric],
    player_metrics: list[SportsMetric],
    timeline: TacticalTimeline | None,
) -> MatchAnalysis:
    available, unavailable = _partition(team_metrics + player_metrics)
    return MatchAnalysis(
        match_id=match_id,
        team_metrics=team_metrics,
        player_metrics=player_metrics,
        unavailable=unavailable,
        coverage=_coverage(len(team_metrics) + len(player_metrics), len(available)),
        timeline=timeline,
        team_ids=sorted({m.team_id for m in team_metrics if m.team_id}),
    )


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
        team_id=raw.get("team_id"),
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

    async def _get_unscoped_team_metrics(
        self, client: httpx.AsyncClient, match_id: str
    ) -> list[SportsMetric]:
        """Team metrics from the un-scoped route, or [] if it isn't served.

        A backend that predates this route answers 404; that is a "fall back
        to the scoped routes" signal, not a failure, so it must not surface
        as ProviderUnavailableError the way _get_json would.
        """
        try:
            response = await client.get(f"/api/team_intelligence/{match_id}")
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                f"Football backend at {self._base_url} unreachable for "
                f"team_intelligence/{match_id}: {exc}"
            ) from exc
        if response.status_code != 200:
            return []
        try:
            payload = response.json()
        except ValueError:
            return []
        if not isinstance(payload, list):
            return []
        return [_to_metric(item) for item in payload]

    async def _team_metrics(self, client: httpx.AsyncClient, match_id: str) -> list[SportsMetric]:
        """Every team metric for the match, whatever team_id it was stored under.

        The scoped routes (/api/tactical/formation, /team_shape) default to
        team_id="unassigned", but the pipeline writes real ids ("team-home").
        Asking those routes without a team_id therefore returned nothing for
        every genuinely processed match -- the team half of the report went
        silently missing and coverage was computed over players alone.

        /api/team_intelligence/{match_id} is not team-scoped, so it returns
        the rows that actually exist. The scoped routes stay as the fallback
        for a backend that only serves those.
        """
        unscoped = await self._get_unscoped_team_metrics(client, match_id)
        if unscoped:
            return unscoped

        formation = await self._get_formation(client, match_id)
        team_shape_raw = await self._get_json(client, f"/api/tactical/team_shape/{match_id}")
        metrics = [_to_metric(item) for item in team_shape_raw]
        if formation is not None:
            metrics.insert(0, formation)
        return metrics

    async def _timeline(
        self, client: httpx.AsyncClient, match_id: str
    ) -> TacticalTimeline | None:
        """Phase 4's tactical timeline, or None when the backend has none.

        A missing timeline is a normal state (Phase 4 is not implemented
        yet), so a 404 or an unparseable body is not an error -- the caller
        reports the absence rather than failing the whole report.
        """
        try:
            response = await client.get(TIMELINE_PATH.format(match_id=match_id))
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        return parse_timeline(payload, match_id=match_id)

    async def get_match_analysis(self, match_id: str) -> MatchAnalysis:
        async with self._client() as client:
            team_metrics = await self._team_metrics(client, match_id)
            players_raw = await self._get_json(client, f"/api/player_intelligence/{match_id}")
            timeline = await self._timeline(client, match_id)

        player_metrics = [
            _to_metric({**metric, "player_id": player["player_id"]})
            for player in players_raw
            for metric in player["metrics"]
        ]

        return _build_analysis(match_id, team_metrics, player_metrics, timeline)

    async def get_player_analysis(self, match_id: str, player_id: int) -> MatchAnalysis:
        async with self._client() as client:
            team_metrics = await self._team_metrics(client, match_id)
            player_raw = await self._get_json(
                client, f"/api/player_intelligence/{match_id}/{player_id}"
            )
            timeline = await self._timeline(client, match_id)

        player_metrics = [_to_metric({**item, "player_id": player_id}) for item in player_raw]

        return _build_analysis(match_id, team_metrics, player_metrics, timeline)
