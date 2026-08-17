"""
Mental-readiness assessments, read from the football backend over HTTP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from nexus.core.exceptions import ProviderUnavailableError


@dataclass(frozen=True)
class PsychologyAssessment:
    """The backend's assessment contract, carried through unchanged."""

    player_id: str
    match_id: str | None
    mental_readiness: int
    focus: int
    confidence: int
    stress: int
    pressure_risk: str
    mental_performance_risk: str
    factors: dict[str, str]
    method: str
    confidence_level: str
    schema_version: str
    computed_at: str
    data_source: str = "self_reported"
    historical_context: list[str] = field(default_factory=list)
    disclaimer: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "PsychologyAssessment":
        return cls(
            player_id=payload["player_id"],
            match_id=payload.get("match_id"),
            mental_readiness=payload["mental_readiness"],
            focus=payload["focus"],
            confidence=payload["confidence"],
            stress=payload["stress"],
            pressure_risk=payload["pressure_risk"],
            mental_performance_risk=payload["mental_performance_risk"],
            factors=dict(payload.get("factors") or {}),
            method=payload["method"],
            confidence_level=payload.get("confidence_level", ""),
            schema_version=payload["schema_version"],
            computed_at=str(payload["computed_at"]),
            data_source=payload.get("data_source", "self_reported"),
            historical_context=list(payload.get("historical_context") or []),
            disclaimer=payload.get("disclaimer", ""),
            raw=payload,
        )


class PsychologyClient:
    """Reads GET /api/psychology/... from the football backend."""

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
            base_url=self._base_url,
            timeout=self._timeout_seconds,
            transport=self._transport,
        )

    async def _get(self, path: str, params: dict | None = None) -> dict[str, Any] | None:
        """Returns the parsed body, or None on a 404."""
        async with self._client() as client:
            try:
                response = await client.get(path, params=params)
            except httpx.HTTPError as exc:
                raise ProviderUnavailableError(
                    f"Football backend at {self._base_url} unreachable for {path}: {exc}"
                ) from exc
            if response.status_code == 404:
                return None
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise ProviderUnavailableError(
                    f"Football backend at {self._base_url} errored for {path}: {exc}"
                ) from exc
            return response.json()

    async def get_latest(
        self, player_id: str, match_id: str | None = None
    ) -> PsychologyAssessment | None:
        """Most recent assessment for this player, or None if they have not
        submitted a questionnaire yet."""
        payload = await self._get(
            f"/api/psychology/{player_id}/latest",
            params={"match_id": match_id} if match_id else None,
        )
        return PsychologyAssessment.from_json(payload) if payload is not None else None

    async def get_assessment(
        self, player_id: str, assessment_id: str
    ) -> PsychologyAssessment | None:
        """One specific historical assessment, or None if it does not exist for
        this player."""
        payload = await self._get(f"/api/psychology/{player_id}/{assessment_id}")
        return PsychologyAssessment.from_json(payload) if payload is not None else None
