"""
Pre-match readiness assessments, read from the football backend over HTTP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from nexus.core.exceptions import ProviderUnavailableError


@dataclass(frozen=True)
class PreMatchAssessment:
    """The backend's assessment contract, carried through unchanged."""

    player_id: str
    match_id: str | None
    physical_readiness: float
    fatigue_score: float
    recovery_score: float
    performance_risk: str
    workload_risk: str
    key_positive_factors: list[str]
    key_negative_factors: list[str]
    method: str
    schema_version: str
    computed_at: str
    data_source: str = "self_reported"
    notes: str | None = None
    disclaimer: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "PreMatchAssessment":
        return cls(
            player_id=payload["player_id"],
            match_id=payload.get("match_id"),
            physical_readiness=payload["physical_readiness"],
            fatigue_score=payload["fatigue_score"],
            recovery_score=payload["recovery_score"],
            performance_risk=payload["performance_risk"],
            workload_risk=payload["workload_risk"],
            key_positive_factors=list(payload.get("key_positive_factors") or []),
            key_negative_factors=list(payload.get("key_negative_factors") or []),
            method=payload["method"],
            schema_version=payload["schema_version"],
            computed_at=str(payload["computed_at"]),
            data_source=payload.get("data_source", "self_reported"),
            notes=payload.get("notes"),
            disclaimer=payload.get("disclaimer", ""),
            raw=payload,
        )

    def as_prompt_context(self) -> str:
        """The exact text handed to the LLM. Every number it is allowed to
        mention appears here; anything absent is something it must say it
        does not know rather than estimate."""
        positives = "\n".join(f"  - {item}" for item in self.key_positive_factors) or "  - none"
        negatives = "\n".join(f"  - {item}" for item in self.key_negative_factors) or "  - none"
        lines = [
            f"Player: {self.player_id}",
            f"Match: {self.match_id or 'not linked to a match yet'}",
            f"Physical readiness: {self.physical_readiness}/100",
            f"Fatigue score: {self.fatigue_score}/100 (higher = more fatigued)",
            f"Recovery score: {self.recovery_score}/100",
            f"Performance risk: {self.performance_risk}",
            f"Workload risk: {self.workload_risk}",
            "Positive factors:",
            positives,
            "Negative factors:",
            negatives,
            f"Scoring method: {self.method} (schema {self.schema_version})",
            f"Data source: {self.data_source}",
            f"Computed at: {self.computed_at}",
        ]
        if self.notes:
            lines.append(
                f"Player's free-text note (informational only, not scored): {self.notes}"
            )
        return "\n".join(lines)


class PreMatchHealthClient:
    """Reads GET /api/prematch_health/... from the football backend."""

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

    async def _get(self, path: str) -> dict[str, Any] | None:
        """Returns the parsed body, or None on a 404."""
        async with self._client() as client:
            try:
                response = await client.get(path)
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

    async def get_latest(self, player_id: str) -> PreMatchAssessment | None:
        """Most recent assessment for this player, or None if they have not
        submitted a questionnaire yet."""
        payload = await self._get(f"/api/prematch_health/{player_id}/latest")
        return PreMatchAssessment.from_json(payload) if payload is not None else None

    async def get_assessment(
        self, player_id: str, assessment_id: str
    ) -> PreMatchAssessment | None:
        """One specific historical assessment, or None if it does not exist
        for this player."""
        payload = await self._get(
            f"/api/prematch_health/{player_id}/{assessment_id}/assessment"
        )
        return PreMatchAssessment.from_json(payload) if payload is not None else None
