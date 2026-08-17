from __future__ import annotations

import asyncio
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

from nexus.core.exceptions import ProviderUnavailableError
from nexus.logging_setup.logger import get_logger
from nexus.sports.coach import CoachAssistant, CoachReport

logger = get_logger("sports.video")

VideoJobState = Literal["queued", "processing", "completed", "failed"]

_TERMINAL_STATES: frozenset[str] = frozenset({"completed", "failed"})
_DEFAULT_POLL_INTERVAL_SECONDS = 5.0
_DEFAULT_MAX_WAIT_SECONDS = 1800.0


@dataclass
class VideoJobStatus:
    job_id: str
    match_id: str
    state: VideoJobState
    progress: float = 0.0
    detail: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.state in _TERMINAL_STATES


class VideoAnalysisService:
    """Submits footage to the football backend's EXISTING processing
    endpoint, polls until it finishes, then hands the resulting match_id to
    the Group C adapter path for a CoachReport.
    """

    def __init__(
        self,
        base_url: str,
        coach_assistant: CoachAssistant,
        *,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        max_wait_seconds: float = _DEFAULT_MAX_WAIT_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._coach_assistant = coach_assistant
        self._timeout_seconds = timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._max_wait_seconds = max_wait_seconds
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url, timeout=self._timeout_seconds, transport=self._transport
        )

    async def submit(self, *, video_path_or_url: str, match_id: str) -> VideoJobStatus:
        """Uploads the clip to `POST /api/videos/upload`."""
        source = Path(video_path_or_url)
        if not source.is_file():
            raise ProviderUnavailableError(
                f"Video source is not a readable local file: {video_path_or_url!r}. "
                "The backend's /api/videos/upload route takes a multipart file; "
                "fetch remote media to disk before calling analyze()."
            )

        content_type = mimetypes.guess_type(source.name)[0] or "video/mp4"
        async with self._client() as client:
            try:
                with source.open("rb") as handle:
                    response = await client.post(
                        "/api/videos/upload",
                        files={"file": (source.name, handle, content_type)},
                        data={"metadata": json.dumps({"requested_match_id": match_id})},
                    )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise ProviderUnavailableError(
                    f"Football backend at {self._base_url} could not accept the video job: {exc}"
                ) from exc

            status = _to_status(response.json(), fallback_match_id=match_id)
            if match_id and status.match_id and status.match_id != match_id:
                logger.info(
                    "backend assigned match_id %s for the uploaded clip; the requested "
                    "id %s is not used (the Match row is created at upload time)",
                    status.match_id, match_id,
                )
            return status

    async def status(self, job_id: str) -> VideoJobStatus:
        async with self._client() as client:
            try:
                response = await client.get(f"/api/processing/{job_id}")
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise ProviderUnavailableError(
                    f"Football backend at {self._base_url} could not report status for "
                    f"job {job_id}: {exc}"
                ) from exc
            return _to_status(response.json(), fallback_match_id="")

    async def wait_for_completion(self, job_id: str, *, match_id: str) -> VideoJobStatus:
        """Polls until terminal or the wall-clock budget runs out. The
        budget is a hard cap rather than an unbounded wait: a stuck
        pipeline should surface as an error the caller can act on, not as a
        request that never returns."""
        waited = 0.0
        latest = VideoJobStatus(job_id=job_id, match_id=match_id, state="queued")

        while waited < self._max_wait_seconds:
            latest = await self.status(job_id)
            if latest.is_terminal:
                return latest
            await asyncio.sleep(self._poll_interval_seconds)
            waited += self._poll_interval_seconds

        raise ProviderUnavailableError(
            f"Video job {job_id} did not finish within {self._max_wait_seconds:.0f}s "
            f"(last state={latest.state}, progress={latest.progress:.0%})."
        )

    async def analyze(
        self, *, video_path_or_url: str, match_id: str, player_id: int | None = None
    ) -> CoachReport:
        submitted = await self.submit(video_path_or_url=video_path_or_url, match_id=match_id)
        resolved_match_id = submitted.match_id or match_id

        if not submitted.is_terminal:
            final = await self.wait_for_completion(submitted.job_id, match_id=resolved_match_id)
        else:
            final = submitted

        if final.state == "failed":
            raise ProviderUnavailableError(
                f"Video processing failed for match {resolved_match_id}: "
                f"{final.detail or 'no detail reported by the backend'}"
            )

        logger.info(
            "video job %s completed for match %s; building coach report",
            final.job_id,
            resolved_match_id,
        )
        return await self._coach_assistant.build_report(resolved_match_id, player_id)


def _to_status(payload: Any, *, fallback_match_id: str) -> VideoJobStatus:
    """Maps both real backend payloads onto one status object."""
    if not isinstance(payload, dict):
        raise ProviderUnavailableError(
            f"Football backend returned a non-object video job payload: {payload!r}"
        )
    state = str(payload.get("status", payload.get("state", "queued"))).lower()
    if state not in ("queued", "processing", "completed", "failed"):
        state = "processing"

    raw_progress = float(payload.get("progress", 0.0) or 0.0)
    progress = raw_progress / 100.0 if raw_progress > 1.0 else raw_progress

    match_id = payload.get("match_id") or fallback_match_id
    return VideoJobStatus(
        job_id=str(payload.get("job_id", payload.get("id", ""))),
        match_id=str(match_id or ""),
        state=state,  # type: ignore[arg-type]
        progress=max(0.0, min(1.0, progress)),
        detail=str(payload.get("error") or payload.get("message")
                   or payload.get("detail") or ""),
    )
