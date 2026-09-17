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

    No computer vision lives here, and none should. NEXUS orchestrates and
    narrates; backend/ and ai/ do the vision. The whole point of the HTTP
    seam is that nexus/ never imports from either — which also means the
    football backend can run on another machine entirely.
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
        self._transport = transport  # test seam: inject httpx.MockTransport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url, timeout=self._timeout_seconds, transport=self._transport
        )

    async def submit(self, *, video_path_or_url: str, match_id: str) -> VideoJobStatus:
        """Uploads the clip to `POST /api/videos/upload`.

        That route is **multipart/form-data** (`file`, plus an optional
        `metadata` JSON string) -- not the JSON body this client used to
        send. It is also the point at which the backend CREATES the Match
        row and assigns `match_id` (see backend/api/main.py::upload_video),
        so the caller's `match_id` cannot be honoured on upload: whatever
        the backend returns is authoritative. `match_id` is still accepted
        because it is a useful correlation hint in logs when the two
        disagree, and because `analyze()` falls back to it if the backend
        somehow omits one.
        """
        source = Path(video_path_or_url)
        if not source.is_file():
            # A URL cannot be streamed straight into a multipart upload
            # without first fetching it, and silently fetching arbitrary
            # remote media from inside the orchestration layer is not a
            # decision this client should make on its own.
            raise ProviderUnavailableError(
                f"Video source is not a readable local file: {video_path_or_url!r}. "
                "The backend's /api/videos/upload route takes a multipart file; "
                "fetch remote media to disk before calling analyze()."
            )

        with source.open("rb") as handle:
            return await self.submit_file(
                filename=source.name,
                content=handle.read(),
                match_id=match_id,
                content_type=mimetypes.guess_type(source.name)[0] or "video/mp4",
            )

    async def submit_file(
        self,
        *,
        filename: str,
        content: bytes,
        match_id: str,
        content_type: str | None = None,
    ) -> VideoJobStatus:
        """Uploads bytes already in hand — the path taken when a client POSTs
        a multipart file to nexus rather than naming a file on the backend's
        own disk."""
        resolved_type = content_type or mimetypes.guess_type(filename)[0] or "video/mp4"
        async with self._client() as client:
            try:
                response = await client.post(
                    "/api/videos/upload",
                    files={"file": (filename, content, resolved_type)},
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

    async def analyze_upload(
        self,
        *,
        filename: str,
        content: bytes,
        match_id: str,
        player_id: int | None = None,
        content_type: str | None = None,
    ) -> CoachReport:
        """Full path for an uploaded file: submit -> poll -> CoachReport."""
        submitted = await self.submit_file(
            filename=filename, content=content, match_id=match_id, content_type=content_type
        )
        resolved_match_id = submitted.match_id or match_id

        final = (
            submitted
            if submitted.is_terminal
            else await self.wait_for_completion(submitted.job_id, match_id=resolved_match_id)
        )
        if final.state == "failed":
            raise ProviderUnavailableError(
                f"Video processing failed for match {resolved_match_id}: "
                f"{final.detail or 'no detail reported by the backend'}"
            )

        logger.info(
            "uploaded video job %s completed for match %s; building coach report",
            final.job_id, resolved_match_id,
        )
        return await self._coach_assistant.build_report(resolved_match_id, player_id)

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

    async def coach_report_after_processing(
        self, *, job_id: str, match_id: str, player_id: int | None = None
    ) -> CoachReport | None:
        """Build the coach report for an ALREADY-SUBMITTED job, never raising.

        This is the post-processing hook: processing has its own lifecycle in
        the backend, and the coaching layer sits downstream of it. A provider
        outage, a router with no model, or a bug in report building must
        therefore surface as "no report yet" — never as a failed video job.

        Returns None on any failure, having logged the reason. Callers that
        want the error should use analyze() instead.
        """
        try:
            final = await self.wait_for_completion(job_id, match_id=match_id)
        except ProviderUnavailableError as exc:
            logger.warning(
                "coach report skipped for match %s: job %s never reached a "
                "terminal state (%s)", match_id, job_id, exc,
            )
            return None

        if final.state != "completed":
            logger.info(
                "coach report skipped for match %s: job %s ended in state %s (%s)",
                match_id, job_id, final.state, final.detail or "no detail",
            )
            return None

        resolved_match_id = final.match_id or match_id
        try:
            return await self._coach_assistant.build_report(resolved_match_id, player_id)
        except Exception:
            # Deliberately broad: this runs after processing has already
            # succeeded, so nothing raised here should retroactively turn a
            # completed job into a failed one.
            logger.exception(
                "coach report failed for match %s (job %s); the processing job "
                "is unaffected", resolved_match_id, job_id,
            )
            return None


def _to_status(payload: Any, *, fallback_match_id: str) -> VideoJobStatus:
    """Maps both real backend payloads onto one status object.

    `VideoUploadResponse` and `ProcessingStatusResponse` (backend/api/
    schemas.py) both carry `job_id`, `match_id` and `status`; only the
    status response carries `progress`, `message` and `error`. The field is
    `status`, not `state` -- the old `state`-first lookup silently fell
    through to the default on every real response.
    """
    if not isinstance(payload, dict):
        raise ProviderUnavailableError(
            f"Football backend returned a non-object video job payload: {payload!r}"
        )
    state = str(payload.get("status", payload.get("state", "queued"))).lower()
    if state not in ("queued", "processing", "completed", "failed"):
        state = "processing"

    # ProcessingStatusResponse.progress is an int PERCENT (0-100); this
    # dataclass and its `{progress:.0%}` formatting are a 0-1 fraction.
    # Normalise here rather than at each read site, where "97" would have
    # been rendered as "9700%".
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
