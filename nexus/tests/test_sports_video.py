"""VideoAnalysisService against the football backend's REAL routes.

The service previously called `POST /api/video/process` and
`GET /api/video/status/{job_id}`; neither exists. The real routes are
`POST /api/videos/upload` (multipart) and `GET /api/processing/{job_id}`
(backend/api/main.py:105 and :205).

These tests assert the wire contract -- method, path, multipart encoding
and the real response field names from backend/api/schemas.py -- because a
compile-clean client calling a 404 is exactly the failure being fixed. The
mock replies are copied from `VideoUploadResponse` and
`ProcessingStatusResponse`, so if those schemas change these tests fail
rather than the integration silently rotting again.
"""
from __future__ import annotations

import httpx
import pytest

from nexus.core.exceptions import ProviderUnavailableError
from nexus.sports.coach import CoachReport
from nexus.sports.video import VideoAnalysisService, _to_status


class _StubCoach:
    """Stands in for CoachAssistant: build_report() is the terminus of the
    flow under test, and running a real LLM router here would test the
    provider, not the video path."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    async def build_report(self, match_id: str, player_id: int | None = None) -> CoachReport:
        self.calls.append((match_id, player_id))
        return CoachReport(
            match_id=match_id, findings=[], unavailable_metrics=[],
            coverage=0.0, narrative="stub narrative", model_used="stub",
        )


UPLOAD_RESPONSE = {
    "video_id": "vid-1", "job_id": "job-1", "match_id": "match-backend-assigned",
    "filename": "clip.mp4", "status": "queued", "message": "Video uploaded.",
}


def _status_response(status: str, progress: int, **extra):
    body = {
        "job_id": "job-1", "video_id": "vid-1", "match_id": "match-backend-assigned",
        "status": status, "progress": progress, "message": "working",
        "error": None, "created_at": "2026-08-13T00:00:00", "updated_at": "2026-08-13T00:00:01",
        "started_at": None, "completed_at": None,
    }
    body.update(extra)
    return body


@pytest.fixture
def clip(tmp_path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"\x00\x00\x00\x18ftypmp42fake-video-bytes")
    return p


def _service(handler, coach=None):
    return VideoAnalysisService(
        "http://backend.test", coach or _StubCoach(),
        poll_interval_seconds=0.0, max_wait_seconds=5.0,
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.asyncio
async def test_submit_posts_multipart_to_the_real_upload_route(clip):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["content_type"] = request.headers.get("content-type", "")
        seen["body"] = request.content
        return httpx.Response(201, json=UPLOAD_RESPONSE)

    status = await _service(handler).submit(video_path_or_url=str(clip), match_id="requested-id")

    assert seen["method"] == "POST"
    assert seen["path"] == "/api/videos/upload"          # not /api/video/process
    assert seen["content_type"].startswith("multipart/form-data")
    assert b"fake-video-bytes" in seen["body"]
    assert b'name="file"' in seen["body"]
    assert b'name="metadata"' in seen["body"]
    # The backend creates the Match row, so its id wins over the caller's.
    assert status.match_id == "match-backend-assigned"
    assert status.job_id == "job-1"
    assert status.state == "queued"


@pytest.mark.asyncio
async def test_status_gets_the_real_processing_route(clip):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, json=_status_response("processing", 45))

    status = await _service(handler).status("job-1")

    assert seen["path"] == "/api/processing/job-1"        # not /api/video/status/job-1
    assert status.state == "processing"
    # 45 (percent, int) must arrive as 0.45, since the dataclass renders it
    # with `{progress:.0%}`.
    assert status.progress == pytest.approx(0.45)


@pytest.mark.asyncio
async def test_full_flow_upload_poll_then_build_report(clip):
    coach = _StubCoach()
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if request.url.path == "/api/videos/upload":
            return httpx.Response(201, json=UPLOAD_RESPONSE)
        n = sum(1 for c in calls if c.startswith("GET"))
        if n < 2:
            return httpx.Response(200, json=_status_response("processing", 60))
        return httpx.Response(200, json=_status_response("completed", 100))

    report = await _service(handler, coach).analyze(
        video_path_or_url=str(clip), match_id="requested-id", player_id=7,
    )

    assert calls[0] == "POST /api/videos/upload"
    assert all(c == "GET /api/processing/job-1" for c in calls[1:])
    # build_report ran against the BACKEND's match id, not the requested one.
    assert coach.calls == [("match-backend-assigned", 7)]
    assert report.match_id == "match-backend-assigned"
    assert report.narrative == "stub narrative"


@pytest.mark.asyncio
async def test_failed_job_raises_with_the_backend_error_text(clip):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/videos/upload":
            return httpx.Response(201, json=UPLOAD_RESPONSE)
        return httpx.Response(200, json=_status_response(
            "failed", 30, error="PipelineAssetError: ball checkpoint missing"))

    with pytest.raises(ProviderUnavailableError, match="ball checkpoint missing"):
        await _service(handler).analyze(video_path_or_url=str(clip), match_id="m")


@pytest.mark.asyncio
async def test_non_file_source_is_refused_rather_than_silently_fetched():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request should be made for a non-file source")

    with pytest.raises(ProviderUnavailableError, match="not a readable local file"):
        await _service(handler).submit(
            video_path_or_url="https://example.test/clip.mp4", match_id="m")


def test_to_status_reads_status_not_state():
    """Regression guard: the old implementation looked for `state` first and
    fell through to the "queued" default on every real response, so a
    completed job never looked terminal."""
    s = _to_status(_status_response("completed", 100), fallback_match_id="")
    assert s.state == "completed" and s.is_terminal
    assert s.progress == pytest.approx(1.0)


def test_to_status_prefers_error_over_message_for_detail():
    s = _to_status(_status_response("failed", 10, error="boom"), fallback_match_id="")
    assert s.detail == "boom"


class _ExplodingCoach:
    """A coach that always fails — stands in for a provider outage, a router
    with no model, or a bug in report building."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.calls = 0
        self._exc = exc or RuntimeError("provider is down")

    async def build_report(self, match_id: str, player_id: int | None = None):
        self.calls += 1
        raise self._exc


@pytest.mark.asyncio
async def test_coach_failure_after_processing_never_fails_the_job():
    """The coach runs downstream of processing, so its failure must surface
    as "no report" rather than turning a completed job into a failed one."""
    coach = _ExplodingCoach()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_response("completed", 100))

    report = await _service(handler, coach).coach_report_after_processing(
        job_id="job-1", match_id="match-1")

    assert report is None
    assert coach.calls == 1


@pytest.mark.asyncio
async def test_provider_unavailable_after_processing_is_swallowed_too():
    coach = _ExplodingCoach(ProviderUnavailableError("no model available"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_response("completed", 100))

    assert await _service(handler, coach).coach_report_after_processing(
        job_id="job-1", match_id="match-1") is None


@pytest.mark.asyncio
async def test_no_coach_report_is_built_for_a_failed_job():
    coach = _StubCoach()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_response("failed", 30, error="pipeline died"))

    report = await _service(handler, coach).coach_report_after_processing(
        job_id="job-1", match_id="match-1")

    assert report is None
    assert coach.calls == []


@pytest.mark.asyncio
async def test_coach_report_after_processing_returns_the_report_on_success():
    coach = _StubCoach()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_response("completed", 100))

    report = await _service(handler, coach).coach_report_after_processing(
        job_id="job-1", match_id="match-1", player_id=9)

    assert report is not None
    assert report.narrative == "stub narrative"
    assert coach.calls == [("match-backend-assigned", 9)]


@pytest.mark.asyncio
async def test_a_stuck_job_yields_no_report_rather_than_raising():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_response("processing", 50))

    service = VideoAnalysisService(
        "http://backend.test", _StubCoach(),
        poll_interval_seconds=0.0, max_wait_seconds=0.0,
        transport=httpx.MockTransport(handler),
    )

    assert await service.coach_report_after_processing(
        job_id="job-1", match_id="match-1") is None


@pytest.mark.asyncio
async def test_analyze_upload_posts_bytes_and_returns_a_report():
    coach = _StubCoach()
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/videos/upload":
            seen["body"] = request.content
            seen["content_type"] = request.headers.get("content-type", "")
            return httpx.Response(201, json=UPLOAD_RESPONSE)
        return httpx.Response(200, json=_status_response("completed", 100))

    report = await _service(handler, coach).analyze_upload(
        filename="clip.mp4", content=b"raw-upload-bytes", match_id="requested-id", player_id=3)

    assert seen["content_type"].startswith("multipart/form-data")
    assert b"raw-upload-bytes" in seen["body"]
    assert coach.calls == [("match-backend-assigned", 3)]
    assert report.match_id == "match-backend-assigned"
