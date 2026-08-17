"""VideoAnalysisService against the football backend's REAL routes."""
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
    assert seen["path"] == "/api/videos/upload"
    assert seen["content_type"].startswith("multipart/form-data")
    assert b"fake-video-bytes" in seen["body"]
    assert b'name="file"' in seen["body"]
    assert b'name="metadata"' in seen["body"]
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

    assert seen["path"] == "/api/processing/job-1"
    assert status.state == "processing"
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
