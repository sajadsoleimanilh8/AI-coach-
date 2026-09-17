"""
API tests for POST /api/videos/upload (backend/api/main.py::upload_video),
covering the broker-unavailable failure path in particular.

Runs the real FastAPI app against a throwaway SQLite file: DATABASE_URL is
pointed at a temp path BEFORE backend.database.session is first imported, so
the app's own startup Base.metadata.create_all() builds the schema there and
the developer's sports_strategy.db is never touched. That also means these
tests exercise the real table definitions, not a hand-built test schema -- if
a column or FK is wrong in models.py, it fails here.

Same structure as the sibling test_psychology_api.py / test_prematch_health_api.py
in this directory.

Neither test needs a running Redis: process_video_job.delay is patched out in
both directions -- raising kombu's OperationalError (broker down) and
succeeding silently (broker up).
"""

import os
import tempfile

import pytest
from kombu.exceptions import OperationalError

# Must happen before any backend.database import binds the engine.
_TEST_DB_DIR = tempfile.mkdtemp(prefix="video_upload_tests_")
_TEST_DB_PATH = os.path.join(_TEST_DB_DIR, "test_video_upload.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB_PATH}"

from fastapi.testclient import TestClient  # noqa: E402

from backend.api import main as api_main  # noqa: E402
from backend.api.main import app  # noqa: E402
from backend.database.models import (  # noqa: E402
    Match,
    ProcessingJob,
    ProcessingStatus,
    Video,
)
from backend.database.session import Base, SessionLocal, engine  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:  # triggers the startup create_all
        yield test_client


@pytest.fixture(autouse=True)
def clean_tables():
    """Each test starts from an empty upload schema so the "exactly one job"
    assertions below cannot be polluted by an earlier test's upload. Only this
    module's tables are cleared -- nothing else in the schema is touched."""
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        session.query(ProcessingJob).delete()
        session.query(Video).delete()
        session.query(Match).delete()
        session.commit()
    finally:
        session.close()
    yield


@pytest.fixture(autouse=True)
def temp_upload_dir(tmp_path, monkeypatch):
    """Keep the bytes the endpoint writes out of the repo's storage/uploads."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(api_main, "UPLOAD_DIR", upload_dir)
    return upload_dir


# Matches the endpoint's actual multipart contract: a required `file` part
# whose content_type is in ALLOWED_VIDEO_TYPES, plus an optional `metadata`
# form field carrying JSON.
UPLOAD_FILES = {"file": ("training_clip.mp4", b"\x00\x00\x00\x18ftypmp42fake", "video/mp4")}
UPLOAD_DATA = {"metadata": '{"team": "Blue FC", "opponent": "Red United"}'}


def _only_job():
    session = SessionLocal()
    try:
        return session.query(ProcessingJob).one()
    finally:
        session.close()


def test_upload_marks_job_failed_when_broker_unavailable(client, monkeypatch):
    """Regression: with Redis down, process_video_job.delay() raises
    kombu.exceptions.OperationalError. That used to escape the endpoint as a
    raw unhandled 500 and strand the ProcessingJob at status=queued forever --
    a job no worker would ever pick up, that the frontend's status poller
    would nonetheless keep polling indefinitely."""

    def raise_broker_down(*args, **kwargs):
        raise OperationalError("Error 111 connecting to localhost:6379. Connection refused.")

    monkeypatch.setattr(api_main.process_video_job, "delay", raise_broker_down)

    response = client.post("/api/videos/upload", files=UPLOAD_FILES, data=UPLOAD_DATA)

    # Not a raw 500 from an unhandled OperationalError.
    assert response.status_code == 201, response.text

    body = response.json()
    assert body["status"] == ProcessingStatus.failed.value
    # The upload itself succeeded, so the caller still gets the identifiers it
    # needs to reference the failed upload.
    assert body["video_id"]
    assert body["job_id"]
    assert body["match_id"]
    assert body["message"]

    job = _only_job()
    assert job.id == body["job_id"]
    assert job.status == ProcessingStatus.failed
    assert job.status != ProcessingStatus.queued, "job must not be stranded as queued"
    assert job.error
    assert job.completed_at is not None


def test_upload_queues_job_when_broker_available(client, monkeypatch):
    """The success path is unaffected by the failure handling above."""
    enqueued = []

    def record_enqueue(job_id, *args, **kwargs):
        enqueued.append(job_id)

    monkeypatch.setattr(api_main.process_video_job, "delay", record_enqueue)

    response = client.post("/api/videos/upload", files=UPLOAD_FILES, data=UPLOAD_DATA)

    assert response.status_code == 201, response.text

    body = response.json()
    assert body["status"] == ProcessingStatus.queued.value
    assert body["video_id"]
    assert body["job_id"]
    assert body["match_id"]

    job = _only_job()
    assert job.id == body["job_id"]
    assert job.status == ProcessingStatus.queued
    assert job.completed_at is None
    assert enqueued == [body["job_id"]]


# --- upload bounds -------------------------------------------------------
# Regression cover for the unbounded write: the endpoint used to stream the
# whole body through shutil.copyfileobj with no size check, so a single request
# could fill the storage volume.


def test_upload_over_the_limit_is_rejected(client, monkeypatch, temp_upload_dir):
    monkeypatch.setattr(api_main, "MAX_UPLOAD_BYTES", 1024)
    response = client.post(
        "/api/videos/upload",
        files={"file": ("big.mp4", b"\x00" * 4096, "video/mp4")},
        data=UPLOAD_DATA,
    )
    assert response.status_code == 413


def test_rejected_upload_leaves_no_partial_file(client, monkeypatch, temp_upload_dir):
    """The 413 path must clean up after itself, or the limit hands an attacker
    exactly the disk fill it exists to prevent."""
    monkeypatch.setattr(api_main, "MAX_UPLOAD_BYTES", 1024)
    client.post(
        "/api/videos/upload",
        files={"file": ("big.mp4", b"\x00" * 4096, "video/mp4")},
        data=UPLOAD_DATA,
    )
    assert list(temp_upload_dir.iterdir()) == []


def test_upload_under_the_limit_still_succeeds(client, monkeypatch, temp_upload_dir):
    monkeypatch.setattr(api_main, "MAX_UPLOAD_BYTES", 1024 * 1024)
    response = client.post("/api/videos/upload", files=UPLOAD_FILES, data=UPLOAD_DATA)
    assert response.status_code == 201


def test_octet_stream_accepted_only_with_a_video_extension(client, temp_upload_dir):
    """octet-stream used to be a blanket allow, which made the content-type
    allowlist mean nothing. It is now gated on the filename."""
    ok = client.post(
        "/api/videos/upload",
        files={"file": ("clip.mp4", b"fake", "application/octet-stream")},
        data=UPLOAD_DATA,
    )
    assert ok.status_code == 201

    rejected = client.post(
        "/api/videos/upload",
        files={"file": ("payload.exe", b"MZfake", "application/octet-stream")},
        data=UPLOAD_DATA,
    )
    assert rejected.status_code == 415
