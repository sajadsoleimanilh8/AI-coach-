"""
API tests for POST /api/videos/upload (backend/api/main.py::upload_video),
covering the broker-unavailable failure path in particular.
"""

import os
import sys
import tempfile

import pytest
from kombu.exceptions import OperationalError

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

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
    with TestClient(app) as test_client:
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
    """

    def raise_broker_down(*args, **kwargs):
        raise OperationalError("Error 111 connecting to localhost:6379. Connection refused.")

    monkeypatch.setattr(api_main.process_video_job, "delay", raise_broker_down)

    response = client.post("/api/videos/upload", files=UPLOAD_FILES, data=UPLOAD_DATA)

    assert response.status_code == 201, response.text

    body = response.json()
    assert body["status"] == ProcessingStatus.failed.value
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
