"""Tests for the app-wide API key dependency (backend/auth/api_key.py).

The two behaviours that matter: enforcement is completely inert when
SSC_API_KEY is unset (so the documented local-dev flow in RUN.md is unchanged),
and it is total when the variable is set (so a router added later cannot ship
unauthenticated by omission).
"""

import os
import tempfile

import pytest

_TEST_DB_DIR = tempfile.mkdtemp(prefix="api_key_auth_tests_")
os.environ["DATABASE_URL"] = f"sqlite:///{os.path.join(_TEST_DB_DIR, 'test_auth.db')}"

from fastapi.testclient import TestClient  # noqa: E402

from backend.api.main import app  # noqa: E402
from backend.auth.api_key import API_KEY_ENV, API_KEY_HEADER, auth_enabled  # noqa: E402

KEY = "correct-horse-battery-staple"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, KEY)


def test_auth_disabled_when_env_unset(no_key):
    assert auth_enabled() is False


def test_auth_enabled_when_env_set(with_key):
    assert auth_enabled() is True


def test_blank_key_does_not_enable_auth(monkeypatch):
    """A variable set to whitespace is an operator mistake, not a secret. It
    must not half-enable auth with an unguessable-but-empty key."""
    monkeypatch.setenv(API_KEY_ENV, "   ")
    assert auth_enabled() is False


def test_request_passes_without_header_when_auth_disabled(client, no_key):
    assert client.get("/health").status_code == 200
    # 404 is the right answer for a job that does not exist; the point is
    # that the request was not turned away at the door.
    assert client.get("/api/processing/no-such-job").status_code == 404


def test_request_rejected_without_header_when_auth_enabled(client, with_key):
    response = client.get("/api/processing/no-such-job")
    assert response.status_code == 401
    assert API_KEY_HEADER in response.json()["detail"]


def test_request_accepted_with_correct_header(client, with_key):
    assert client.get("/api/processing/no-such-job", headers={API_KEY_HEADER: KEY}).status_code != 401


def test_request_rejected_with_wrong_header(client, with_key):
    assert client.get("/api/processing/no-such-job", headers={API_KEY_HEADER: "wrong"}).status_code == 401


def test_health_stays_public_so_probes_need_no_secret(client, with_key):
    """Container HEALTHCHECKs and load balancer probes hit /health without the
    key. It returns no data, so leaving it open costs nothing."""
    assert client.get("/health").status_code == 200


def test_upload_endpoint_is_also_protected(client, with_key):
    """The constructor-level dependency must cover POST routes too, not just
    the GETs above -- upload is the expensive one to leave open."""
    response = client.post(
        "/api/videos/upload",
        files={"file": ("clip.mp4", b"fake", "video/mp4")},
    )
    assert response.status_code == 401


def test_media_route_accepts_key_as_query_param(client, with_key):
    """<video src> cannot send headers. 404 (no such video) proves the request
    got past auth; 401 would mean playback breaks whenever auth is on."""
    response = client.get("/api/videos/no-such-video/file", params={"api_key": KEY})
    assert response.status_code == 404


def test_query_param_key_is_rejected_off_the_media_routes(client, with_key):
    response = client.get("/api/processing/no-such-job", params={"api_key": KEY})
    assert response.status_code == 401


def test_media_route_still_rejects_a_wrong_query_key(client, with_key):
    response = client.get("/api/videos/no-such-video/processed", params={"api_key": "wrong"})
    assert response.status_code == 401
