"""
GET /api/matches/{match_id}/events.

WHY THESE EXIST. The pipeline has written Event rows -- passes, shots,
turnovers, first touches -- since Phase 3, and there was no endpoint anywhere
that read them back. `events` was a table the product could not see: the
pass/shot/turnover heuristics ran on every job and produced output nothing
could display. These tests pin the endpoint that closed that gap, and in
particular pin the `space` contract, which is the part that must not drift.

`space` says whether possession was resolved in pitch metres (calibration
validated) or in image pixels (it did not, and the bounding-box-height
fallback ran instead). Those are not interchangeable readings, and the whole
point of reporting it is that a consumer must not be able to treat the weaker
one as the stronger by accident.
"""

from __future__ import annotations

import os
import tempfile

os.environ.setdefault(
    "DATABASE_URL",
    f"sqlite:///{os.path.join(tempfile.mkdtemp(), 'events_api.db')}",
)

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.database.models import Event, Match, new_id
from backend.database.session import Base, SessionLocal, engine

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_tables():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        db.query(Event).delete()
        db.query(Match).delete()
        db.commit()
    finally:
        db.close()
    yield


def _match(**kwargs) -> str:
    db = SessionLocal()
    try:
        match = Match(home_team="Home", away_team="Away", video_path="/tmp/x.mp4",
                      duration=0.0, **kwargs)
        db.add(match)
        db.commit()
        return match.match_id
    finally:
        db.close()


def _event(match_id: str, **kwargs):
    db = SessionLocal()
    try:
        defaults = dict(event_type="pass", timestamp=1.0, player_id=1)
        defaults.update(kwargs)
        db.add(Event(match_id=match_id, **defaults))
        db.commit()
    finally:
        db.close()


def test_unknown_match_is_404_not_an_empty_list():
    """An empty list would say "this match happened and had no events", which
    is a different fact from "there is no such match"."""
    response = client.get(f"/api/matches/{new_id()}/events")

    assert response.status_code == 404


def test_a_match_with_no_events_reports_an_honest_empty_set():
    match_id = _match()

    body = client.get(f"/api/matches/{match_id}/events").json()

    assert body["total"] == 0
    assert body["events"] == []
    assert body["by_type"] == {}
    # Not "pitch": no events means no possession was resolved either way, and
    # claiming the stronger instrument on an empty set would be a free
    # upgrade.
    assert body["space"] == "none"


def test_events_come_back_in_chronological_order():
    match_id = _match()
    _event(match_id, timestamp=5.0, event_type="shot")
    _event(match_id, timestamp=1.0, event_type="pass")
    _event(match_id, timestamp=3.0, event_type="turnover")

    body = client.get(f"/api/matches/{match_id}/events").json()

    assert [e["timestamp"] for e in body["events"]] == [1.0, 3.0, 5.0]


def test_events_carrying_pitch_metres_report_pitch_space():
    match_id = _match()
    _event(match_id, pitch_x_m=10.0, pitch_y_m=30.0, homography_confidence=0.8)

    body = client.get(f"/api/matches/{match_id}/events").json()

    assert body["space"] == "pitch"
    assert body["events"][0]["space"] == "pitch"
    assert body["events"][0]["pitch_x_m"] == 10.0


def test_events_without_pitch_metres_report_image_space():
    """The fallback's events. They are real detections, but nothing about
    them is metric, and the response has to say so."""
    match_id = _match()
    _event(match_id, pitch_x_m=None, pitch_y_m=None,
           metadata_json={"space": "image", "pass_distance_m": 12.0,
                          "pass_distance_m_is_estimate": True})

    body = client.get(f"/api/matches/{match_id}/events").json()

    assert body["space"] == "image"
    event = body["events"][0]
    assert event["space"] == "image"
    assert event["pitch_x_m"] is None
    assert event["metadata"]["pass_distance_m_is_estimate"] is True


def test_a_mixed_match_reports_the_weaker_instrument():
    """One calibrated event among uncalibrated ones must not let the whole set
    be read as metric."""
    match_id = _match()
    _event(match_id, timestamp=1.0, pitch_x_m=10.0, pitch_y_m=30.0)
    _event(match_id, timestamp=2.0, pitch_x_m=None, pitch_y_m=None)

    body = client.get(f"/api/matches/{match_id}/events").json()

    assert body["space"] == "image"
    # Per-event, each one still reports what it individually is.
    assert [e["space"] for e in body["events"]] == ["pitch", "image"]


def test_by_type_counts_the_whole_match_not_the_returned_page():
    match_id = _match()
    for i in range(3):
        _event(match_id, timestamp=float(i), event_type="pass")
    _event(match_id, timestamp=9.0, event_type="shot")

    body = client.get(f"/api/matches/{match_id}/events?limit=1").json()

    assert body["returned"] == 1
    # A truncated list must not report its own length as the total.
    assert body["total"] == 4
    assert body["by_type"] == {"pass": 3, "shot": 1}


def test_the_type_filter_narrows_the_list_but_not_the_counts():
    match_id = _match()
    _event(match_id, timestamp=1.0, event_type="pass")
    _event(match_id, timestamp=2.0, event_type="shot")

    body = client.get(f"/api/matches/{match_id}/events?event_type=shot").json()

    assert [e["event_type"] for e in body["events"]] == ["shot"]
    # A caller filtering to shots still needs to know passes exist.
    assert body["by_type"] == {"pass": 1, "shot": 1}
    assert body["total"] == 2
