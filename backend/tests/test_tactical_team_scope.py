"""Team scoping on /api/tactical/formation and /api/tactical/team_shape.

Regression: both routes defaulted team_id to "unassigned", but the pipeline
writes "team-home"/"team-away" whenever team assignment splits the players --
the normal case. A request that did not name a team therefore got a 404 (or an
empty list) for every properly processed match.
"""

import os
import tempfile
import uuid

import pytest

_TEST_DB_DIR = tempfile.mkdtemp(prefix="tactical_team_scope_tests_")
os.environ["DATABASE_URL"] = f"sqlite:///{os.path.join(_TEST_DB_DIR, 'test_tactical.db')}"

from fastapi.testclient import TestClient  # noqa: E402

from backend.api.main import app  # noqa: E402
from backend.database.models import (  # noqa: E402
    Match,
    MetricConfidence,
    MetricMethod,
    TeamMetric,
)
from backend.database.session import Base, SessionLocal, engine  # noqa: E402


@pytest.fixture(scope="module")
def client():
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as test_client:
        yield test_client


def _match_with(rows: list[tuple[str, str, object]]) -> str:
    """Create a match (unique id, so no cross-test cache hits) with the given
    (team_id, metric_name, value) team metrics."""
    db = SessionLocal()
    try:
        match = Match(match_id=str(uuid.uuid4()), home_team="H", away_team="A",
                      video_path="x.mp4", duration=0.0)
        db.add(match)
        db.flush()
        for team_id, name, value in rows:
            db.add(TeamMetric(
                match_id=match.match_id,
                team_id=team_id,
                metric_name=name,
                value_label=value if isinstance(value, str) else None,
                value_numeric=value if isinstance(value, (int, float)) else None,
                method=MetricMethod.heuristic_proxy,
                confidence=MetricConfidence.normal,
                sample_size=10,
                sub_scores={},
                schema_version="v3",
            ))
        db.commit()
        return match.match_id
    finally:
        db.close()


SPLIT = [
    ("team-home", "formation", "4-3-3"),
    ("team-away", "formation", "4-4-2"),
    ("team-home", "compactness_score", 71.0),
    ("team-away", "compactness_score", 64.0),
]


def test_formation_without_team_finds_a_split_match(client):
    """The bug: this used to 404 because it looked for "unassigned"."""
    match_id = _match_with(SPLIT)
    response = client.get(f"/api/tactical/formation/{match_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["team_id"] == "team-home"
    assert body["value"] == "4-3-3"


def test_formation_for_a_named_team(client):
    match_id = _match_with(SPLIT)
    body = client.get(f"/api/tactical/formation/{match_id}", params={"team_id": "team-away"}).json()
    assert body["team_id"] == "team-away"
    assert body["value"] == "4-4-2"


def test_formation_when_teams_were_not_split(client):
    match_id = _match_with([("unassigned", "formation", "4-2-3-1")])
    body = client.get(f"/api/tactical/formation/{match_id}").json()
    assert body["team_id"] == "unassigned"


def test_formation_is_still_404_when_nothing_was_computed(client):
    """No placeholder formation, ever."""
    match_id = _match_with([])
    assert client.get(f"/api/tactical/formation/{match_id}").status_code == 404


def test_formation_for_a_team_with_no_row_is_404(client):
    match_id = _match_with([("unassigned", "formation", "4-2-3-1")])
    response = client.get(f"/api/tactical/formation/{match_id}", params={"team_id": "team-home"})
    assert response.status_code == 404


def test_team_shape_without_team_returns_every_team(client):
    """The bug: this used to return [] for a split match."""
    match_id = _match_with(SPLIT)
    rows = client.get(f"/api/tactical/team_shape/{match_id}").json()
    assert {(r["team_id"], r["value"]) for r in rows} == {("team-home", 71.0), ("team-away", 64.0)}
    assert [r["team_id"] for r in rows] == ["team-home", "team-away"], "home first, deterministic order"


def test_team_shape_for_a_named_team(client):
    match_id = _match_with(SPLIT)
    rows = client.get(f"/api/tactical/team_shape/{match_id}", params={"team_id": "team-away"}).json()
    assert [(r["team_id"], r["value"]) for r in rows] == [("team-away", 64.0)]


def test_team_shape_excludes_formation_rows(client):
    match_id = _match_with(SPLIT)
    rows = client.get(f"/api/tactical/team_shape/{match_id}").json()
    assert all(r["metric_name"] != "formation" for r in rows)
