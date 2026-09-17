"""
API tests for the Pre-Match Psychology Intelligence router
(backend/api/psychology.py).

Runs the real FastAPI app against a throwaway SQLite file: DATABASE_URL is
pointed at a temp path BEFORE backend.database.session is first imported, so
the app's own startup Base.metadata.create_all() builds the schema there and
the developer's sports_strategy.db is never touched. That also means these
tests exercise the real table definitions, not a hand-built test schema -- if a
column or FK is wrong in models.py, it fails here.

Same structure as the sibling test_prematch_health_api.py in this directory.
"""

import os
import tempfile

import pytest

# Must happen before any backend.database import binds the engine.
_TEST_DB_DIR = tempfile.mkdtemp(prefix="psychology_tests_")
_TEST_DB_PATH = os.path.join(_TEST_DB_DIR, "test_psychology.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB_PATH}"

from fastapi.testclient import TestClient  # noqa: E402

from backend.api.main import app  # noqa: E402
from backend.database.models import (  # noqa: E402
    Match,
    MetricConfidence,
    MetricMethod,
    PlayerMetric,
    PsychologyAssessment,
)
from backend.database.session import Base, SessionLocal, engine  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:  # triggers the startup create_all
        yield test_client


@pytest.fixture(autouse=True)
def clean_tables():
    """Each test starts from an empty psychology schema so history/latest
    assertions cannot be polluted by an earlier test's submissions. Only this
    module's tables are cleared -- nothing else in the schema is touched."""
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        session.query(PsychologyAssessment).delete()
        session.query(PlayerMetric).delete()
        session.query(Match).delete()
        session.commit()
    finally:
        session.close()
    yield


STRONG_PAYLOAD = {
    "concentration_level": 9,
    "focus_maintenance": 9,
    "mental_clarity_raw": 9,
    "pre_match_stress": 2,
    "importance_pressure": 3,
    "nervousness": 2,
    "performance_confidence": 9,
    "tactical_confidence_raw": 8,
    "match_motivation": 9,
    "competitive_motivation_raw": 9,
    "mistake_recovery_speed": 8,
    "pressure_performance_effect": "improves",
    "post_error_calm": 8,
}

PRESSURED_PAYLOAD = {
    **STRONG_PAYLOAD,
    "concentration_level": 4,
    "focus_maintenance": 4,
    "mental_clarity_raw": 4,
    "pre_match_stress": 9,
    "importance_pressure": 9,
    "nervousness": 9,
    "performance_confidence": 3,
    "tactical_confidence_raw": 3,
    "mistake_recovery_speed": 3,
    "pressure_performance_effect": "reduces",
    "post_error_calm": 3,
}


def _make_match(match_id: str = "match-1") -> str:
    session = SessionLocal()
    try:
        session.add(
            Match(
                match_id=match_id,
                home_team="Home",
                away_team="Away",
                video_path="/tmp/x.mp4",
                duration=90.0,
            )
        )
        session.commit()
    finally:
        session.close()
    return match_id


def _make_player_metric(match_id: str, cv_player_id: int, name: str, value, confidence):
    session = SessionLocal()
    try:
        session.add(
            PlayerMetric(
                match_id=match_id,
                player_id=cv_player_id,
                metric_name=name,
                value=value,
                method=MetricMethod.heuristic_proxy,
                confidence=confidence,
                sample_size=10,
                sub_scores={},
                schema_version="v3",
            )
        )
        session.commit()
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Validation -- Pydantic produces the 422, not hand-rolled checks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_value", [0, 11, -3])
def test_out_of_range_rating_is_rejected_with_422(client, bad_value):
    response = client.post(
        "/api/psychology/p1/submit",
        json={**STRONG_PAYLOAD, "concentration_level": bad_value},
    )
    assert response.status_code == 422


def test_unknown_pressure_effect_is_rejected_with_422(client):
    response = client.post(
        "/api/psychology/p1/submit",
        json={**STRONG_PAYLOAD, "pressure_performance_effect": "sometimes"},
    )
    assert response.status_code == 422


def test_missing_item_is_rejected_with_422(client):
    payload = {k: v for k, v in STRONG_PAYLOAD.items() if k != "post_error_calm"}
    assert client.post("/api/psychology/p1/submit", json=payload).status_code == 422


def test_rejected_submission_is_not_persisted(client):
    client.post(
        "/api/psychology/p1/submit", json={**STRONG_PAYLOAD, "nervousness": 99}
    )
    assert client.get("/api/psychology/p1/latest").status_code == 404


def test_unknown_match_id_is_rejected_rather_than_stored_dangling(client):
    response = client.post(
        "/api/psychology/p1/submit", json={**STRONG_PAYLOAD, "match_id": "no-such-match"}
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Submit -> latest round trip
# ---------------------------------------------------------------------------
def test_submit_returns_the_scored_contract(client):
    response = client.post("/api/psychology/p1/submit", json=STRONG_PAYLOAD)
    assert response.status_code == 201
    body = response.json()

    assert body["player_id"] == "p1"
    assert body["mental_readiness"] >= 70
    assert body["mental_performance_risk"] == "low"
    assert body["pressure_risk"] in ("low", "moderate", "high")
    assert body["method"] == "heuristic_proxy"
    assert body["confidence_level"] == "normal"
    assert body["schema_version"]
    assert body["data_source"] == "self_reported"
    assert body["submission_index"] == 1


def test_submitted_scores_are_integers_on_the_wire(client):
    body = client.post("/api/psychology/p1/submit", json=STRONG_PAYLOAD).json()
    for key in ("mental_readiness", "focus", "confidence", "stress"):
        assert isinstance(body[key], int)


def test_response_carries_a_non_clinical_disclaimer(client):
    body = client.post("/api/psychology/p1/submit", json=STRONG_PAYLOAD).json()
    disclaimer = body["disclaimer"].lower()
    assert "not emotion detection" in disclaimer
    assert "not a diagnosis" in disclaimer


def test_factors_and_phrase_lists_agree(client):
    body = client.post("/api/psychology/p1/submit", json=STRONG_PAYLOAD).json()
    positives = [d for d, label in body["factors"].items() if label == "positive"]
    negatives = [d for d, label in body["factors"].items() if label == "negative"]
    assert len(body["key_positive_factors"]) == len(positives)
    assert len(body["key_negative_factors"]) == len(negatives)


def test_submit_then_latest_returns_the_same_assessment(client):
    submitted = client.post("/api/psychology/p1/submit", json=STRONG_PAYLOAD).json()
    latest = client.get("/api/psychology/p1/latest").json()
    assert latest["assessment_id"] == submitted["assessment_id"]
    assert latest["mental_readiness"] == submitted["mental_readiness"]


def test_latest_404s_when_nothing_has_been_submitted(client):
    response = client.get("/api/psychology/nobody/latest")
    assert response.status_code == 404
    assert "nobody" in response.json()["detail"]


def test_latest_returns_the_most_recent_submission(client):
    client.post("/api/psychology/p1/submit", json=STRONG_PAYLOAD)
    second = client.post("/api/psychology/p1/submit", json=PRESSURED_PAYLOAD).json()

    latest = client.get("/api/psychology/p1/latest").json()
    assert latest["assessment_id"] == second["assessment_id"]
    assert latest["submission_index"] == 2


def test_latest_can_be_filtered_by_match(client):
    match_id = _make_match()
    client.post("/api/psychology/p1/submit", json=STRONG_PAYLOAD)
    linked = client.post(
        "/api/psychology/p1/submit", json={**PRESSURED_PAYLOAD, "match_id": match_id}
    ).json()

    filtered = client.get(f"/api/psychology/p1/latest?match_id={match_id}").json()
    assert filtered["assessment_id"] == linked["assessment_id"]
    assert filtered["match_id"] == match_id


def test_latest_404s_for_a_match_with_no_submission(client):
    match_id = _make_match()
    client.post("/api/psychology/p1/submit", json=STRONG_PAYLOAD)  # unlinked
    assert client.get(f"/api/psychology/p1/latest?match_id={match_id}").status_code == 404


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
def test_history_is_newest_first(client):
    client.post("/api/psychology/p1/submit", json=STRONG_PAYLOAD)
    client.post("/api/psychology/p1/submit", json=PRESSURED_PAYLOAD)
    client.post("/api/psychology/p1/submit", json=STRONG_PAYLOAD)

    history = client.get("/api/psychology/p1/history").json()
    assert [row["submission_index"] for row in history] == [3, 2, 1]


def test_history_is_empty_not_404_for_a_player_with_none(client):
    response = client.get("/api/psychology/nobody/history")
    assert response.status_code == 200
    assert response.json() == []


def test_history_respects_the_limit(client):
    for _ in range(3):
        client.post("/api/psychology/p1/submit", json=STRONG_PAYLOAD)
    assert len(client.get("/api/psychology/p1/history?limit=2").json()) == 2


def test_history_is_scoped_to_one_player(client):
    client.post("/api/psychology/p1/submit", json=STRONG_PAYLOAD)
    client.post("/api/psychology/p2/submit", json=STRONG_PAYLOAD)
    assert len(client.get("/api/psychology/p1/history").json()) == 1


def test_submission_index_is_per_player(client):
    client.post("/api/psychology/p1/submit", json=STRONG_PAYLOAD)
    body = client.post("/api/psychology/p2/submit", json=STRONG_PAYLOAD).json()
    assert body["submission_index"] == 1


# ---------------------------------------------------------------------------
# Single-assessment reads
# ---------------------------------------------------------------------------
def test_features_endpoint_serves_the_normalized_vector(client):
    submitted = client.post("/api/psychology/p1/submit", json=STRONG_PAYLOAD).json()
    response = client.get(
        f"/api/psychology/p1/{submitted['assessment_id']}/features"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["features"]["mental_clarity"] == 90.0
    # Direction ships with the vector -- a bare 0-100 number is ambiguous.
    assert body["feature_directions"]["stress_score"] == "higher_is_worse"
    assert body["feature_directions"]["focus_score"] == "higher_is_better"


def test_assessment_endpoint_returns_the_full_contract(client):
    submitted = client.post("/api/psychology/p1/submit", json=STRONG_PAYLOAD).json()
    body = client.get(f"/api/psychology/p1/{submitted['assessment_id']}").json()
    assert body == submitted


def test_latest_is_not_captured_as_an_assessment_id(client):
    """Route ordering guard: /{player_id}/latest is declared before
    /{player_id}/{assessment_id}, so "latest" must not bind as an id."""
    client.post("/api/psychology/p1/submit", json=STRONG_PAYLOAD)
    assert client.get("/api/psychology/p1/latest").status_code == 200
    assert client.get("/api/psychology/p1/history").status_code == 200


def test_another_players_assessment_is_not_readable(client):
    submitted = client.post("/api/psychology/p1/submit", json=STRONG_PAYLOAD).json()
    assert (
        client.get(f"/api/psychology/intruder/{submitted['assessment_id']}").status_code
        == 404
    )


def test_unknown_assessment_id_404s(client):
    client.post("/api/psychology/p1/submit", json=STRONG_PAYLOAD)
    assert client.get("/api/psychology/p1/does-not-exist").status_code == 404


# ---------------------------------------------------------------------------
# Historical integration (§5)
# ---------------------------------------------------------------------------
def test_history_is_only_consulted_when_cv_player_id_is_supplied(client):
    """Without an explicit cv_player_id there is no link into CV data at all --
    the two ID spaces are never joined by guesswork."""
    match_id = _make_match()
    _make_player_metric(match_id, 7, "press_resistance_score", 70.0, MetricConfidence.normal)

    body = client.post(
        "/api/psychology/p1/submit", json={**STRONG_PAYLOAD, "match_id": match_id}
    ).json()
    assert body["historical_context"] == []
    assert body["cv_player_id"] is None


def test_supplied_cv_player_id_folds_in_labelled_proxies(client):
    match_id = _make_match()
    _make_player_metric(match_id, 7, "press_resistance_score", 70.0, MetricConfidence.normal)
    _make_player_metric(match_id, 7, "decision_making_score", 64.0, MetricConfidence.normal)

    body = client.post(
        "/api/psychology/p1/submit",
        json={**STRONG_PAYLOAD, "match_id": match_id, "cv_player_id": 7},
    ).json()

    assert body["cv_player_id"] == 7
    assert body["historical_context"]
    # Every proxy is explicitly labelled as observed performance, never as a
    # measure of a mental state.
    assert all("proxy" in item for item in body["historical_context"])
    assert body["confidence_level"] == "normal"


def test_history_does_not_change_the_self_report_score(client):
    match_id = _make_match()
    _make_player_metric(match_id, 7, "press_resistance_score", 12.0, MetricConfidence.normal)
    _make_player_metric(match_id, 7, "decision_making_score", 15.0, MetricConfidence.normal)

    without = client.post("/api/psychology/p1/submit", json=STRONG_PAYLOAD).json()
    with_history = client.post(
        "/api/psychology/p2/submit",
        json={**STRONG_PAYLOAD, "match_id": match_id, "cv_player_id": 7},
    ).json()

    for key in ("mental_readiness", "focus", "confidence", "stress"):
        assert without[key] == with_history[key]


def test_unusable_history_degrades_confidence_but_still_scores(client):
    match_id = _make_match()
    _make_player_metric(
        match_id, 7, "press_resistance_score", None, MetricConfidence.low_upstream_confidence
    )
    _make_player_metric(
        match_id, 7, "decision_making_score", None, MetricConfidence.low_upstream_confidence
    )

    body = client.post(
        "/api/psychology/p1/submit",
        json={**STRONG_PAYLOAD, "match_id": match_id, "cv_player_id": 7},
    ).json()

    assert body["confidence_level"] == "low_upstream_confidence"
    assert body["mental_readiness"] >= 70  # the self-report score still stands


def test_missing_cv_rows_are_not_a_degraded_state(client):
    """A cv_player_id with no matching rows is "no history", not "bad history"."""
    match_id = _make_match()
    body = client.post(
        "/api/psychology/p1/submit",
        json={**STRONG_PAYLOAD, "match_id": match_id, "cv_player_id": 99},
    ).json()
    assert body["confidence_level"] == "normal"
    assert body["historical_context"] == []


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def test_raw_answers_and_features_are_both_persisted(client):
    """Storing the answers is what makes a score auditable and lets a formula
    change be re-run over historical submissions."""
    submitted = client.post("/api/psychology/p1/submit", json=STRONG_PAYLOAD).json()

    session = SessionLocal()
    try:
        row = session.get(PsychologyAssessment, submitted["assessment_id"])
        assert row.responses["concentration_level"] == 9
        assert row.responses["pressure_performance_effect"] == "improves"
        assert row.features["mental_clarity"] == 90.0
        assert row.method == MetricMethod.heuristic_proxy
        assert row.confidence_level == MetricConfidence.normal
        # The per-domain audit trail behind the headline numbers.
        assert row.sub_scores["component_metrics"]
    finally:
        session.close()
