"""
API tests for the Pre-Match Health Intelligence router
(backend/api/prematch_health.py).

Runs the real FastAPI app against a throwaway SQLite file: DATABASE_URL is
pointed at a temp path BEFORE backend.database.session is first imported, so
the app's own startup Base.metadata.create_all() builds the schema there and
the developer's sports_strategy.db is never touched. That also means these
tests exercise the real table definitions, not a hand-built test schema --
if a column or FK is wrong in models.py, it fails here.
"""

import os
import tempfile

import pytest

# Must happen before any backend.database import binds the engine.
_TEST_DB_DIR = tempfile.mkdtemp(prefix="prematch_health_tests_")
_TEST_DB_PATH = os.path.join(_TEST_DB_DIR, "test_prematch_health.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB_PATH}"

from fastapi.testclient import TestClient  # noqa: E402

from backend.api.main import app  # noqa: E402
from backend.database.models import (  # noqa: E402
    Match,
    MetricMethod,
    PreMatchHealthAssessment,
    PreMatchQuestionnaire,
    RiskLevel,
)
from backend.database.session import Base, SessionLocal, engine  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:  # triggers the startup create_all
        yield test_client


@pytest.fixture(autouse=True)
def clean_tables():
    """Each test starts from an empty pre-match schema so history/latest
    assertions cannot be polluted by an earlier test's submissions. Only this
    module's tables are cleared -- nothing else in the schema is touched."""
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        session.query(PreMatchHealthAssessment).delete()
        session.query(PreMatchQuestionnaire).delete()
        session.query(Match).delete()
        session.commit()
    finally:
        session.close()
    yield


HEALTHY_PAYLOAD = {
    "sleep_duration_hours": 8.5,
    "sleep_quality": 9,
    "bedtime": "22:30",
    "wake_time": "07:00",
    "night_awakenings": 0,
    "trained_last_24h": False,
    "trained_last_48h": False,
    "training_duration_minutes": 0.0,
    "training_intensity": 1,
    "high_intensity_activity": False,
    "hours_since_last_training": None,
    "fatigue": 2,
    "muscle_soreness": 2,
    "pain_level": 1,
    "perceived_readiness": 9,
    "hydration_liters": 3.0,
    "nutrition_quality": 9,
    "hours_since_last_meal": 2.5,
}

FATIGUED_PAYLOAD = {
    **HEALTHY_PAYLOAD,
    "sleep_duration_hours": 4.5,
    "sleep_quality": 3,
    "night_awakenings": 3,
    "trained_last_24h": True,
    "trained_last_48h": True,
    "training_duration_minutes": 150.0,
    "training_intensity": 9,
    "high_intensity_activity": True,
    "hours_since_last_training": 10.0,
    "fatigue": 9,
    "muscle_soreness": 9,
    "pain_level": 6,
    "perceived_readiness": 3,
    "hydration_liters": 1.0,
    "nutrition_quality": 4,
    "hours_since_last_meal": 7.0,
}


def submit(client, player_id: str, payload: dict | None = None):
    return client.post(
        f"/api/prematch_health/{player_id}/submit", json=payload or HEALTHY_PAYLOAD
    )


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
def test_submit_returns_201_and_a_full_assessment(client):
    response = submit(client, "p123")
    assert response.status_code == 201, response.text

    body = response.json()
    assert body["player_id"] == "p123"
    assert body["match_id"] is None
    assert body["method"] == "heuristic_proxy"
    assert body["schema_version"] == "v1"
    assert body["data_source"] == "self_reported"
    assert body["performance_risk"] == "low"
    assert body["physical_readiness"] >= 75
    assert 0 <= body["fatigue_score"] <= 100
    assert 0 <= body["recovery_score"] <= 100
    assert "good hydration" in body["key_positive_factors"]
    assert body["key_negative_factors"] == []
    assert body["assessment_id"] and body["questionnaire_id"]


def test_response_carries_a_non_medical_disclaimer(client):
    body = submit(client, "p-disclaimer").json()
    assert "not a medical assessment" in body["disclaimer"].lower()


def test_submit_persists_both_rows_matching_what_was_submitted(client):
    body = submit(client, "p-persist", FATIGUED_PAYLOAD).json()

    session = SessionLocal()
    try:
        questionnaire = session.get(PreMatchQuestionnaire, body["questionnaire_id"])
        assessment = session.get(PreMatchHealthAssessment, body["assessment_id"])

        # The raw answers are stored exactly as submitted.
        assert questionnaire is not None
        assert questionnaire.player_id == "p-persist"
        answers = questionnaire.questionnaire_json
        assert answers["sleep_duration_hours"] == 4.5
        assert answers["fatigue"] == 9
        assert answers["high_intensity_activity"] is True
        assert answers["bedtime"] == "22:30"

        # The computed row matches the response exactly.
        assert assessment is not None
        assert assessment.questionnaire_id == questionnaire.id
        assert assessment.physical_readiness == body["physical_readiness"]
        assert assessment.fatigue_score == body["fatigue_score"]
        assert assessment.recovery_score == body["recovery_score"]
        assert assessment.performance_risk == RiskLevel(body["performance_risk"])
        assert assessment.workload_risk == RiskLevel(body["workload_risk"])
        assert assessment.method == MetricMethod.heuristic_proxy
        assert assessment.schema_version == "v1"

        # The 1:1 relationship resolves in both directions.
        assert questionnaire.assessment.id == assessment.id
        assert assessment.questionnaire.id == questionnaire.id

        # The explainability record survived the round trip.
        assert {f["dimension"] for f in assessment.factors} >= {
            "sleep",
            "recent_training_load",
            "fatigue",
            "recovery",
        }
    finally:
        session.close()


def test_deleting_a_questionnaire_cascades_to_its_assessment(client):
    body = submit(client, "p-cascade").json()
    session = SessionLocal()
    try:
        questionnaire = session.get(PreMatchQuestionnaire, body["questionnaire_id"])
        session.delete(questionnaire)
        session.commit()
        assert session.get(PreMatchHealthAssessment, body["assessment_id"]) is None
    finally:
        session.close()


def test_notes_are_echoed_back_but_do_not_change_any_score(client):
    plain = submit(client, "p-notes-a").json()
    noted = submit(
        client,
        "p-notes-b",
        {**HEALTHY_PAYLOAD, "caffeine_or_supplement_notes": "200mg caffeine"},
    ).json()

    assert noted["notes"] == "200mg caffeine"
    assert plain["notes"] is None
    for field in (
        "physical_readiness",
        "fatigue_score",
        "recovery_score",
        "performance_risk",
        "workload_risk",
    ):
        assert noted[field] == plain[field]


def test_identical_submissions_produce_identical_scores(client):
    first = submit(client, "p-determinism").json()
    second = submit(client, "p-determinism").json()
    for field in (
        "physical_readiness",
        "fatigue_score",
        "recovery_score",
        "performance_risk",
        "workload_risk",
        "key_positive_factors",
        "key_negative_factors",
    ):
        assert first[field] == second[field]


# ---------------------------------------------------------------------------
# match_id linkage
# ---------------------------------------------------------------------------
def test_submit_links_to_an_existing_match(client):
    session = SessionLocal()
    try:
        match = Match(home_team="Home", away_team="Away", video_path="/tmp/x.mp4", duration=0.0)
        session.add(match)
        session.commit()
        match_id = match.match_id
    finally:
        session.close()

    body = submit(client, "p-match", {**HEALTHY_PAYLOAD, "match_id": match_id}).json()
    assert body["match_id"] == match_id

    session = SessionLocal()
    try:
        assessment = session.get(PreMatchHealthAssessment, body["assessment_id"])
        assert assessment.match_id == match_id
        assert assessment.questionnaire.match_id == match_id
    finally:
        session.close()


def test_submit_rejects_an_unknown_match_id(client):
    """A dangling match reference must not be stored -- SQLite does not
    enforce the FK, so the router checks it."""
    response = submit(
        client, "p-bad-match", {**HEALTHY_PAYLOAD, "match_id": "does-not-exist"}
    )
    assert response.status_code == 404
    assert "Match not found" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Validation -- 422, never a silent clamp
# ---------------------------------------------------------------------------
SCALE_FIELDS = [
    "sleep_quality",
    "training_intensity",
    "fatigue",
    "muscle_soreness",
    "pain_level",
    "perceived_readiness",
    "nutrition_quality",
]


@pytest.mark.parametrize("field_name", SCALE_FIELDS)
@pytest.mark.parametrize("bad_value", [0, 11])
def test_scale_fields_reject_out_of_range_with_422(client, field_name, bad_value):
    response = submit(client, "p-invalid", {**HEALTHY_PAYLOAD, field_name: bad_value})
    assert response.status_code == 422
    assert any(field_name in str(err["loc"]) for err in response.json()["detail"])


@pytest.mark.parametrize("bad_hours", [0, -1, 24.5, 30])
def test_sleep_duration_rejects_impossible_values_with_422(client, bad_hours):
    response = submit(client, "p-invalid", {**HEALTHY_PAYLOAD, "sleep_duration_hours": bad_hours})
    assert response.status_code == 422


def test_negative_training_duration_rejected_with_422(client):
    response = submit(
        client, "p-invalid", {**HEALTHY_PAYLOAD, "training_duration_minutes": -1}
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "field_name", ["night_awakenings", "hydration_liters", "hours_since_last_meal"]
)
def test_negative_values_rejected_with_422(client, field_name):
    response = submit(client, "p-invalid", {**HEALTHY_PAYLOAD, field_name: -1})
    assert response.status_code == 422


def test_out_of_range_submission_persists_nothing(client):
    """A rejected questionnaire must leave no partial row behind."""
    assert submit(client, "p-nothing", {**HEALTHY_PAYLOAD, "fatigue": 99}).status_code == 422
    session = SessionLocal()
    try:
        assert session.query(PreMatchQuestionnaire).count() == 0
        assert session.query(PreMatchHealthAssessment).count() == 0
    finally:
        session.close()


def test_missing_required_field_rejected_with_422(client):
    payload = {k: v for k, v in HEALTHY_PAYLOAD.items() if k != "fatigue"}
    assert submit(client, "p-invalid", payload).status_code == 422


def test_boundary_values_are_accepted(client):
    payload = {
        **HEALTHY_PAYLOAD,
        "sleep_duration_hours": 24.0,
        "sleep_quality": 1,
        "fatigue": 10,
        "muscle_soreness": 1,
        "pain_level": 10,
        "perceived_readiness": 1,
        "nutrition_quality": 10,
        "training_duration_minutes": 0.0,
        "night_awakenings": 0,
    }
    assert submit(client, "p-boundary", payload).status_code == 201


# ---------------------------------------------------------------------------
# latest / history / features / assessment
# ---------------------------------------------------------------------------
def test_latest_404s_before_any_submission(client):
    response = client.get("/api/prematch_health/nobody/latest")
    assert response.status_code == 404
    assert "No pre-match assessment" in response.json()["detail"]


def test_submit_then_latest_round_trip(client):
    submitted = submit(client, "p-round").json()
    latest = client.get("/api/prematch_health/p-round/latest").json()
    assert latest["assessment_id"] == submitted["assessment_id"]
    assert latest["physical_readiness"] == submitted["physical_readiness"]
    assert latest["key_negative_factors"] == submitted["key_negative_factors"]


def test_latest_returns_the_most_recent_of_several(client):
    submit(client, "p-multi", HEALTHY_PAYLOAD)
    last = submit(client, "p-multi", FATIGUED_PAYLOAD).json()
    latest = client.get("/api/prematch_health/p-multi/latest").json()
    assert latest["assessment_id"] == last["assessment_id"]
    assert latest["performance_risk"] == last["performance_risk"]


def test_history_returns_every_submission_newest_first(client):
    submit(client, "p-hist", HEALTHY_PAYLOAD)
    submit(client, "p-hist", FATIGUED_PAYLOAD)
    third = submit(client, "p-hist", HEALTHY_PAYLOAD).json()

    history = client.get("/api/prematch_health/p-hist/history").json()
    assert len(history) == 3
    assert history[0]["assessment_id"] == third["assessment_id"]
    # Ordered by the explicit submission counter, not by computed_at -- the
    # OS clock is too coarse to separate submissions made this close together
    # (all three of these can share one timestamp).
    assert [h["submission_index"] for h in history] == [3, 2, 1]


def test_submission_index_increments_per_player_independently(client):
    """The ordering guarantee /latest depends on: submissions made inside a
    single clock tick still order correctly, and one player's counter is not
    affected by another's."""
    for expected in (1, 2, 3):
        assert submit(client, "p-seq-a").json()["submission_index"] == expected
    assert submit(client, "p-seq-b").json()["submission_index"] == 1

    latest = client.get("/api/prematch_health/p-seq-a/latest").json()
    assert latest["submission_index"] == 3


def test_history_respects_limit(client):
    for _ in range(4):
        submit(client, "p-limit")
    assert len(client.get("/api/prematch_health/p-limit/history?limit=2").json()) == 2


def test_history_rejects_an_invalid_limit(client):
    assert client.get("/api/prematch_health/p-limit/history?limit=0").status_code == 422


def test_history_is_empty_for_an_unknown_player(client):
    response = client.get("/api/prematch_health/nobody/history")
    assert response.status_code == 200
    assert response.json() == []


def test_history_is_scoped_to_one_player(client):
    submit(client, "p-one")
    submit(client, "p-two")
    assert len(client.get("/api/prematch_health/p-one/history").json()) == 1


def test_features_endpoint_serves_the_normalized_vector(client):
    submitted = submit(client, "p-features").json()
    response = client.get(
        f"/api/prematch_health/p-features/{submitted['assessment_id']}/features"
    )
    assert response.status_code == 200

    body = response.json()
    features = body["features"]
    for name in (
        "sleep_duration",
        "sleep_quality",
        "training_duration",
        "training_intensity",
        "recent_training_load",
        "high_intensity_load",
        "fatigue_score",
        "muscle_soreness",
        "pain_level",
        "hydration_score",
        "nutrition_score",
        "recovery_score",
        "physical_readiness",
    ):
        assert name in features

    # Provenance is not a model input.
    assert "data_source" not in features
    # Every scored feature declares which way it points.
    assert body["feature_directions"]["hydration_score"] == "higher_is_better"
    assert body["feature_directions"]["muscle_soreness"] == "higher_is_worse"
    assert body["method"] == "heuristic_proxy"
    assert body["schema_version"] == "v1"


def test_assessment_endpoint_returns_the_full_structured_output(client):
    submitted = submit(client, "p-assessment", FATIGUED_PAYLOAD).json()
    body = client.get(
        f"/api/prematch_health/p-assessment/{submitted['assessment_id']}/assessment"
    ).json()

    # The Section 7 contract, in full.
    for field in (
        "player_id",
        "match_id",
        "physical_readiness",
        "fatigue_score",
        "recovery_score",
        "performance_risk",
        "workload_risk",
        "key_positive_factors",
        "key_negative_factors",
        "method",
        "schema_version",
        "computed_at",
    ):
        assert field in body

    assert body == submitted
    assert body["key_negative_factors"]
    assert all(
        f["label"] in {"positive", "negative", "neutral"} for f in body["factors"]
    )


def test_assessment_404s_for_an_unknown_id(client):
    submit(client, "p-404")
    assert (
        client.get("/api/prematch_health/p-404/no-such-id/assessment").status_code == 404
    )


def test_assessment_is_not_readable_under_another_players_path(client):
    """An assessment id must not be readable by a player it does not belong
    to, even though the id alone would be enough to find the row."""
    owned = submit(client, "p-owner").json()
    submit(client, "p-other")
    response = client.get(
        f"/api/prematch_health/p-other/{owned['assessment_id']}/assessment"
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Nothing existing regressed
# ---------------------------------------------------------------------------
def test_existing_health_endpoint_still_works(client):
    assert client.get("/health").json()["status"] == "ok"
