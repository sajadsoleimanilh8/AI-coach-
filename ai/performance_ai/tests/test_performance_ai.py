"""
Unit tests for the Pre-Match Health Intelligence engine
(ai/performance_ai/match_readiness_predictor/).
"""

import os
import sys

import pytest

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from datetime import time

from ai.performance_ai.match_readiness_predictor.constants import (
    FATIGUE_WEIGHTS,
    NUTRITION_WEIGHTS,
    READINESS_WEIGHTS,
    RECOVERY_WEIGHTS,
    SESSION_LOAD_WEIGHTS,
    SLEEP_WEIGHTS,
)
from ai.performance_ai.match_readiness_predictor.features import (
    FEATURE_DIRECTIONS,
    extract_features,
)
from ai.performance_ai.match_readiness_predictor.questionnaire import (
    PreMatchQuestionnaireInput,
    QuestionnaireValidationError,
)
from ai.performance_ai.match_readiness_predictor.score import score_match_readiness
from ai.performance_ai.match_readiness_predictor.scorer import (
    HeuristicReadinessScorer,
    ReadinessScorer,
)




def healthy_questionnaire(**overrides) -> PreMatchQuestionnaireInput:
    """Well-slept, unfatigued, no recent load."""
    base = dict(
        player_id="p-healthy",
        sleep_duration_hours=8.5,
        sleep_quality=9,
        bedtime=time(22, 30),
        wake_time=time(7, 0),
        night_awakenings=0,
        trained_last_24h=False,
        trained_last_48h=False,
        training_duration_minutes=0.0,
        training_intensity=1,
        high_intensity_activity=False,
        hours_since_last_training=None,
        fatigue=2,
        muscle_soreness=2,
        pain_level=1,
        perceived_readiness=9,
        hydration_liters=3.0,
        nutrition_quality=9,
        hours_since_last_meal=2.5,
    )
    base.update(overrides)
    return PreMatchQuestionnaireInput(**base)


def fatigued_questionnaire(**overrides) -> PreMatchQuestionnaireInput:
    """Short sleep, hard high-intensity session yesterday, very sore."""
    base = dict(
        player_id="p-fatigued",
        sleep_duration_hours=4.5,
        sleep_quality=3,
        bedtime=time(1, 30),
        wake_time=time(6, 0),
        night_awakenings=3,
        trained_last_24h=True,
        trained_last_48h=True,
        training_duration_minutes=150.0,
        training_intensity=9,
        high_intensity_activity=True,
        hours_since_last_training=10.0,
        fatigue=9,
        muscle_soreness=9,
        pain_level=6,
        perceived_readiness=3,
        hydration_liters=1.0,
        nutrition_quality=4,
        hours_since_last_meal=7.0,
    )
    base.update(overrides)
    return PreMatchQuestionnaireInput(**base)


def mixed_questionnaire(**overrides) -> PreMatchQuestionnaireInput:
    """Good sleep, but a heavy recent workload and moderate soreness."""
    base = dict(
        player_id="p-mixed",
        sleep_duration_hours=8.0,
        sleep_quality=8,
        bedtime=time(23, 0),
        wake_time=time(7, 0),
        night_awakenings=1,
        trained_last_24h=True,
        trained_last_48h=True,
        training_duration_minutes=120.0,
        training_intensity=8,
        high_intensity_activity=True,
        hours_since_last_training=12.0,
        fatigue=5,
        muscle_soreness=5,
        pain_level=2,
        perceived_readiness=6,
        hydration_liters=2.5,
        nutrition_quality=7,
        hours_since_last_meal=3.0,
    )
    base.update(overrides)
    return PreMatchQuestionnaireInput(**base)




@pytest.mark.parametrize(
    "weights",
    [
        SLEEP_WEIGHTS,
        SESSION_LOAD_WEIGHTS,
        RECOVERY_WEIGHTS,
        FATIGUE_WEIGHTS,
        NUTRITION_WEIGHTS,
        READINESS_WEIGHTS,
    ],
)
def test_weight_sets_sum_to_one(weights):
    assert sum(weights.values()) == pytest.approx(1.0)




def test_features_are_bounded_0_100():
    for q in (healthy_questionnaire(), fatigued_questionnaire(), mixed_questionnaire()):
        vector = extract_features(q).as_dict()
        for name, value in vector.items():
            if value is None or name in {"time_in_bed_hours", "sleep_efficiency_pct"}:
                continue
            assert 0.0 <= value <= 100.0, f"{name}={value} out of range for {q.player_id}"


def test_every_scored_feature_declares_a_direction():
    """A feature with no documented direction is unreadable to a future ML
    consumer -- and to a reviewer."""
    informational = {"time_in_bed_hours", "sleep_efficiency_pct"}
    scored = set(extract_features(healthy_questionnaire()).as_dict()) - informational
    assert scored == set(FEATURE_DIRECTIONS)


def test_sleep_duration_normalizes_between_the_3h_floor_and_the_8h_target():
    def duration(hours):
        return extract_features(healthy_questionnaire(sleep_duration_hours=hours)).sleep_duration

    assert duration(3.0) == 0.0
    assert duration(2.0) == 0.0
    assert duration(5.5) == 50.0
    assert duration(8.0) == 100.0
    assert duration(11.0) == 100.0


def test_no_recent_training_means_zero_recent_load():
    """Duration/intensity answers describe a session that is no longer
    recent -- they must not leak into recent load."""
    f = extract_features(
        healthy_questionnaire(
            trained_last_24h=False,
            trained_last_48h=False,
            training_duration_minutes=180.0,
            training_intensity=10,
        )
    )
    assert f.recent_training_load == 0.0
    assert f.high_intensity_load == 0.0


def test_recent_load_falls_as_hours_since_training_grows():
    loads = [
        extract_features(mixed_questionnaire(hours_since_last_training=h)).recent_training_load
        for h in (0.0, 12.0, 24.0, 48.0, 72.0, 96.0)
    ]
    assert loads == sorted(loads, reverse=True)
    assert loads[0] > 0
    assert loads[-1] == 0.0


def test_recent_load_rises_with_duration_and_intensity():
    light = extract_features(mixed_questionnaire(training_duration_minutes=30.0, training_intensity=3))
    heavy = extract_features(mixed_questionnaire(training_duration_minutes=120.0, training_intensity=9))
    assert heavy.recent_training_load > light.recent_training_load


def test_time_in_bed_and_efficiency_are_informational_only():
    """Changing only bedtime/wake_time must not move any score -- a single
    questionnaire has no baseline to judge sleep timing against."""
    a = score_match_readiness(healthy_questionnaire(bedtime=time(22, 0), wake_time=time(7, 0)))
    b = score_match_readiness(healthy_questionnaire(bedtime=time(0, 30), wake_time=time(11, 0)))
    assert a.features.time_in_bed_hours != b.features.time_in_bed_hours
    assert a.physical_readiness == b.physical_readiness
    assert a.fatigue_score == b.fatigue_score
    assert a.recovery_score == b.recovery_score


def test_time_in_bed_wraps_past_midnight():
    f = extract_features(healthy_questionnaire(bedtime=time(23, 0), wake_time=time(7, 0)))
    assert f.time_in_bed_hours == 8.0


def test_sleep_efficiency_capped_at_100_when_self_report_exceeds_time_in_bed():
    f = extract_features(
        healthy_questionnaire(sleep_duration_hours=9.0, bedtime=time(23, 0), wake_time=time(7, 0))
    )
    assert f.sleep_efficiency_pct == 100.0




def test_notes_are_carried_through_but_never_scored():
    without = score_match_readiness(healthy_questionnaire())
    with_notes = score_match_readiness(
        healthy_questionnaire(caffeine_or_supplement_notes="200mg caffeine, creatine")
    )
    assert with_notes.notes == "200mg caffeine, creatine"
    assert without.notes is None
    assert with_notes.as_dict()["features"] == without.as_dict()["features"]
    assert with_notes.physical_readiness == without.physical_readiness
    assert with_notes.fatigue_score == without.fatigue_score
    assert with_notes.recovery_score == without.recovery_score
    assert with_notes.performance_risk == without.performance_risk
    assert with_notes.workload_risk == without.workload_risk




def test_healthy_scenario():
    result = score_match_readiness(healthy_questionnaire())

    assert result.recovery_score == pytest.approx(84.5)

    assert result.fatigue_score == pytest.approx(8.5, abs=0.1)

    assert result.physical_readiness == pytest.approx(90.5, abs=0.2)

    assert result.physical_readiness >= 75
    assert result.performance_risk == "low"
    assert result.workload_risk == "low"
    assert "good hydration" in result.key_positive_factors()
    assert result.key_negative_factors() == []




def test_fatigued_scenario():
    result = score_match_readiness(fatigued_questionnaire())

    assert result.fatigue_score >= 65
    assert result.physical_readiness < 45
    assert result.performance_risk in {"moderate", "high"}
    assert result.workload_risk == "high"

    negatives = result.key_negative_factors()
    assert "high recent training load" in negatives
    assert "high muscle soreness" in negatives
    assert "insufficient or poor sleep" in negatives


def test_fatigued_scores_worse_than_healthy_on_every_headline():
    healthy = score_match_readiness(healthy_questionnaire())
    fatigued = score_match_readiness(fatigued_questionnaire())
    assert fatigued.physical_readiness < healthy.physical_readiness
    assert fatigued.recovery_score < healthy.recovery_score
    assert fatigued.fatigue_score > healthy.fatigue_score




def test_mixed_scenario_sits_strictly_between_healthy_and_fatigued():
    healthy = score_match_readiness(healthy_questionnaire())
    mixed = score_match_readiness(mixed_questionnaire())
    fatigued = score_match_readiness(fatigued_questionnaire())

    assert fatigued.physical_readiness < mixed.physical_readiness < healthy.physical_readiness
    assert healthy.fatigue_score < mixed.fatigue_score < fatigued.fatigue_score


def test_mixed_scenario_reports_both_a_positive_and_a_negative_factor():
    """The point of the mixed case: good sleep alongside a heavy workload
    must surface as both, not be averaged into a single bland verdict."""
    mixed = score_match_readiness(mixed_questionnaire())
    assert "good sleep quality and duration" in mixed.key_positive_factors()
    assert "high recent training load" in mixed.key_negative_factors()




def test_performance_risk_bands_match_the_readiness_thresholds():
    for q in (healthy_questionnaire(), mixed_questionnaire(), fatigued_questionnaire()):
        result = score_match_readiness(q)
        if result.physical_readiness >= 70:
            assert result.performance_risk == "low"
        elif result.physical_readiness >= 45:
            assert result.performance_risk == "moderate"
        else:
            assert result.performance_risk == "high"


def test_workload_risk_high_on_hard_high_intensity_session_yesterday():
    """The second limb of the rule: even when the decayed load number itself
    is not yet >= 70, a hard high-intensity session in the last 24h is."""
    result = score_match_readiness(
        healthy_questionnaire(
            trained_last_24h=True,
            high_intensity_activity=True,
            training_intensity=8,
            training_duration_minutes=20.0,
            hours_since_last_training=20.0,
        )
    )
    assert result.features.recent_training_load < 70
    assert result.workload_risk == "high"


def test_workload_risk_moderate_between_the_two_bands():
    """A real session yesterday, but neither heavy enough nor
    high-intensity enough to trip either limb of the "high" rule."""
    result = score_match_readiness(
        mixed_questionnaire(
            training_intensity=7,
            training_duration_minutes=90.0,
            high_intensity_activity=False,
            hours_since_last_training=20.0,
        )
    )
    assert 40 <= result.features.recent_training_load < 70
    assert result.workload_risk == "moderate"


def test_workload_risk_low_on_a_light_session_with_no_high_intensity_work():
    result = score_match_readiness(
        mixed_questionnaire(
            training_intensity=5,
            training_duration_minutes=60.0,
            high_intensity_activity=False,
            hours_since_last_training=20.0,
        )
    )
    assert result.features.recent_training_load < 40
    assert result.workload_risk == "low"




def test_same_input_twice_gives_identical_output():
    for factory in (healthy_questionnaire, mixed_questionnaire, fatigued_questionnaire):
        first = score_match_readiness(factory())
        second = score_match_readiness(factory())
        assert first.as_dict() == second.as_dict()


def test_two_players_with_identical_answers_get_identical_scores():
    """No per-player state, no lookup table: player_id cannot move a number."""
    a = score_match_readiness(healthy_questionnaire(player_id="player-a"))
    b = score_match_readiness(healthy_questionnaire(player_id="player-b"))
    assert a.as_dict() == b.as_dict()


def test_heuristic_scorer_implements_the_interface_and_declares_its_method():
    scorer = HeuristicReadinessScorer()
    assert isinstance(scorer, ReadinessScorer)
    assessment = scorer.score(extract_features(healthy_questionnaire()))
    assert assessment.method == "heuristic_proxy"
    assert assessment.schema_version == "v1"
    assert assessment.data_source == "self_reported"


def test_readiness_scorer_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        ReadinessScorer()


def test_feature_vector_contains_every_name_the_spec_requires():
    vector = score_match_readiness(healthy_questionnaire()).to_feature_vector()
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
        assert name in vector, f"{name} missing from the stored feature vector"


def test_every_dimension_is_classified():
    result = score_match_readiness(mixed_questionnaire())
    dimensions = {f.dimension for f in result.factors}
    assert dimensions == {
        "sleep",
        "recent_training_load",
        "fatigue",
        "muscle_soreness",
        "pain",
        "hydration",
        "nutrition",
        "perceived_readiness",
        "recovery",
    }
    for factor in result.factors:
        assert factor.label in {"positive", "negative", "neutral"}
        assert 0.0 <= factor.score <= 100.0




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
@pytest.mark.parametrize("bad_value", [0, 11, -1, 100])
def test_scale_fields_reject_out_of_range(field_name, bad_value):
    with pytest.raises(QuestionnaireValidationError):
        healthy_questionnaire(**{field_name: bad_value})


@pytest.mark.parametrize("field_name", SCALE_FIELDS)
@pytest.mark.parametrize("good_value", [1, 5, 10])
def test_scale_fields_accept_the_full_valid_range(field_name, good_value):
    assert healthy_questionnaire(**{field_name: good_value}) is not None


@pytest.mark.parametrize("bad_hours", [0.0, -1.0, 24.5, 30.0])
def test_sleep_duration_rejects_impossible_values(bad_hours):
    with pytest.raises(QuestionnaireValidationError):
        healthy_questionnaire(sleep_duration_hours=bad_hours)


def test_sleep_duration_accepts_the_boundary():
    assert healthy_questionnaire(sleep_duration_hours=24.0) is not None


def test_negative_training_duration_rejected():
    with pytest.raises(QuestionnaireValidationError):
        healthy_questionnaire(training_duration_minutes=-1.0)


def test_negative_night_awakenings_rejected():
    with pytest.raises(QuestionnaireValidationError):
        healthy_questionnaire(night_awakenings=-1)


def test_negative_hydration_rejected():
    with pytest.raises(QuestionnaireValidationError):
        healthy_questionnaire(hydration_liters=-0.5)


def test_negative_hours_since_last_meal_rejected():
    with pytest.raises(QuestionnaireValidationError):
        healthy_questionnaire(hours_since_last_meal=-2.0)


def test_negative_hours_since_last_training_rejected():
    with pytest.raises(QuestionnaireValidationError):
        healthy_questionnaire(hours_since_last_training=-3.0)


def test_hours_since_last_training_may_be_none():
    assert healthy_questionnaire(hours_since_last_training=None) is not None


def test_empty_player_id_rejected():
    with pytest.raises(QuestionnaireValidationError):
        healthy_questionnaire(player_id="   ")
