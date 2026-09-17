"""
Unit tests for the Pre-Match Psychology Intelligence scoring engine.

Mirrors ai/player_intelligence/tests/test_player_intelligence.py: the same
one-assertion-per-behaviour style, no snapshots of
arbitrary numbers. Where a number IS asserted exactly, it is one the formula
pins down (a boundary, a documented floor, a stated worked example); everything
about the overall shape of the scoring is asserted comparatively, so a
reweighting that preserves the intended ordering does not fail the suite for
no reason.
"""


import pytest

from ai.psychology_ai.confidence_score.score import score_confidence
from ai.psychology_ai.constants import (
    CONFIDENCE_WEIGHTS,
    FOCUS_WEIGHTS,
    MOTIVATION_WEIGHTS,
    PRESSURE_CONTEXT_WEIGHT,
    PRESSURE_EFFECT_WEIGHT,
    READINESS_WEIGHTS,
    STRESS_WEIGHTS,
)
from ai.psychology_ai.feature_extraction import (
    FEATURE_DIRECTIONS,
    FEATURE_NAMES,
    QuestionnaireValidationError,
    extract_features,
)
from ai.psychology_ai.focus_score.score import score_focus
from ai.psychology_ai.mental_readiness.score import (
    assess_psychology,
    score_component_metrics,
    score_mental_readiness,
)
from ai.psychology_ai.model_interface import (
    HeuristicReadinessModel,
    PsychologyReadinessModel,
)
from ai.psychology_ai.motivation_score.score import score_motivation
from ai.psychology_ai.pressure_index.score import score_pressure_index
from ai.psychology_ai.stress_analysis.score import score_stress

# ---------------------------------------------------------------------------
# Fixtures: three questionnaires spanning the intended range.
# ---------------------------------------------------------------------------
STRONG = dict(
    concentration_level=9,
    focus_maintenance=9,
    mental_clarity_raw=9,
    pre_match_stress=2,
    importance_pressure=3,
    nervousness=2,
    performance_confidence=9,
    tactical_confidence_raw=8,
    match_motivation=9,
    competitive_motivation_raw=9,
    mistake_recovery_speed=8,
    pressure_performance_effect="improves",
    post_error_calm=8,
)

HIGH_PRESSURE = dict(
    concentration_level=4,
    focus_maintenance=4,
    mental_clarity_raw=4,
    pre_match_stress=9,
    importance_pressure=9,
    nervousness=9,
    performance_confidence=3,
    tactical_confidence_raw=3,
    match_motivation=6,
    competitive_motivation_raw=6,
    mistake_recovery_speed=3,
    pressure_performance_effect="reduces",
    post_error_calm=3,
)

MIXED = dict(
    concentration_level=6,
    focus_maintenance=6,
    mental_clarity_raw=6,
    pre_match_stress=8,
    importance_pressure=8,
    nervousness=7,
    performance_confidence=6,
    tactical_confidence_raw=6,
    match_motivation=9,
    competitive_motivation_raw=9,
    mistake_recovery_speed=5,
    pressure_performance_effect="no_change",
    post_error_calm=5,
)

ALL_MIN = {**{item: 1 for item in STRONG if item != "pressure_performance_effect"},
           "pressure_performance_effect": "reduces"}
ALL_MAX = {**{item: 10 for item in STRONG if item != "pressure_performance_effect"},
           "pressure_performance_effect": "improves"}


# ---------------------------------------------------------------------------
# Feature extraction: scaling
# ---------------------------------------------------------------------------
def test_every_declared_feature_is_produced():
    features = extract_features(STRONG)
    assert set(features) == set(FEATURE_NAMES)
    assert set(features) == set(FEATURE_DIRECTIONS)


def test_scale_floor_is_ten_not_zero():
    """A rating of 1 maps to 10, never 0 -- the questionnaire's floor is 1, and
    treating "1" as "zero" would overstate what was reported."""
    features = extract_features(ALL_MIN)
    assert features["mental_clarity"] == 10.0
    assert features["confidence_score"] == 10.0
    assert features["focus_score"] == 10.0


def test_scale_ceiling_is_one_hundred():
    features = extract_features(ALL_MAX)
    assert features["mental_clarity"] == 100.0
    assert features["confidence_score"] == 100.0
    assert features["focus_score"] == 100.0


def test_midpoint_interpolates_linearly():
    responses = {**STRONG, "mental_clarity_raw": 5}
    assert extract_features(responses)["mental_clarity"] == 50.0


def test_averaged_items_are_meaned_then_scaled():
    """focus_score = mean(concentration, maintenance) * 10."""
    responses = {**STRONG, "concentration_level": 4, "focus_maintenance": 8}
    assert extract_features(responses)["focus_score"] == 60.0


def test_error_recovery_is_meaned_then_scaled():
    responses = {**STRONG, "mistake_recovery_speed": 3, "post_error_calm": 7}
    assert extract_features(responses)["error_recovery"] == 50.0


def test_stress_score_is_meaned_then_scaled():
    responses = {**STRONG, "pre_match_stress": 4, "importance_pressure": 6}
    assert extract_features(responses)["stress_score"] == 50.0


# ---------------------------------------------------------------------------
# Feature extraction: pressure_sensitivity, one case per categorical answer
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "effect,importance,expected",
    [
        # base * 0.65 + (importance * 10) * 0.35, per the documented formula.
        ("improves", 1, 20.0 * PRESSURE_EFFECT_WEIGHT + 10.0 * PRESSURE_CONTEXT_WEIGHT),
        ("no_change", 5, 50.0 * PRESSURE_EFFECT_WEIGHT + 50.0 * PRESSURE_CONTEXT_WEIGHT),
        ("reduces", 10, 80.0 * PRESSURE_EFFECT_WEIGHT + 100.0 * PRESSURE_CONTEXT_WEIGHT),
    ],
)
def test_pressure_sensitivity_matches_the_documented_formula(effect, importance, expected):
    responses = {
        **STRONG,
        "pressure_performance_effect": effect,
        "importance_pressure": importance,
    }
    assert extract_features(responses)["pressure_sensitivity"] == pytest.approx(expected)


def test_pressure_sensitivity_orders_the_three_answers():
    """improves < no_change < reduces, holding match importance constant."""
    sensitivities = [
        extract_features({**STRONG, "pressure_performance_effect": effect})[
            "pressure_sensitivity"
        ]
        for effect in ("improves", "no_change", "reduces")
    ]
    assert sensitivities[0] < sensitivities[1] < sensitivities[2]


def test_pressure_sensitivity_rises_with_match_importance():
    low = extract_features({**STRONG, "importance_pressure": 1})["pressure_sensitivity"]
    high = extract_features({**STRONG, "importance_pressure": 10})["pressure_sensitivity"]
    assert high > low


# ---------------------------------------------------------------------------
# Feature extraction: validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_value", [0, 11, -1])
def test_out_of_range_rating_is_rejected_not_clamped(bad_value):
    with pytest.raises(QuestionnaireValidationError):
        extract_features({**STRONG, "concentration_level": bad_value})


def test_non_integer_rating_is_rejected():
    with pytest.raises(QuestionnaireValidationError):
        extract_features({**STRONG, "nervousness": "7"})


def test_boolean_is_not_accepted_as_a_rating():
    """bool subclasses int in Python, so True would otherwise pass as a 1."""
    with pytest.raises(QuestionnaireValidationError):
        extract_features({**STRONG, "nervousness": True})


def test_unknown_pressure_effect_is_rejected():
    with pytest.raises(QuestionnaireValidationError):
        extract_features({**STRONG, "pressure_performance_effect": "maybe"})


def test_missing_answer_is_rejected():
    incomplete = {k: v for k, v in STRONG.items() if k != "post_error_calm"}
    with pytest.raises(QuestionnaireValidationError):
        extract_features(incomplete)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "weights",
    [FOCUS_WEIGHTS, CONFIDENCE_WEIGHTS, STRESS_WEIGHTS, MOTIVATION_WEIGHTS, READINESS_WEIGHTS],
)
def test_composite_weights_sum_to_one(weights):
    assert sum(weights.values()) == pytest.approx(1.0)


def test_pressure_sensitivity_weights_sum_to_one():
    assert pytest.approx(1.0) == PRESSURE_EFFECT_WEIGHT + PRESSURE_CONTEXT_WEIGHT

# ---------------------------------------------------------------------------
# Per-domain scorers: normal path and both gating paths
# ---------------------------------------------------------------------------

SCORERS = [
    (score_focus, "focus_score"),
    (score_confidence, "confidence_score"),
    (score_stress, "stress_score"),
    (score_motivation, "motivation_score"),
    (score_pressure_index, "pressure_index"),
    (score_mental_readiness, "mental_readiness"),
]


@pytest.mark.parametrize("scorer,metric_name", SCORERS)
def test_scorer_returns_a_player_metric_shaped_dict(scorer, metric_name):
    result = scorer(extract_features(STRONG))
    assert result["metric_name"] == metric_name
    assert result["method"] == "heuristic_proxy"
    assert result["confidence"] == "normal"
    assert 0.0 <= result["value"] <= 100.0
    assert result["sample_size"] > 0
    assert isinstance(result["sub_scores"], dict)
    assert result["schema_version"]


@pytest.mark.parametrize("scorer,metric_name", SCORERS)
def test_scorer_gates_to_low_sample_when_inputs_are_missing(scorer, metric_name):
    """value=None means genuinely could not be computed -- never a defaulted 0."""
    result = scorer({})
    assert result["value"] is None
    assert result["confidence"] == "low_sample"
    assert result["sample_size"] == 0


# Only the scorers that actually consult a CV proxy can reach the upstream
# path; the ones with no honest observable analogue (stress, motivation)
# declare no proxies and are excluded on purpose.
UPSTREAM_SCORERS = [
    (score_focus, {"focus_proxy": {"value": None, "confidence": "low_upstream_confidence"}}),
    (
        score_confidence,
        {"confidence_proxy": {"value": None, "confidence": "low_upstream_confidence"}},
    ),
    (
        score_pressure_index,
        {"pressure_response_proxy": {"value": 40.0, "confidence": "low_upstream_confidence"}},
    ),
]


@pytest.mark.parametrize("scorer,historical", UPSTREAM_SCORERS)
def test_unusable_history_flags_confidence_but_keeps_the_value(scorer, historical):
    """History corroborates; it never gates. The self-report score still
    stands, and only the confidence label degrades."""
    result = scorer(extract_features(STRONG), historical)
    assert result["value"] is not None
    assert result["confidence"] == "low_upstream_confidence"


def test_usable_history_leaves_confidence_normal():
    result = score_focus(
        extract_features(STRONG),
        {"focus_proxy": {"value": 71.0, "confidence": "normal"}},
    )
    assert result["confidence"] == "normal"


def test_absent_history_is_not_a_degraded_state():
    """The common case -- no CV data at all -- must not look like a problem."""
    assert score_focus(extract_features(STRONG), None)["confidence"] == "normal"
    assert score_focus(extract_features(STRONG), {})["confidence"] == "normal"


def test_stress_scorer_is_higher_is_worse():
    calm = score_stress(extract_features(STRONG))["value"]
    stressed = score_stress(extract_features(HIGH_PRESSURE))["value"]
    assert stressed > calm
    assert score_stress(extract_features(STRONG))["sub_scores"]["direction"] == "higher_is_worse"


def test_pressure_index_bands_its_own_value():
    low = score_pressure_index(extract_features(STRONG))
    high = score_pressure_index(extract_features(HIGH_PRESSURE))
    assert low["sub_scores"]["pressure_risk"] == "low"
    assert high["sub_scores"]["pressure_risk"] == "high"


def test_pressure_index_does_not_band_a_missing_value():
    assert score_pressure_index({})["sub_scores"]["pressure_risk"] is None


def test_component_metrics_covers_every_domain():
    metrics = score_component_metrics(extract_features(STRONG))
    assert [m["metric_name"] for m in metrics] == [name for _, name in SCORERS]


# ---------------------------------------------------------------------------
# The three headline scenarios
# ---------------------------------------------------------------------------
def test_strong_mental_state():
    """High focus + high confidence + low stress -> high readiness, low risk."""
    result = assess_psychology(STRONG)
    assert result["mental_readiness"] >= 70
    assert result["mental_performance_risk"] == "low"
    assert result["factors"]["focus"] == "positive"
    assert result["factors"]["confidence"] == "positive"
    # A "positive" stress factor means LOW reported stress.
    assert result["factors"]["stress"] == "positive"


def test_high_pressure_state():
    """Low confidence + high stress + high pressure sensitivity -> lower
    readiness than the strong case, and elevated pressure risk."""
    strong = assess_psychology(STRONG)
    pressured = assess_psychology(HIGH_PRESSURE)
    assert pressured["mental_readiness"] < strong["mental_readiness"]
    assert pressured["pressure_risk"] in ("moderate", "high")
    assert pressured["factors"]["confidence"] == "negative"
    assert pressured["factors"]["stress"] == "negative"


def test_mixed_state_falls_between_the_other_two():
    """Comparative, not a hardcoded number: reweighting the formula should not
    fail this test as long as the intended ordering survives."""
    strong = assess_psychology(STRONG)["mental_readiness"]
    pressured = assess_psychology(HIGH_PRESSURE)["mental_readiness"]
    mixed = assess_psychology(MIXED)["mental_readiness"]
    assert pressured < mixed < strong


def test_mixed_state_keeps_high_motivation_positive_despite_high_stress():
    """Every dimension is classified independently -- one bad dimension must
    not drag the others' labels with it."""
    result = assess_psychology(MIXED)
    assert result["factors"]["motivation"] == "positive"
    assert result["factors"]["stress"] == "negative"


# ---------------------------------------------------------------------------
# Determinism and edge cases
# ---------------------------------------------------------------------------
def test_identical_input_produces_identical_output():
    assert assess_psychology(STRONG) == assess_psychology(STRONG)


def test_determinism_holds_for_every_fixture():
    for responses in (STRONG, HIGH_PRESSURE, MIXED, ALL_MIN, ALL_MAX):
        assert assess_psychology(responses) == assess_psychology(responses)


def test_two_players_with_identical_answers_score_identically():
    """No per-player state, no lookup table, no randomness."""
    assert assess_psychology(MIXED) == assess_psychology(dict(MIXED))


@pytest.mark.parametrize("responses", [ALL_MIN, ALL_MAX])
def test_extreme_inputs_do_not_error_and_stay_in_range(responses):
    result = assess_psychology(responses)
    for key in ("mental_readiness", "focus", "confidence", "stress"):
        assert 0 <= result[key] <= 100


def test_all_minimum_answers_put_every_dimension_at_the_scale_floor():
    """The per-dimension scores hit the documented floor of 10.

    Note mental_readiness itself does NOT land at 0, and should not: an
    all-1s questionnaire reports minimum stress too, and low stress is good.
    The aggregate of a self-contradictory questionnaire is not an extreme, and
    asserting otherwise would be asserting a bug.
    """
    result = assess_psychology(ALL_MIN)
    assert result["focus"] == 10
    assert result["confidence"] == 10
    assert result["stress"] == 10


def test_all_maximum_answers_put_every_dimension_at_the_scale_ceiling():
    result = assess_psychology(ALL_MAX)
    assert result["focus"] == 100
    assert result["confidence"] == 100
    assert result["stress"] == 100


# ---------------------------------------------------------------------------
# The §4 output contract
# ---------------------------------------------------------------------------
def test_contract_shape_and_types():
    result = assess_psychology(STRONG)
    for key in ("mental_readiness", "focus", "confidence", "stress"):
        assert isinstance(result[key], int)
    assert result["pressure_risk"] in ("low", "moderate", "high")
    assert result["mental_performance_risk"] in ("low", "moderate", "high")
    assert set(result["factors"]) == {
        "focus",
        "confidence",
        "stress",
        "error_recovery",
        "motivation",
        "pressure_response",
    }
    assert all(
        label in ("positive", "neutral", "negative") for label in result["factors"].values()
    )


def test_contract_carries_the_persistence_metadata():
    result = assess_psychology(STRONG)
    assert result["method"] == "heuristic_proxy"
    assert result["confidence_level"] in (
        "normal",
        "low_sample",
        "low_upstream_confidence",
    )
    assert result["sample_size"] == len(FEATURE_NAMES)
    assert result["schema_version"]
    assert result["data_source"] == "self_reported"


def test_confidence_score_and_confidence_level_are_separate_keys():
    """The 0-100 confidence SCORE and the MetricConfidence TIER must not
    collide -- they are different things that both want the word 'confidence'."""
    result = assess_psychology(STRONG)
    assert isinstance(result["confidence"], int)
    assert isinstance(result["confidence_level"], str)


def test_model_reports_low_sample_when_features_are_missing():
    result = HeuristicReadinessModel().predict({})
    assert result["mental_readiness"] is None
    assert result["confidence_level"] == "low_sample"
    assert result["factors"] == {}


def test_heuristic_model_implements_the_interface():
    assert isinstance(HeuristicReadinessModel(), PsychologyReadinessModel)


def test_a_substitute_model_can_be_injected():
    """The seam a future XGBoost/LSTM model plugs into: assess_psychology must
    route through whatever model it is handed, not a hardcoded heuristic."""

    class _StubModel(PsychologyReadinessModel):
        def predict(self, features, historical=None):
            return {
                "mental_readiness": 42,
                "focus": 1,
                "confidence": 2,
                "stress": 3,
                "pressure_risk": "low",
                "mental_performance_risk": "low",
                "factors": {},
                "sub_scores": {},
                "method": "ml_trained",
                "confidence_level": "normal",
                "sample_size": 10,
                "schema_version": "stub",
                "data_source": "self_reported",
                "historical_context": [],
            }

    result = assess_psychology(STRONG, model=_StubModel())
    assert result["mental_readiness"] == 42
    assert result["method"] == "ml_trained"


# ---------------------------------------------------------------------------
# Historical integration (§5)
# ---------------------------------------------------------------------------
def test_history_never_changes_a_headline_score():
    """Corroboration only. If history could move the number, "works from
    self-report alone" would not be true."""
    without = assess_psychology(STRONG)
    with_history = assess_psychology(
        STRONG,
        historical={"focus_proxy": {"value": 12.0, "confidence": "normal"}},
    )
    for key in ("mental_readiness", "focus", "confidence", "stress"):
        assert without[key] == with_history[key]
    assert without["factors"] == with_history["factors"]


def test_history_is_surfaced_as_a_labelled_performance_proxy():
    result = assess_psychology(
        STRONG,
        historical={"focus_proxy": {"value": 71.0, "confidence": "normal"}},
    )
    assert result["historical_context"]
    assert "proxy" in result["historical_context"][0]
    assert result["sub_scores"]["historical_proxies"]["focus_proxy"] == 71.0


def test_wholly_unusable_history_downgrades_confidence_only():
    result = assess_psychology(
        STRONG,
        historical={"focus_proxy": {"value": None, "confidence": "low_upstream_confidence"}},
    )
    assert result["confidence_level"] == "low_upstream_confidence"
    assert result["mental_readiness"] == assess_psychology(STRONG)["mental_readiness"]
