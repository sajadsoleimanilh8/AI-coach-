"""
Questionnaire answers -> normalized 0-100 feature vector.

Pure functions, no state, no I/O, no clock, no randomness. This module is the
ONLY place raw 1-10 answers become 0-100 features: nothing downstream ever
sees a raw answer, so there is exactly one scaling rule to audit and no way
for two layers to disagree about what "7" means.

Every formula below is a plain arithmetic statement of the product spec, kept
visible and commented rather than hidden behind a normalizer object. A reader
should be able to check any single feature against the answer it came from
without leaving this file.

WHAT THIS IS NOT
----------------
Not emotion detection and not a psychological assessment. These features
describe what a player reported about their own readiness for a match. The
names are performance names on purpose -- "pressure_sensitivity",
"error_recovery" -- and no feature here is, or may be presented as, a measure
of a mental-health state.

DIRECTION IS NOT UNIFORM
------------------------
`focus_score` is higher-is-better while `stress_score` and
`pressure_sensitivity` are higher-is-worse. Rather than silently flipping some
inputs so they all point the same way (which would make the stored feature
vector disagree with the questionnaire it came from), each feature keeps the
direction its own question has, and FEATURE_DIRECTIONS states that direction
explicitly for every one of them. The scoring layer inverts where it needs to.

WHAT IS DELIBERATELY NOT COMPUTED HERE
--------------------------------------
`mental_readiness`. It is the scoring layer's *output*, not an input: a future
XGBoost/LSTM model implementing PsychologyReadinessModel would consume exactly
the vector below and produce mental_readiness from it, so computing it here
would make the input and the prediction the same object. It rejoins the stored
feature record one layer up, in model_interface.py. Keeping that boundary is
what makes the model interface genuinely swappable.
"""

from __future__ import annotations

from ai.psychology_ai.constants import (
    CATEGORICAL_ITEMS,
    PRESSURE_CONTEXT_WEIGHT,
    PRESSURE_EFFECT_BASE,
    PRESSURE_EFFECT_VALUES,
    PRESSURE_EFFECT_WEIGHT,
    SCALE_ITEMS,
    SCALE_MAX,
    SCALE_MIN,
    clamp,
    mean,
    scale_1_to_10,
)

# Direction of every extracted feature, for anyone (human or model) reading a
# stored vector. "higher_is_better" -> more is good; "higher_is_worse" -> more
# is bad. Served alongside the vector by the API's /features endpoint, because
# a bare list of 0-100 numbers is ambiguous without it.
FEATURE_DIRECTIONS: dict[str, str] = {
    "focus_score": "higher_is_better",
    "mental_clarity": "higher_is_better",
    "stress_score": "higher_is_worse",
    "nervousness_score": "higher_is_worse",
    "confidence_score": "higher_is_better",
    "tactical_confidence": "higher_is_better",
    "motivation_score": "higher_is_better",
    "competitive_motivation": "higher_is_better",
    "pressure_sensitivity": "higher_is_worse",
    "error_recovery": "higher_is_better",
}

# The features the extractor emits, in the order the spec lists them. Used by
# the tests to assert the contract has not silently grown or shrunk.
FEATURE_NAMES = tuple(FEATURE_DIRECTIONS)


class QuestionnaireValidationError(ValueError):
    """Raised when a submitted answer is outside the range the questionnaire
    can honestly represent.

    A ValueError subclass so callers that just want "bad input" semantics need
    not import this name. Nothing is clamped: silently rounding an impossible
    12 down to 10 would turn a data-entry mistake into a real-looking answer.

    The API layer rarely reaches this -- Pydantic rejects the same ranges with
    a 422 first, from constraints built on the same constants -- but a caller
    using this engine directly gets the identical rule.
    """


def _require_scale(name: str, value: object) -> int:
    """Validates one 1-10 item and returns it as an int.

    `bool` is rejected explicitly because it is a subclass of `int` in Python,
    so `True` would otherwise sail through as a perfectly valid rating of 1.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise QuestionnaireValidationError(
            f"{name} must be an integer {SCALE_MIN}-{SCALE_MAX}, got {value!r}"
        )
    if not SCALE_MIN <= value <= SCALE_MAX:
        raise QuestionnaireValidationError(
            f"{name} must be between {SCALE_MIN} and {SCALE_MAX}, got {value}"
        )
    return value


def _require_pressure_effect(value: object) -> str:
    if value not in PRESSURE_EFFECT_VALUES:
        raise QuestionnaireValidationError(
            f"pressure_performance_effect must be one of "
            f"{list(PRESSURE_EFFECT_VALUES)}, got {value!r}"
        )
    return str(value)


def validate_responses(responses: dict) -> dict:
    """Checks all 13 items are present and in range; returns them normalized
    to plain ints/strings.

    Separated from extract_features so the validation rule can be reused (and
    tested) on its own, and so extract_features itself stays a straight
    arithmetic function over already-valid answers.
    """
    if not isinstance(responses, dict):
        raise QuestionnaireValidationError(
            f"responses must be a dict of questionnaire answers, got {type(responses).__name__}"
        )

    missing = [
        item
        for item in SCALE_ITEMS + CATEGORICAL_ITEMS
        if responses.get(item) is None
    ]
    if missing:
        raise QuestionnaireValidationError(
            f"missing questionnaire answers: {', '.join(sorted(missing))}"
        )

    clean: dict = {item: _require_scale(item, responses[item]) for item in SCALE_ITEMS}
    clean["pressure_performance_effect"] = _require_pressure_effect(
        responses["pressure_performance_effect"]
    )
    return clean


def _pressure_sensitivity(pressure_performance_effect: str, importance_pressure: int) -> float:
    """0-100, higher = MORE sensitive to pressure (i.e. worse).

    Exact formula, also stated in constants.py next to its weights:

        base    = PRESSURE_EFFECT_BASE[effect]   # improves 20 / no_change 50 / reduces 80
        context = importance_pressure * 10       # the 1-10 answer on the 0-100 scale
        result  = base * 0.65 + context * 0.35

    The categorical answer dominates because it is the player's own read on how
    pressure affects them; match importance is the situational context
    deciding how much that disposition is about to be tested. Fully
    deterministic -- a lookup and two multiplications, no randomness anywhere.

    Worked bounds (asserted in the tests):
      improves  + importance 1  -> 0.65*20 + 0.35*10  = 16.5  (low)
      no_change + importance 5  -> 0.65*50 + 0.35*50  = 50.0  (mid)
      reduces   + importance 10 -> 0.65*80 + 0.35*100 = 87.0  (high)
    """
    base = PRESSURE_EFFECT_BASE[pressure_performance_effect]
    context = scale_1_to_10(importance_pressure)
    return clamp(base * PRESSURE_EFFECT_WEIGHT + context * PRESSURE_CONTEXT_WEIGHT)


def extract_features(responses: dict) -> dict:
    """The one entry point: 13 validated answers in, 10 bounded features out.

    Deterministic -- no clock reads, no randomness, no I/O, no external state.
    The same answers always produce the same features, for every player.

    Scaling rule for every 1-10 item is `10 * rating` (see
    constants.scale_1_to_10), so 1 -> 10, 10 -> 100, and mid-values interpolate
    linearly. Note the documented floor: a rating of 1 becomes 10, never 0,
    because the questionnaire's own floor is 1 and treating "1" as "zero"
    would overstate what was reported.

    Raises QuestionnaireValidationError on a missing or out-of-range answer.
    """
    q = validate_responses(responses)

    # --- focus -------------------------------------------------------------
    # Two items averaged then scaled: concentration right now, and the ability
    # to hold it for the whole match.
    focus_score = scale_1_to_10(mean(q["concentration_level"], q["focus_maintenance"]))
    mental_clarity = scale_1_to_10(q["mental_clarity_raw"])

    # --- stress ------------------------------------------------------------
    # Higher is worse. General pre-match stress averaged with the pressure the
    # player attributes specifically to the match's importance.
    stress_score = scale_1_to_10(mean(q["pre_match_stress"], q["importance_pressure"]))
    nervousness_score = scale_1_to_10(q["nervousness"])

    # --- confidence --------------------------------------------------------
    confidence_score = scale_1_to_10(q["performance_confidence"])
    tactical_confidence = scale_1_to_10(q["tactical_confidence_raw"])

    # --- motivation --------------------------------------------------------
    motivation_score = scale_1_to_10(q["match_motivation"])
    competitive_motivation = scale_1_to_10(q["competitive_motivation_raw"])

    # --- pressure response -------------------------------------------------
    pressure_sensitivity = _pressure_sensitivity(
        q["pressure_performance_effect"], q["importance_pressure"]
    )
    # How quickly they say they recover after a mistake, averaged with how calm
    # they say they stay after one. Higher is better.
    error_recovery = scale_1_to_10(mean(q["mistake_recovery_speed"], q["post_error_calm"]))

    return {
        "focus_score": round(focus_score, 2),
        "mental_clarity": round(mental_clarity, 2),
        "stress_score": round(stress_score, 2),
        "nervousness_score": round(nervousness_score, 2),
        "confidence_score": round(confidence_score, 2),
        "tactical_confidence": round(tactical_confidence, 2),
        "motivation_score": round(motivation_score, 2),
        "competitive_motivation": round(competitive_motivation, 2),
        "pressure_sensitivity": round(pressure_sensitivity, 2),
        "error_recovery": round(error_recovery, 2),
    }
