"""
Questionnaire answers -> normalized 0-100 feature vector.
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

FEATURE_NAMES = tuple(FEATURE_DIRECTIONS)


class QuestionnaireValidationError(ValueError):
    """Raised when a submitted answer is outside the range the questionnaire
    can honestly represent.
    """


def _require_scale(name: str, value: object) -> int:
    """Validates one 1-10 item and returns it as an int."""
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
    """0-100, higher = MORE sensitive to pressure (i.e. worse)."""
    base = PRESSURE_EFFECT_BASE[pressure_performance_effect]
    context = scale_1_to_10(importance_pressure)
    return clamp(base * PRESSURE_EFFECT_WEIGHT + context * PRESSURE_CONTEXT_WEIGHT)


def extract_features(responses: dict) -> dict:
    """The one entry point: 13 validated answers in, 10 bounded features out."""
    q = validate_responses(responses)

    focus_score = scale_1_to_10(mean(q["concentration_level"], q["focus_maintenance"]))
    mental_clarity = scale_1_to_10(q["mental_clarity_raw"])

    stress_score = scale_1_to_10(mean(q["pre_match_stress"], q["importance_pressure"]))
    nervousness_score = scale_1_to_10(q["nervousness"])

    confidence_score = scale_1_to_10(q["performance_confidence"])
    tactical_confidence = scale_1_to_10(q["tactical_confidence_raw"])

    motivation_score = scale_1_to_10(q["match_motivation"])
    competitive_motivation = scale_1_to_10(q["competitive_motivation_raw"])

    pressure_sensitivity = _pressure_sensitivity(
        q["pressure_performance_effect"], q["importance_pressure"]
    )
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
