"""
Constants and small pure helpers for the Pre-Match Psychology Intelligence
mental-readiness engine.
"""

from __future__ import annotations

SCHEMA_VERSION = "v1"

METHOD_HEURISTIC = "heuristic_proxy"

CONFIDENCE_NORMAL = "normal"
CONFIDENCE_LOW_SAMPLE = "low_sample"
CONFIDENCE_LOW_UPSTREAM = "low_upstream_confidence"

DATA_SOURCE = "self_reported"


SCALE_MIN = 1
SCALE_MAX = 10

PRESSURE_EFFECT_IMPROVES = "improves"
PRESSURE_EFFECT_NO_CHANGE = "no_change"
PRESSURE_EFFECT_REDUCES = "reduces"
PRESSURE_EFFECT_VALUES = (
    PRESSURE_EFFECT_IMPROVES,
    PRESSURE_EFFECT_NO_CHANGE,
    PRESSURE_EFFECT_REDUCES,
)

SCALE_ITEMS = (
    "concentration_level",
    "focus_maintenance",
    "mental_clarity_raw",
    "pre_match_stress",
    "importance_pressure",
    "nervousness",
    "performance_confidence",
    "tactical_confidence_raw",
    "match_motivation",
    "competitive_motivation_raw",
    "mistake_recovery_speed",
    "post_error_calm",
)
CATEGORICAL_ITEMS = ("pressure_performance_effect",)
QUESTIONNAIRE_ITEMS = SCALE_ITEMS + CATEGORICAL_ITEMS


PRESSURE_EFFECT_BASE = {
    PRESSURE_EFFECT_IMPROVES: 20.0,
    PRESSURE_EFFECT_NO_CHANGE: 50.0,
    PRESSURE_EFFECT_REDUCES: 80.0,
}
PRESSURE_EFFECT_WEIGHT = 0.65
PRESSURE_CONTEXT_WEIGHT = 0.35


FOCUS_WEIGHTS = {"focus": 0.60, "mental_clarity": 0.40}

CONFIDENCE_WEIGHTS = {"performance": 0.60, "tactical": 0.40}

STRESS_WEIGHTS = {"stress": 0.60, "nervousness": 0.40}

MOTIVATION_WEIGHTS = {"match": 0.55, "competitive": 0.45}

READINESS_WEIGHTS = {
    "focus": 0.25,
    "confidence": 0.25,
    "inverse_stress": 0.20,
    "motivation": 0.15,
    "error_recovery": 0.15,
}


PRESSURE_PENALTY_KNEE = 60.0
PRESSURE_PENALTY_SLOPE = 0.25

PRESSURE_RISK_WEIGHTS = {"sensitivity": 0.60, "stress": 0.40}
PRESSURE_RISK_HIGH_MIN = 65.0
PRESSURE_RISK_MODERATE_MIN = 45.0

PERFORMANCE_RISK_LOW_MIN = 70.0
PERFORMANCE_RISK_MODERATE_MIN = 45.0

FACTOR_POSITIVE_MIN = 65.0
FACTOR_NEGATIVE_MAX = 45.0

RISK_LOW = "low"
RISK_MODERATE = "moderate"
RISK_HIGH = "high"

FACTOR_POSITIVE = "positive"
FACTOR_NEGATIVE = "negative"
FACTOR_NEUTRAL = "neutral"

MIN_SAMPLE_ITEMS = 2


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Bounds a score to [low, high]. Every sub-score and headline score in
    this engine passes through here, so no formula can emit an out-of-range
    number for the API/DB layer to store."""
    return float(min(max(value, low), high))


def scale_1_to_10(rating: float) -> float:
    """Maps a 1-10 self-rating onto 0-100 as `10 * rating`, so 1 -> 10 and
    10 -> 100.
    """
    return float(10.0 * rating)


def invert(score_0_100: float) -> float:
    """Turns a 'higher is worse' score (stress, pressure sensitivity) into a
    'higher is better' goodness score, so factor classification and the
    readiness formula can use one direction for every dimension."""
    return clamp(100.0 - score_0_100)


def mean(*values: float) -> float:
    """Arithmetic mean. Present so the composite formulas below read the way
    the spec states them (`mean(a, b) * 10`) rather than as inlined division."""
    return float(sum(values)) / len(values)
