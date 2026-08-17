"""
Constants and small pure helpers for the Pre-Match Health Intelligence
readiness engine.
"""

from __future__ import annotations

SCHEMA_VERSION = "v1"

METHOD_HEURISTIC = "heuristic_proxy"

DATA_SOURCE = "self_reported"


SCALE_MIN = 1
SCALE_MAX = 10

SLEEP_HOURS_MIN_EXCLUSIVE = 0.0
SLEEP_HOURS_MAX = 24.0


SLEEP_TARGET_HOURS = 8.0
SLEEP_FLOOR_HOURS = 3.0

AWAKENING_PENALTY_PER_EVENT = 15.0

HYDRATION_TARGET_LITERS = 3.0

TRAINING_DURATION_FULL_LOAD_MIN = 120.0

TRAINING_RECENCY_DECAY_HOURS = 72.0

TRAINING_RECENCY_48H_FACTOR = 0.6

HIGH_INTENSITY_LOAD_BONUS = 15.0

MEAL_TIMING_IDEAL_MIN_HOURS = 1.5
MEAL_TIMING_IDEAL_MAX_HOURS = 4.0
MEAL_TIMING_RECENT_FLOOR = 60.0
MEAL_TIMING_LATE_PENALTY_PER_HOUR = 12.5


SLEEP_WEIGHTS = {"duration": 0.55, "quality": 0.35, "awakenings": 0.10}

SESSION_LOAD_WEIGHTS = {"duration": 0.45, "intensity": 0.55}

RECOVERY_WEIGHTS = {
    "fatigue": 0.30,
    "muscle_soreness": 0.25,
    "pain": 0.25,
    "perceived_readiness": 0.20,
}

FATIGUE_WEIGHTS = {
    "self_reported_fatigue": 0.40,
    "recent_training_load": 0.35,
    "sleep_debt": 0.15,
    "high_intensity": 0.10,
}

NUTRITION_WEIGHTS = {"quality": 0.75, "timing": 0.25}

READINESS_WEIGHTS = {
    "recovery": 0.30,
    "inverse_fatigue": 0.25,
    "hydration": 0.15,
    "nutrition": 0.10,
    "perceived_readiness": 0.20,
}


TRAINING_LOAD_PENALTY_KNEE = 60.0
TRAINING_LOAD_PENALTY_SLOPE = 0.25

PERFORMANCE_RISK_LOW_MIN = 70.0
PERFORMANCE_RISK_MODERATE_MIN = 45.0

WORKLOAD_RISK_HIGH_LOAD_MIN = 70.0
WORKLOAD_RISK_LOW_LOAD_MAX = 40.0
WORKLOAD_RISK_INTENSITY_MIN = 8

FACTOR_POSITIVE_MIN = 65.0
FACTOR_NEGATIVE_MAX = 45.0

RISK_LOW = "low"
RISK_MODERATE = "moderate"
RISK_HIGH = "high"

FACTOR_POSITIVE = "positive"
FACTOR_NEGATIVE = "negative"
FACTOR_NEUTRAL = "neutral"


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Bounds a score to [low, high]. Every sub-score and headline score in
    this engine passes through here, so no formula can emit an out-of-range
    number for the API/DB layer to store."""
    return float(min(max(value, low), high))


def scale_1_to_10(rating: int) -> float:
    """Maps a 1-10 self-rating onto 0-100 the way the spec's formulas do
    (10 * rating), so a rating of 1 is 10 and a rating of 10 is 100. Note
    this deliberately never reaches 0 -- the questionnaire's own floor is 1,
    and pretending a "1" means "zero" would overstate what was reported."""
    return float(10 * rating)


def invert(score_0_100: float) -> float:
    """Turns a 'higher is worse' score (fatigue, training load) into a
    'higher is better' goodness score, so factor classification can use one
    pair of thresholds for every dimension."""
    return clamp(100.0 - score_0_100)
