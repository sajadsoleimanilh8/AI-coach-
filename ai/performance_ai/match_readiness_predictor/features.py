"""
Questionnaire -> normalized feature vector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import time

from ai.performance_ai.match_readiness_predictor.constants import (
    AWAKENING_PENALTY_PER_EVENT,
    HIGH_INTENSITY_LOAD_BONUS,
    HYDRATION_TARGET_LITERS,
    MEAL_TIMING_IDEAL_MAX_HOURS,
    MEAL_TIMING_IDEAL_MIN_HOURS,
    MEAL_TIMING_LATE_PENALTY_PER_HOUR,
    MEAL_TIMING_RECENT_FLOOR,
    NUTRITION_WEIGHTS,
    SESSION_LOAD_WEIGHTS,
    SLEEP_FLOOR_HOURS,
    SLEEP_TARGET_HOURS,
    SLEEP_WEIGHTS,
    TRAINING_DURATION_FULL_LOAD_MIN,
    TRAINING_RECENCY_48H_FACTOR,
    TRAINING_RECENCY_DECAY_HOURS,
    clamp,
    scale_1_to_10,
)
from ai.performance_ai.match_readiness_predictor.questionnaire import (
    PreMatchQuestionnaireInput,
)

FEATURE_DIRECTIONS: dict[str, str] = {
    "sleep_duration": "higher_is_better",
    "sleep_quality": "higher_is_better",
    "sleep_awakenings_score": "higher_is_better",
    "sleep_score": "higher_is_better",
    "sleep_debt": "higher_is_worse",
    "training_duration": "load",
    "training_intensity": "load",
    "session_load": "load",
    "training_recency_factor": "load",
    "recent_training_load": "load",
    "high_intensity_load": "load",
    "trained_last_24h_flag": "flag",
    "trained_last_48h_flag": "flag",
    "high_intensity_flag": "flag",
    "self_reported_fatigue": "higher_is_worse",
    "muscle_soreness": "higher_is_worse",
    "pain_level": "higher_is_worse",
    "hydration_score": "higher_is_better",
    "nutrition_quality": "higher_is_better",
    "meal_timing_score": "higher_is_better",
    "nutrition_score": "higher_is_better",
    "perceived_readiness": "higher_is_better",
}


@dataclass(frozen=True)
class PreMatchFeatures:
    """Normalized 0-100 model inputs derived from one questionnaire."""

    sleep_duration: float
    sleep_quality: float
    sleep_awakenings_score: float
    sleep_score: float
    sleep_debt: float

    training_duration: float
    training_intensity: float
    session_load: float
    training_recency_factor: float
    recent_training_load: float
    high_intensity_load: float

    trained_last_24h_flag: float
    trained_last_48h_flag: float
    high_intensity_flag: float

    self_reported_fatigue: float
    muscle_soreness: float
    pain_level: float
    perceived_readiness: float

    hydration_score: float
    nutrition_quality: float
    meal_timing_score: float
    nutrition_score: float

    time_in_bed_hours: float | None = None
    sleep_efficiency_pct: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _time_in_bed_hours(bedtime: time | None, wake_time: time | None) -> float | None:
    """Clock hours between bedtime and wake_time, wrapping past midnight."""
    if bedtime is None or wake_time is None:
        return None
    bed_minutes = bedtime.hour * 60 + bedtime.minute
    wake_minutes = wake_time.hour * 60 + wake_time.minute
    delta = (wake_minutes - bed_minutes) % (24 * 60)
    if delta == 0:
        return None
    return round(delta / 60.0, 2)


def _sleep_efficiency_pct(sleep_hours: float, in_bed_hours: float | None) -> float | None:
    """Reported sleep as a percentage of time in bed, capped at 100."""
    if in_bed_hours is None or in_bed_hours <= 0:
        return None
    return round(clamp(100.0 * sleep_hours / in_bed_hours), 1)


def _meal_timing_score(hours_since_last_meal: float) -> float:
    """0-100 on how well-timed the last meal is for an upcoming match."""
    if hours_since_last_meal < MEAL_TIMING_IDEAL_MIN_HOURS:
        span = 100.0 - MEAL_TIMING_RECENT_FLOOR
        return clamp(
            MEAL_TIMING_RECENT_FLOOR
            + span * (hours_since_last_meal / MEAL_TIMING_IDEAL_MIN_HOURS)
        )
    if hours_since_last_meal <= MEAL_TIMING_IDEAL_MAX_HOURS:
        return 100.0
    over = hours_since_last_meal - MEAL_TIMING_IDEAL_MAX_HOURS
    return clamp(100.0 - MEAL_TIMING_LATE_PENALTY_PER_HOUR * over)


def _training_recency_factor(q: PreMatchQuestionnaireInput) -> float:
    """0.0-1.0: how much of the last session's load still counts as "recent"."""
    if q.trained_last_24h:
        flag_factor = 1.0
    elif q.trained_last_48h:
        flag_factor = TRAINING_RECENCY_48H_FACTOR
    else:
        flag_factor = 0.0

    if q.hours_since_last_training is None:
        return flag_factor

    decay = clamp(
        1.0 - (q.hours_since_last_training / TRAINING_RECENCY_DECAY_HOURS), 0.0, 1.0
    )
    return min(flag_factor, decay)


def extract_features(q: PreMatchQuestionnaireInput) -> PreMatchFeatures:
    """The one entry point: a validated questionnaire in, a bounded,
    documented feature vector out. Deterministic -- no clock reads, no
    randomness, no I/O."""

    sleep_duration = clamp(
        100.0
        * (q.sleep_duration_hours - SLEEP_FLOOR_HOURS)
        / (SLEEP_TARGET_HOURS - SLEEP_FLOOR_HOURS)
    )
    sleep_quality = scale_1_to_10(q.sleep_quality)
    sleep_awakenings_score = clamp(
        100.0 - AWAKENING_PENALTY_PER_EVENT * q.night_awakenings
    )
    sleep_score = clamp(
        SLEEP_WEIGHTS["duration"] * sleep_duration
        + SLEEP_WEIGHTS["quality"] * sleep_quality
        + SLEEP_WEIGHTS["awakenings"] * sleep_awakenings_score
    )
    sleep_debt = clamp(100.0 - sleep_score)

    training_duration = clamp(
        100.0 * (q.training_duration_minutes / TRAINING_DURATION_FULL_LOAD_MIN)
    )
    training_intensity = scale_1_to_10(q.training_intensity)
    session_load = clamp(
        SESSION_LOAD_WEIGHTS["duration"] * training_duration
        + SESSION_LOAD_WEIGHTS["intensity"] * training_intensity
    )
    recency_factor = _training_recency_factor(q)
    high_intensity_bonus = HIGH_INTENSITY_LOAD_BONUS if q.high_intensity_activity else 0.0
    recent_training_load = clamp((session_load + high_intensity_bonus) * recency_factor)
    high_intensity_load = clamp(100.0 * recency_factor) if q.high_intensity_activity else 0.0

    self_reported_fatigue = scale_1_to_10(q.fatigue)
    muscle_soreness = scale_1_to_10(q.muscle_soreness)
    pain_level = scale_1_to_10(q.pain_level)
    perceived_readiness = scale_1_to_10(q.perceived_readiness)

    hydration_score = clamp(100.0 * (q.hydration_liters / HYDRATION_TARGET_LITERS))
    nutrition_quality = scale_1_to_10(q.nutrition_quality)
    meal_timing_score = _meal_timing_score(q.hours_since_last_meal)
    nutrition_score = clamp(
        NUTRITION_WEIGHTS["quality"] * nutrition_quality
        + NUTRITION_WEIGHTS["timing"] * meal_timing_score
    )

    time_in_bed_hours = _time_in_bed_hours(q.bedtime, q.wake_time)
    sleep_efficiency_pct = _sleep_efficiency_pct(q.sleep_duration_hours, time_in_bed_hours)

    return PreMatchFeatures(
        sleep_duration=round(sleep_duration, 2),
        sleep_quality=round(sleep_quality, 2),
        sleep_awakenings_score=round(sleep_awakenings_score, 2),
        sleep_score=round(sleep_score, 2),
        sleep_debt=round(sleep_debt, 2),
        training_duration=round(training_duration, 2),
        training_intensity=round(training_intensity, 2),
        session_load=round(session_load, 2),
        training_recency_factor=round(100.0 * recency_factor, 2),
        recent_training_load=round(recent_training_load, 2),
        high_intensity_load=round(high_intensity_load, 2),
        trained_last_24h_flag=100.0 if q.trained_last_24h else 0.0,
        trained_last_48h_flag=100.0 if q.trained_last_48h else 0.0,
        high_intensity_flag=100.0 if q.high_intensity_activity else 0.0,
        self_reported_fatigue=round(self_reported_fatigue, 2),
        muscle_soreness=round(muscle_soreness, 2),
        pain_level=round(pain_level, 2),
        perceived_readiness=round(perceived_readiness, 2),
        hydration_score=round(hydration_score, 2),
        nutrition_quality=round(nutrition_quality, 2),
        meal_timing_score=round(meal_timing_score, 2),
        nutrition_score=round(nutrition_score, 2),
        time_in_bed_hours=time_in_bed_hours,
        sleep_efficiency_pct=sleep_efficiency_pct,
    )
