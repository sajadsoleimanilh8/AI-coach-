"""
The validated pre-match self-report -- the single input to this engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time

from ai.performance_ai.match_readiness_predictor.constants import (
    SCALE_MAX,
    SCALE_MIN,
    SLEEP_HOURS_MAX,
    SLEEP_HOURS_MIN_EXCLUSIVE,
)


class QuestionnaireValidationError(ValueError):
    """Raised when a submitted answer is outside the range the questionnaire
    can honestly represent. A ValueError subclass so callers that just want
    'bad input' semantics need not import this name."""


def _require_scale(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise QuestionnaireValidationError(
            f"{name} must be an integer {SCALE_MIN}-{SCALE_MAX}, got {value!r}"
        )
    if not SCALE_MIN <= value <= SCALE_MAX:
        raise QuestionnaireValidationError(
            f"{name} must be between {SCALE_MIN} and {SCALE_MAX}, got {value}"
        )


def _require_non_negative(name: str, value: float) -> None:
    if value is None or value < 0:
        raise QuestionnaireValidationError(f"{name} must be >= 0, got {value!r}")


@dataclass(frozen=True)
class PreMatchQuestionnaireInput:
    """One player's self-report for one upcoming match."""

    player_id: str

    match_id: str | None = None

    sleep_duration_hours: float = 8.0
    sleep_quality: int = 5
    bedtime: time | None = None
    wake_time: time | None = None
    night_awakenings: int = 0

    trained_last_24h: bool = False
    trained_last_48h: bool = False
    training_duration_minutes: float = 0.0
    training_intensity: int = 1
    high_intensity_activity: bool = False
    hours_since_last_training: float | None = None

    fatigue: int = 5
    muscle_soreness: int = 5
    pain_level: int = 1
    perceived_readiness: int = 5

    hydration_liters: float = 0.0
    nutrition_quality: int = 5
    hours_since_last_meal: float = 0.0

    caffeine_or_supplement_notes: str | None = field(default=None)

    def __post_init__(self) -> None:
        if not self.player_id or not str(self.player_id).strip():
            raise QuestionnaireValidationError("player_id must be a non-empty string")

        if self.sleep_duration_hours is None or not (
            SLEEP_HOURS_MIN_EXCLUSIVE < self.sleep_duration_hours <= SLEEP_HOURS_MAX
        ):
            raise QuestionnaireValidationError(
                f"sleep_duration_hours must be > {SLEEP_HOURS_MIN_EXCLUSIVE} and "
                f"<= {SLEEP_HOURS_MAX}, got {self.sleep_duration_hours!r}"
            )

        _require_scale("sleep_quality", self.sleep_quality)
        _require_scale("training_intensity", self.training_intensity)
        _require_scale("fatigue", self.fatigue)
        _require_scale("muscle_soreness", self.muscle_soreness)
        _require_scale("pain_level", self.pain_level)
        _require_scale("perceived_readiness", self.perceived_readiness)
        _require_scale("nutrition_quality", self.nutrition_quality)

        if isinstance(self.night_awakenings, bool) or not isinstance(self.night_awakenings, int):
            raise QuestionnaireValidationError(
                f"night_awakenings must be an integer >= 0, got {self.night_awakenings!r}"
            )
        _require_non_negative("night_awakenings", self.night_awakenings)

        _require_non_negative("training_duration_minutes", self.training_duration_minutes)
        _require_non_negative("hydration_liters", self.hydration_liters)
        _require_non_negative("hours_since_last_meal", self.hours_since_last_meal)

        if self.hours_since_last_training is not None:
            _require_non_negative("hours_since_last_training", self.hours_since_last_training)

    def to_storable_dict(self) -> dict:
        """JSON-serializable mirror of the raw answers, for the
        PreMatchQuestionnaire row. Times become "HH:MM" strings; everything
        else is already a primitive."""
        return {
            "player_id": self.player_id,
            "match_id": self.match_id,
            "sleep_duration_hours": self.sleep_duration_hours,
            "sleep_quality": self.sleep_quality,
            "bedtime": self.bedtime.strftime("%H:%M") if self.bedtime else None,
            "wake_time": self.wake_time.strftime("%H:%M") if self.wake_time else None,
            "night_awakenings": self.night_awakenings,
            "trained_last_24h": self.trained_last_24h,
            "trained_last_48h": self.trained_last_48h,
            "training_duration_minutes": self.training_duration_minutes,
            "training_intensity": self.training_intensity,
            "high_intensity_activity": self.high_intensity_activity,
            "hours_since_last_training": self.hours_since_last_training,
            "fatigue": self.fatigue,
            "muscle_soreness": self.muscle_soreness,
            "pain_level": self.pain_level,
            "perceived_readiness": self.perceived_readiness,
            "hydration_liters": self.hydration_liters,
            "nutrition_quality": self.nutrition_quality,
            "hours_since_last_meal": self.hours_since_last_meal,
            "caffeine_or_supplement_notes": self.caffeine_or_supplement_notes,
        }
