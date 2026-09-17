"""
The validated pre-match self-report -- the single input to this engine.

Pure Python: a dataclass plus explicit range validation, no Pydantic, no
FastAPI, no DB. That keeps the engine usable (and unit-testable) on its own,
exactly like every other module under ai/, while the API layer gets its 422s
from Pydantic constraints built on the SAME bounds imported from
constants.py -- so the two rejection rules cannot drift.

Out-of-range values raise QuestionnaireValidationError. Nothing is clamped:
silently rounding an impossible 30-hour sleep down to 24 would turn a data-
entry mistake into a real-looking reading.
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
    """One player's self-report for one upcoming match.

    Frozen so nothing downstream can mutate the answers after validation --
    the assessment stored against these answers has to stay explainable by
    them.

    Grouped exactly as the product spec groups them: sleep, physical
    activity, recovery, hydration & nutrition, plus identifiers.
    """

    # --- identifiers -------------------------------------------------------
    # A plain external identifier supplied by the caller/frontend. NOT the
    # integer ByteTrack tracking ID used by PlayerDetection/PlayerTracking/
    # Event/PlayerMetric: those are scoped to a single processed video and
    # carry no cross-match identity (see the module comment in
    # backend/api/player_intelligence.py refusing to resolve them to real
    # players). A pre-match questionnaire is filled in before any tracking
    # exists, so there is no tracking ID to reuse and nothing to resolve
    # against.
    player_id: str

    # Only set when a Match row already exists for the upcoming game (Match
    # rows are created at video-upload time). A genuine pre-match submission
    # legitimately has none yet.
    match_id: str | None = None

    # --- sleep -------------------------------------------------------------
    sleep_duration_hours: float = 8.0   # (0, 24]
    sleep_quality: int = 5              # 1-10
    bedtime: time | None = None
    wake_time: time | None = None
    night_awakenings: int = 0           # >= 0

    # --- physical activity -------------------------------------------------
    trained_last_24h: bool = False
    trained_last_48h: bool = False
    training_duration_minutes: float = 0.0  # >= 0
    training_intensity: int = 1             # 1-10
    high_intensity_activity: bool = False   # sprint / high-intensity work
    hours_since_last_training: float | None = None  # >= 0, None if not recent

    # --- recovery ----------------------------------------------------------
    fatigue: int = 5             # 1-10, higher = more fatigued
    muscle_soreness: int = 5     # 1-10, higher = more sore
    pain_level: int = 1          # 1-10, higher = more self-reported pain
    perceived_readiness: int = 5  # 1-10, higher = feels more ready

    # --- hydration & nutrition ---------------------------------------------
    # Litres, not a 1-10 self-rating. Chosen deliberately: this questionnaire
    # already leans heavily on subjective 1-10 scales (fatigue, soreness,
    # pain, perceived readiness, sleep quality, nutrition quality), and
    # hydration is the one dimension here a player can actually quantify
    # rather than estimate. Scoring it against a stated litre target also
    # makes hydration_score auditable ("2.1L against a 3.0L target") in a way
    # "hydration: 7/10" is not.
    hydration_liters: float = 0.0   # >= 0
    nutrition_quality: int = 5      # 1-10
    hours_since_last_meal: float = 0.0  # >= 0

    # Free text. Informational ONLY -- never reaches the scoring layer (see
    # features.py). Surfaced verbatim on the API response so a coach can read
    # it, but it must not move any number.
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
