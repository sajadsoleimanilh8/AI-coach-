"""
The intelligence layer: a scorer interface plus the one implementation that
actually exists today.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

from ai.performance_ai.match_readiness_predictor.constants import (
    DATA_SOURCE,
    FACTOR_NEGATIVE,
    FACTOR_NEGATIVE_MAX,
    FACTOR_NEUTRAL,
    FACTOR_POSITIVE,
    FACTOR_POSITIVE_MIN,
    FATIGUE_WEIGHTS,
    METHOD_HEURISTIC,
    PERFORMANCE_RISK_LOW_MIN,
    PERFORMANCE_RISK_MODERATE_MIN,
    READINESS_WEIGHTS,
    RECOVERY_WEIGHTS,
    RISK_HIGH,
    RISK_LOW,
    RISK_MODERATE,
    SCHEMA_VERSION,
    TRAINING_LOAD_PENALTY_KNEE,
    TRAINING_LOAD_PENALTY_SLOPE,
    WORKLOAD_RISK_HIGH_LOAD_MIN,
    WORKLOAD_RISK_INTENSITY_MIN,
    WORKLOAD_RISK_LOW_LOAD_MAX,
    clamp,
    invert,
)
from ai.performance_ai.match_readiness_predictor.features import PreMatchFeatures

_DIMENSION_PHRASES: dict[str, tuple[str, str, str]] = {
    "sleep": ("good sleep quality and duration", "insufficient or poor sleep", "adequate sleep"),
    "recent_training_load": (
        "low recent training load",
        "high recent training load",
        "moderate recent training load",
    ),
    "fatigue": ("low overall fatigue", "high overall fatigue", "moderate overall fatigue"),
    "muscle_soreness": ("little muscle soreness", "high muscle soreness", "some muscle soreness"),
    "pain": ("little self-reported pain", "high self-reported pain", "some self-reported pain"),
    "hydration": ("good hydration", "low hydration", "moderate hydration"),
    "nutrition": ("good nutrition", "poor nutrition", "moderate nutrition"),
    "perceived_readiness": (
        "high self-perceived readiness",
        "low self-perceived readiness",
        "moderate self-perceived readiness",
    ),
    "recovery": ("strong overall recovery", "poor overall recovery", "partial overall recovery"),
}


@dataclass(frozen=True)
class Factor:
    """One input dimension, classified. This is what makes an assessment
    explainable rather than a bare number: the intermediates are kept and
    stored, not thrown away once the headline score is computed."""

    dimension: str
    label: str
    score: float
    detail: str


@dataclass(frozen=True)
class HealthAssessment:
    """Everything the heuristic derived, ready to persist and serve."""

    physical_readiness: float
    fatigue_score: float
    recovery_score: float
    performance_risk: str
    workload_risk: str
    factors: list[Factor]
    features: PreMatchFeatures
    method: str = METHOD_HEURISTIC
    schema_version: str = SCHEMA_VERSION
    data_source: str = DATA_SOURCE
    notes: str | None = field(default=None)

    def key_positive_factors(self) -> list[str]:
        return [f.detail for f in self.factors if f.label == FACTOR_POSITIVE]

    def key_negative_factors(self) -> list[str]:
        return [f.detail for f in self.factors if f.label == FACTOR_NEGATIVE]

    def to_feature_vector(self) -> dict:
        """The complete named feature set: the normalized questionnaire
        inputs plus the three headline scores derived from them.
        """
        vector = self.features.as_dict()
        vector.update(
            {
                "fatigue_score": self.fatigue_score,
                "recovery_score": self.recovery_score,
                "physical_readiness": self.physical_readiness,
            }
        )
        return vector

    def as_dict(self) -> dict:
        return {
            "physical_readiness": self.physical_readiness,
            "fatigue_score": self.fatigue_score,
            "recovery_score": self.recovery_score,
            "performance_risk": self.performance_risk,
            "workload_risk": self.workload_risk,
            "factors": [asdict(f) for f in self.factors],
            "features": self.to_feature_vector(),
            "method": self.method,
            "schema_version": self.schema_version,
            "data_source": self.data_source,
            "notes": self.notes,
        }


class ReadinessScorer(ABC):
    """Strategy interface between feature extraction and the API/DB layer."""

    @abstractmethod
    def score(self, features: PreMatchFeatures) -> HealthAssessment:
        """Deterministic: the same features must always produce the same
        assessment. No clock reads, no randomness, no I/O."""


def _training_load_penalty(recent_training_load: float) -> float:
    """Points deducted from physical_readiness for carrying heavy recent
    load. Zero below the knee, then linear -- reaching 10 points at a
    maxed-out load of 100. Deliberately gentle: heavy recent load is a real
    readiness drag, but it is already represented inside fatigue_score, and
    """
    if recent_training_load <= TRAINING_LOAD_PENALTY_KNEE:
        return 0.0
    return TRAINING_LOAD_PENALTY_SLOPE * (recent_training_load - TRAINING_LOAD_PENALTY_KNEE)


def _classify(dimension: str, goodness: float) -> Factor:
    """Fixed thresholds, one rule for every dimension: >= 65 positive,
    <= 45 negative, otherwise neutral."""
    positive, negative, neutral = _DIMENSION_PHRASES[dimension]
    if goodness >= FACTOR_POSITIVE_MIN:
        label, detail = FACTOR_POSITIVE, positive
    elif goodness <= FACTOR_NEGATIVE_MAX:
        label, detail = FACTOR_NEGATIVE, negative
    else:
        label, detail = FACTOR_NEUTRAL, neutral
    return Factor(dimension=dimension, label=label, score=round(goodness, 2), detail=detail)


def _performance_risk(physical_readiness: float) -> str:
    if physical_readiness >= PERFORMANCE_RISK_LOW_MIN:
        return RISK_LOW
    if physical_readiness >= PERFORMANCE_RISK_MODERATE_MIN:
        return RISK_MODERATE
    return RISK_HIGH


def _workload_risk(f: PreMatchFeatures) -> str:
    """Workload risk is about the training pattern itself, independent of how
    ready the player feels -- a player can feel fine and still be carrying a
    spike. Kept separate from performance_risk for that reason."""
    trained_24h = f.trained_last_24h_flag > 0
    high_intensity = f.high_intensity_flag > 0
    hard_session_yesterday = (
        trained_24h
        and high_intensity
        and f.training_intensity >= WORKLOAD_RISK_INTENSITY_MIN * 10
    )
    if f.recent_training_load >= WORKLOAD_RISK_HIGH_LOAD_MIN or hard_session_yesterday:
        return RISK_HIGH
    if f.recent_training_load < WORKLOAD_RISK_LOW_LOAD_MAX and not (
        trained_24h and high_intensity
    ):
        return RISK_LOW
    return RISK_MODERATE


class HeuristicReadinessScorer(ReadinessScorer):
    """Explainable, deterministic, arithmetic-only scoring."""

    method = METHOD_HEURISTIC

    def score(self, features: PreMatchFeatures) -> HealthAssessment:
        f = features

        recovery_score = clamp(
            RECOVERY_WEIGHTS["fatigue"] * invert(f.self_reported_fatigue)
            + RECOVERY_WEIGHTS["muscle_soreness"] * invert(f.muscle_soreness)
            + RECOVERY_WEIGHTS["pain"] * invert(f.pain_level)
            + RECOVERY_WEIGHTS["perceived_readiness"] * f.perceived_readiness
        )

        fatigue_score = clamp(
            FATIGUE_WEIGHTS["self_reported_fatigue"] * f.self_reported_fatigue
            + FATIGUE_WEIGHTS["recent_training_load"] * f.recent_training_load
            + FATIGUE_WEIGHTS["sleep_debt"] * f.sleep_debt
            + FATIGUE_WEIGHTS["high_intensity"] * f.high_intensity_load
        )

        physical_readiness = clamp(
            READINESS_WEIGHTS["recovery"] * recovery_score
            + READINESS_WEIGHTS["inverse_fatigue"] * invert(fatigue_score)
            + READINESS_WEIGHTS["hydration"] * f.hydration_score
            + READINESS_WEIGHTS["nutrition"] * f.nutrition_score
            + READINESS_WEIGHTS["perceived_readiness"] * f.perceived_readiness
            - _training_load_penalty(f.recent_training_load)
        )

        recovery_score = round(recovery_score, 1)
        fatigue_score = round(fatigue_score, 1)
        physical_readiness = round(physical_readiness, 1)

        factors = [
            _classify("sleep", f.sleep_score),
            _classify("recent_training_load", invert(f.recent_training_load)),
            _classify("fatigue", invert(fatigue_score)),
            _classify("muscle_soreness", invert(f.muscle_soreness)),
            _classify("pain", invert(f.pain_level)),
            _classify("hydration", f.hydration_score),
            _classify("nutrition", f.nutrition_score),
            _classify("perceived_readiness", f.perceived_readiness),
            _classify("recovery", recovery_score),
        ]

        return HealthAssessment(
            physical_readiness=physical_readiness,
            fatigue_score=fatigue_score,
            recovery_score=recovery_score,
            performance_risk=_performance_risk(physical_readiness),
            workload_risk=_workload_risk(f),
            factors=factors,
            features=f,
        )
