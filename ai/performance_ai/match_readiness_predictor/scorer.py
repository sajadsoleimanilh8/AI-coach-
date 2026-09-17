"""
The intelligence layer: a scorer interface plus the one implementation that
actually exists today.

This is a performance-readiness estimation from a self-report. It is not a
medical assessment, not a diagnosis, and not an injury prediction. Nothing in
here should ever be presented as any of those.

Why an interface for a single implementation: no trained readiness/fatigue/
recovery model exists anywhere in this repo (the ai/performance_ai/* folders
are empty stubs), and inventing one -- a "model" that is really a table of
hardcoded numbers -- would be worse than useless, because it would look
trustworthy. So the heuristic is the honest implementation, and the interface
is the seam a real model plugs into later without the API contract moving.
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

# (positive phrase, negative phrase, neutral phrase) per dimension. Plain
# descriptive language about what the player reported -- never diagnostic
# ("high self-reported pain", not "injured").
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
    label: str   # "positive" | "negative" | "neutral"
    score: float  # 0-100 "goodness" -- always higher-is-better, whatever the
                  # underlying feature's own direction was
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
    # Fixed, not inferred: every assessment this module produces comes from a
    # complete, validated self-report. Recorded so no consumer mistakes these
    # numbers for sensor-derived ones.
    data_source: str = DATA_SOURCE
    # Free text carried through untouched from the questionnaire. It reaches
    # this object only so the API can echo it back to a coach -- no formula
    # above ever reads it.
    notes: str | None = field(default=None)

    def key_positive_factors(self) -> list[str]:
        return [f.detail for f in self.factors if f.label == FACTOR_POSITIVE]

    def key_negative_factors(self) -> list[str]:
        return [f.detail for f in self.factors if f.label == FACTOR_NEGATIVE]

    def to_feature_vector(self) -> dict:
        """The complete named feature set: the normalized questionnaire
        inputs plus the three headline scores derived from them.

        This is the shape a future ML/DL model trains on and the shape the
        API's /features endpoint serves. Inputs and outputs live in one dict
        here (rather than in PreMatchFeatures itself) so that the scorer's
        input type stays purely inputs -- see PreMatchFeatures' docstring.
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
    """Strategy interface between feature extraction and the API/DB layer.

    backend/api/prematch_health.py depends on THIS, never on a concrete
    subclass, so swapping the implementation is a one-line change at the
    composition point and touches no endpoint code.

    A future MLReadinessScorer / DLReadinessScorer implements this same
    method, consuming the identical PreMatchFeatures vector (that is why the
    features are normalized and bounded) and returning the identical
    HealthAssessment -- differing only in `method`, which becomes
    MetricMethod.ml_trained instead of heuristic_proxy. Callers reading
    `method` off the response is how a consumer tells which tier produced a
    given number; nothing else about the contract changes.
    """

    @abstractmethod
    def score(self, features: PreMatchFeatures) -> HealthAssessment:
        """Deterministic: the same features must always produce the same
        assessment. No clock reads, no randomness, no I/O."""


def _training_load_penalty(recent_training_load: float) -> float:
    """Points deducted from physical_readiness for carrying heavy recent
    load. Zero below the knee, then linear -- reaching 10 points at a
    maxed-out load of 100. Deliberately gentle: heavy recent load is a real
    readiness drag, but it is already represented inside fatigue_score, and
    double-counting it at full weight would let one dimension dominate the
    headline number."""
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
    # training_intensity is the 1-10 rating scaled to 0-100, so the spec's
    # ">= 8" threshold is >= 80 here.
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
    """Explainable, deterministic, arithmetic-only scoring.

    Every weight and threshold it uses is a named constant in constants.py,
    and every number it emits traces back through one of the formulas below
    to a questionnaire field. There is no lookup table, no per-player special
    case, and no stored state -- two identical questionnaires from two
    different players produce identical assessments, by construction.
    """

    method = METHOD_HEURISTIC

    def score(self, features: PreMatchFeatures) -> HealthAssessment:
        f = features

        # Recovery: how recovered the player reports being right now.
        recovery_score = clamp(
            RECOVERY_WEIGHTS["fatigue"] * invert(f.self_reported_fatigue)
            + RECOVERY_WEIGHTS["muscle_soreness"] * invert(f.muscle_soreness)
            + RECOVERY_WEIGHTS["pain"] * invert(f.pain_level)
            + RECOVERY_WEIGHTS["perceived_readiness"] * f.perceived_readiness
        )

        # Fatigue: how much accumulated load and sleep debt the player is
        # carrying. Higher is worse.
        fatigue_score = clamp(
            FATIGUE_WEIGHTS["self_reported_fatigue"] * f.self_reported_fatigue
            + FATIGUE_WEIGHTS["recent_training_load"] * f.recent_training_load
            + FATIGUE_WEIGHTS["sleep_debt"] * f.sleep_debt
            + FATIGUE_WEIGHTS["high_intensity"] * f.high_intensity_load
        )

        # Headline readiness.
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

        # Every dimension classified against the same thresholds, on a
        # common higher-is-better scale.
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
