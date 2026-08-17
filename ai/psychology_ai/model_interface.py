"""
The intelligence layer: a model interface plus the one implementation that
actually exists today.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ai.psychology_ai.constants import (
    CONFIDENCE_LOW_SAMPLE,
    CONFIDENCE_LOW_UPSTREAM,
    CONFIDENCE_NORMAL,
    CONFIDENCE_WEIGHTS,
    DATA_SOURCE,
    FACTOR_NEGATIVE,
    FACTOR_NEGATIVE_MAX,
    FACTOR_NEUTRAL,
    FACTOR_POSITIVE,
    FACTOR_POSITIVE_MIN,
    FOCUS_WEIGHTS,
    METHOD_HEURISTIC,
    MOTIVATION_WEIGHTS,
    PERFORMANCE_RISK_LOW_MIN,
    PERFORMANCE_RISK_MODERATE_MIN,
    PRESSURE_PENALTY_KNEE,
    PRESSURE_PENALTY_SLOPE,
    PRESSURE_RISK_HIGH_MIN,
    PRESSURE_RISK_MODERATE_MIN,
    PRESSURE_RISK_WEIGHTS,
    READINESS_WEIGHTS,
    RISK_HIGH,
    RISK_LOW,
    RISK_MODERATE,
    SCHEMA_VERSION,
    STRESS_WEIGHTS,
    clamp,
    invert,
)
from ai.psychology_ai.feature_extraction import FEATURE_NAMES

_FACTOR_PHRASES: dict[str, tuple[str, str, str]] = {
    "focus": ("high focus", "reduced focus", "moderate focus"),
    "confidence": ("strong confidence", "low confidence", "moderate confidence"),
    "stress": (
        "low pre-match stress",
        "elevated pre-match stress",
        "moderate pre-match stress",
    ),
    "error_recovery": (
        "good error recovery",
        "slow error recovery",
        "moderate error recovery",
    ),
    "motivation": ("strong motivation", "low motivation", "moderate motivation"),
    "pressure_response": (
        "handles pressure well",
        "sensitive to pressure",
        "neutral pressure response",
    ),
}

FACTOR_DIMENSIONS = (
    "focus",
    "confidence",
    "stress",
    "error_recovery",
    "motivation",
    "pressure_response",
)

HISTORICAL_PROXY_NAMES = (
    "performance_consistency",
    "pressure_response_proxy",
    "focus_proxy",
    "confidence_proxy",
)


class PsychologyReadinessModel(ABC):
    """Strategy interface between feature extraction and the API/DB layer."""

    @abstractmethod
    def predict(self, features: dict, historical: dict | None = None) -> dict:
        """Deterministic: the same features must always produce the same
        assessment. No clock reads, no randomness, no I/O.
        """


def _pressure_penalty(pressure_sensitivity: float) -> float:
    """Points deducted from mental_readiness for high pressure sensitivity."""
    if pressure_sensitivity <= PRESSURE_PENALTY_KNEE:
        return 0.0
    return PRESSURE_PENALTY_SLOPE * (pressure_sensitivity - PRESSURE_PENALTY_KNEE)


def _classify(dimension: str, goodness: float) -> str:
    """One rule for every dimension: >= 65 positive, <= 45 negative, else
    neutral.
    """
    if goodness >= FACTOR_POSITIVE_MIN:
        return FACTOR_POSITIVE
    if goodness <= FACTOR_NEGATIVE_MAX:
        return FACTOR_NEGATIVE
    return FACTOR_NEUTRAL


def factor_phrase(dimension: str, label: str) -> str:
    """The human-readable phrase for one classified dimension."""
    positive, negative, neutral = _FACTOR_PHRASES[dimension]
    if label == FACTOR_POSITIVE:
        return positive
    if label == FACTOR_NEGATIVE:
        return negative
    return neutral


def _pressure_risk(pressure_sensitivity: float, stress_score: float) -> str:
    """Banded on a blend of stated pressure sensitivity and reported stress, so
    neither alone can drive it: a player who says pressure reduces them but
    reports no stress is genuinely not the same risk as one reporting both."""
    composite = (
        PRESSURE_RISK_WEIGHTS["sensitivity"] * pressure_sensitivity
        + PRESSURE_RISK_WEIGHTS["stress"] * stress_score
    )
    if composite >= PRESSURE_RISK_HIGH_MIN:
        return RISK_HIGH
    if composite >= PRESSURE_RISK_MODERATE_MIN:
        return RISK_MODERATE
    return RISK_LOW


def _mental_performance_risk(mental_readiness: float) -> str:
    """Banded on the headline readiness score, using the same cutoffs the
    sibling physical-readiness engine uses, so "moderate" means the same thing
    to a coach reading either tab."""
    if mental_readiness >= PERFORMANCE_RISK_LOW_MIN:
        return RISK_LOW
    if mental_readiness >= PERFORMANCE_RISK_MODERATE_MIN:
        return RISK_MODERATE
    return RISK_HIGH


def _historical_context(historical: dict | None) -> tuple[dict, list[str], str]:
    """Folds CV-derived history into (sub_scores, extra factor phrases, confidence)."""
    if not historical:
        return {}, [], CONFIDENCE_NORMAL

    sub_scores: dict = {}
    phrases: list[str] = []
    usable = 0

    for name in HISTORICAL_PROXY_NAMES:
        proxy = historical.get(name)
        if not isinstance(proxy, dict):
            continue
        value = proxy.get("value")
        confidence = proxy.get("confidence", CONFIDENCE_NORMAL)
        available = value is not None and confidence != CONFIDENCE_LOW_UPSTREAM
        sub_scores[name] = round(float(value), 2) if value is not None else None
        if not available:
            continue
        usable += 1
        phrases.append(
            f"{name.replace('_', ' ')} (observed-performance proxy): {float(value):.1f}/100"
        )

    if usable == 0:
        return sub_scores, phrases, CONFIDENCE_LOW_UPSTREAM
    return sub_scores, phrases, CONFIDENCE_NORMAL


class HeuristicReadinessModel(PsychologyReadinessModel):
    """Explainable, deterministic, arithmetic-only scoring."""

    method = METHOD_HEURISTIC

    def predict(self, features: dict, historical: dict | None = None) -> dict:
        missing = [name for name in FEATURE_NAMES if features.get(name) is None]
        if missing:
            return {
                "mental_readiness": None,
                "focus": None,
                "confidence": None,
                "stress": None,
                "pressure_risk": None,
                "mental_performance_risk": None,
                "factors": {},
                "sub_scores": {"missing_features": sorted(missing)},
                "method": self.method,
                "confidence_level": CONFIDENCE_LOW_SAMPLE,
                "sample_size": len(FEATURE_NAMES) - len(missing),
                "schema_version": SCHEMA_VERSION,
                "data_source": DATA_SOURCE,
            }

        f = features

        focus = clamp(
            FOCUS_WEIGHTS["focus"] * f["focus_score"]
            + FOCUS_WEIGHTS["mental_clarity"] * f["mental_clarity"]
        )
        confidence = clamp(
            CONFIDENCE_WEIGHTS["performance"] * f["confidence_score"]
            + CONFIDENCE_WEIGHTS["tactical"] * f["tactical_confidence"]
        )
        stress = clamp(
            STRESS_WEIGHTS["stress"] * f["stress_score"]
            + STRESS_WEIGHTS["nervousness"] * f["nervousness_score"]
        )
        motivation = clamp(
            MOTIVATION_WEIGHTS["match"] * f["motivation_score"]
            + MOTIVATION_WEIGHTS["competitive"] * f["competitive_motivation"]
        )
        error_recovery = clamp(f["error_recovery"])

        mental_readiness = clamp(
            READINESS_WEIGHTS["focus"] * focus
            + READINESS_WEIGHTS["confidence"] * confidence
            + READINESS_WEIGHTS["inverse_stress"] * invert(stress)
            + READINESS_WEIGHTS["motivation"] * motivation
            + READINESS_WEIGHTS["error_recovery"] * error_recovery
            - _pressure_penalty(f["pressure_sensitivity"])
        )

        goodness = {
            "focus": focus,
            "confidence": confidence,
            "stress": invert(stress),
            "error_recovery": error_recovery,
            "motivation": motivation,
            "pressure_response": invert(f["pressure_sensitivity"]),
        }
        factors = {
            dimension: _classify(dimension, goodness[dimension])
            for dimension in FACTOR_DIMENSIONS
        }

        historical_sub_scores, historical_phrases, historical_confidence = (
            _historical_context(historical)
        )

        sub_scores = {
            "focus": round(focus, 2),
            "confidence": round(confidence, 2),
            "stress": round(stress, 2),
            "motivation": round(motivation, 2),
            "error_recovery": round(error_recovery, 2),
            "pressure_sensitivity": round(f["pressure_sensitivity"], 2),
            "pressure_penalty": round(_pressure_penalty(f["pressure_sensitivity"]), 2),
        }
        if historical_sub_scores:
            sub_scores["historical_proxies"] = historical_sub_scores

        return {
            "mental_readiness": int(round(mental_readiness)),
            "focus": int(round(focus)),
            "confidence": int(round(confidence)),
            "stress": int(round(stress)),
            "pressure_risk": _pressure_risk(f["pressure_sensitivity"], f["stress_score"]),
            "mental_performance_risk": _mental_performance_risk(mental_readiness),
            "factors": factors,
            "sub_scores": sub_scores,
            "method": self.method,
            "confidence_level": historical_confidence,
            "sample_size": len(FEATURE_NAMES),
            "schema_version": SCHEMA_VERSION,
            "data_source": DATA_SOURCE,
            "historical_context": historical_phrases,
        }


DEFAULT_MODEL: PsychologyReadinessModel = HeuristicReadinessModel()
