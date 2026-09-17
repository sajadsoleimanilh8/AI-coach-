"""
The intelligence layer: a model interface plus the one implementation that
actually exists today.

WHY AN INTERFACE FOR A SINGLE IMPLEMENTATION
--------------------------------------------
No trained mental-readiness model exists anywhere in this repo, and none can
until labelled data is collected (self-reported readiness paired with actual
match performance -- see training/train_psychology.py, which says so and
trains nothing). Inventing one -- a "model" that is really a table of
hardcoded numbers -- would be worse than useless, because it would look
trustworthy. So the heuristic is the honest implementation, disclosed as such
via `method="heuristic_proxy"` on every row it produces, and this interface is
the seam a real model plugs into later without the API contract moving.

`models/xgboost/` and `models/lstm/` are reserved in this repo for exactly
that. An XGBoostReadinessModel implements the same `predict(features,
historical)` signature, consumes the identical 0-100 feature vector (which is
why the features are normalized and bounded), and returns the identical
assessment dict -- differing only in `method`, which becomes
MetricMethod.ml_trained. Callers reading `method` off the response is how a
consumer tells which tier produced a given number; nothing else changes.

WHAT THIS IS NOT
----------------
Mental-READINESS estimation from a self-report. Not emotion detection, not a
psychological or clinical assessment, not a diagnosis. Every phrase in
_FACTOR_PHRASES below is performance language, and that is the only place this
engine puts words to a number.
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

# (positive phrase, negative phrase, neutral phrase) per dimension.
#
# Deliberately plain performance language about what the player reported --
# never clinical or diagnostic. "elevated pre-match stress", never "anxiety";
# "pressure response" and "focus proxy", never "emotional state". This table is
# the single place a number becomes a phrase, so the vocabulary rule is
# enforceable by reading one dict.
_FACTOR_PHRASES: dict[str, tuple[str, str, str]] = {
    "focus": ("high focus", "reduced focus", "moderate focus"),
    "confidence": ("strong confidence", "low confidence", "moderate confidence"),
    # Named for the goodness direction: a "positive" stress factor means LOW
    # reported stress. The phrases spell that out so the label cannot be
    # misread on its own.
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

# The dimensions classified into `factors`, in report order. All six genuinely
# contribute to mental_readiness (focus, confidence, stress, motivation and
# error_recovery through READINESS_WEIGHTS; pressure_response through the
# sensitivity penalty), which is the rule for appearing here -- a dimension
# that moved no number would be decorative.
FACTOR_DIMENSIONS = (
    "focus",
    "confidence",
    "stress",
    "error_recovery",
    "motivation",
    "pressure_response",
)

# Historical CV-derived proxies this model will fold in when they are supplied.
# The names are fixed and deliberately performance-flavoured (§5): these are
# proxies for observable on-pitch behaviour, never measurements of a mental
# state. Anything not in this set is ignored rather than guessed at.
HISTORICAL_PROXY_NAMES = (
    "performance_consistency",
    "pressure_response_proxy",
    "focus_proxy",
    "confidence_proxy",
)


class PsychologyReadinessModel(ABC):
    """Strategy interface between feature extraction and the API/DB layer.

    backend/api/psychology.py depends on THIS, never on a concrete subclass,
    so swapping the implementation is a one-line change at the composition
    point and touches no endpoint code.
    """

    @abstractmethod
    def predict(self, features: dict, historical: dict | None = None) -> dict:
        """Deterministic: the same features must always produce the same
        assessment. No clock reads, no randomness, no I/O.

        `historical` is optional corroborating context derived from existing
        PlayerMetric rows. It may add sub-scores and labelled proxy factors,
        and it may lower the reported confidence -- but it must never be
        required, and must never change a headline score. Self-report alone is
        always a complete input.
        """


def _pressure_penalty(pressure_sensitivity: float) -> float:
    """Points deducted from mental_readiness for high pressure sensitivity.

    Zero below the knee, then linear, reaching 10 points at a maxed-out
    sensitivity of 100. Shaped exactly like _training_load_penalty in the
    sibling readiness engine, and gentle for the same reason: match importance
    already sits inside stress_score, so charging it again at full weight would
    let one dimension dominate the headline number.
    """
    if pressure_sensitivity <= PRESSURE_PENALTY_KNEE:
        return 0.0
    return PRESSURE_PENALTY_SLOPE * (pressure_sensitivity - PRESSURE_PENALTY_KNEE)


def _classify(dimension: str, goodness: float) -> str:
    """One rule for every dimension: >= 65 positive, <= 45 negative, else
    neutral.

    `goodness` is always higher-is-better, whatever direction the underlying
    feature had -- callers invert stress and pressure sensitivity before
    calling. That is what lets one pair of thresholds cover all six dimensions,
    and why `factors` cannot disagree with the headline scores: these are the
    same cutoffs the risk bands use.
    """
    if goodness >= FACTOR_POSITIVE_MIN:
        return FACTOR_POSITIVE
    if goodness <= FACTOR_NEGATIVE_MAX:
        return FACTOR_NEGATIVE
    return FACTOR_NEUTRAL


def factor_phrase(dimension: str, label: str) -> str:
    """The human-readable phrase for one classified dimension.

    Exported because nexus/sports/psychology.py turns the same factors dict
    into coach-facing prose and must use the identical wording -- one
    vocabulary, defined once.
    """
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
    """Folds CV-derived history into (sub_scores, extra factor phrases, confidence).

    Returns a confidence of low_upstream_confidence when history was explicitly
    asked for -- the caller supplied a cv_player_id and got rows back -- but
    every proxy in it turned out unusable. The self-report score is still
    computed and returned in that case; only the label degrades. History is
    corroboration, so its absence or poor quality can never zero out a score
    the player's own answers fully support.

    Passing None (the common case -- no CV data for this player) is not a
    degraded state at all and reports normal confidence.
    """
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
        # Mirrors SportsMetric.is_available in nexus/sports/adapter.py: a
        # metric with no value, or one the CV pipeline itself flagged as
        # low-upstream, is not corroboration.
        available = value is not None and confidence != CONFIDENCE_LOW_UPSTREAM
        sub_scores[name] = round(float(value), 2) if value is not None else None
        if not available:
            continue
        usable += 1
        # Explicitly labelled as an observed-performance proxy so it can never
        # read as a measurement of a mental state.
        phrases.append(
            f"{name.replace('_', ' ')} (observed-performance proxy): {float(value):.1f}/100"
        )

    if usable == 0:
        return sub_scores, phrases, CONFIDENCE_LOW_UPSTREAM
    return sub_scores, phrases, CONFIDENCE_NORMAL


class HeuristicReadinessModel(PsychologyReadinessModel):
    """Explainable, deterministic, arithmetic-only scoring.

    Every weight and threshold it uses is a named constant in constants.py, and
    every number it emits traces back through one of the formulas below to a
    questionnaire item. There is no lookup table, no per-player special case,
    no stored state and no randomness -- two identical questionnaires from two
    different players produce identical assessments, by construction.
    """

    method = METHOD_HEURISTIC

    def predict(self, features: dict, historical: dict | None = None) -> dict:
        missing = [name for name in FEATURE_NAMES if features.get(name) is None]
        if missing:
            # Genuinely could not be computed -- reported as such rather than
            # defaulted to 0 or to a midpoint guess, per the repo's
            # value=None + low_sample contract.
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
                # NOT "confidence": that key is the §4 contract's 0-100
                # confidence SCORE. This is the MetricConfidence tier, and it
                # is named for the column it lands in
                # (PsychologyAssessment.confidence_level) so the two can never
                # collide in one flat dict.
                "confidence_level": CONFIDENCE_LOW_SAMPLE,
                "sample_size": len(FEATURE_NAMES) - len(missing),
                "schema_version": SCHEMA_VERSION,
                "data_source": DATA_SOURCE,
            }

        f = features

        # --- headline components ------------------------------------------
        focus = clamp(
            FOCUS_WEIGHTS["focus"] * f["focus_score"]
            + FOCUS_WEIGHTS["mental_clarity"] * f["mental_clarity"]
        )
        confidence = clamp(
            CONFIDENCE_WEIGHTS["performance"] * f["confidence_score"]
            + CONFIDENCE_WEIGHTS["tactical"] * f["tactical_confidence"]
        )
        # Higher is worse, unlike the other three headline scores.
        stress = clamp(
            STRESS_WEIGHTS["stress"] * f["stress_score"]
            + STRESS_WEIGHTS["nervousness"] * f["nervousness_score"]
        )
        motivation = clamp(
            MOTIVATION_WEIGHTS["match"] * f["motivation_score"]
            + MOTIVATION_WEIGHTS["competitive"] * f["competitive_motivation"]
        )
        error_recovery = clamp(f["error_recovery"])

        # --- headline aggregate -------------------------------------------
        mental_readiness = clamp(
            READINESS_WEIGHTS["focus"] * focus
            + READINESS_WEIGHTS["confidence"] * confidence
            + READINESS_WEIGHTS["inverse_stress"] * invert(stress)
            + READINESS_WEIGHTS["motivation"] * motivation
            + READINESS_WEIGHTS["error_recovery"] * error_recovery
            - _pressure_penalty(f["pressure_sensitivity"])
        )

        # --- classification -----------------------------------------------
        # Every dimension put on a common higher-is-better scale first, so one
        # threshold pair covers them all.
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
            # The intermediates behind the headline numbers, kept rather than
            # discarded so any score stays traceable to the answers.
            "focus": round(focus, 2),
            "confidence": round(confidence, 2),
            "stress": round(stress, 2),
            "motivation": round(motivation, 2),
            "error_recovery": round(error_recovery, 2),
            "pressure_sensitivity": round(f["pressure_sensitivity"], 2),
            "pressure_penalty": round(_pressure_penalty(f["pressure_sensitivity"]), 2),
        }
        if historical_sub_scores:
            # Namespaced so a CV-derived proxy can never be confused with a
            # self-reported sub-score.
            sub_scores["historical_proxies"] = historical_sub_scores

        return {
            # The §4 contract. Rounded to ints -- these are estimates from
            # 1-10 self-ratings, and decimal places would imply a precision the
            # input does not have.
            "mental_readiness": int(round(mental_readiness)),
            "focus": int(round(focus)),
            "confidence": int(round(confidence)),
            "stress": int(round(stress)),
            "pressure_risk": _pressure_risk(f["pressure_sensitivity"], f["stress_score"]),
            "mental_performance_risk": _mental_performance_risk(mental_readiness),
            "factors": factors,
            # Additive metadata, for the persistence contract in §0/§6.
            "sub_scores": sub_scores,
            "method": self.method,
            # See the note on the early return above: the metric-confidence
            # tier is `confidence_level`, never `confidence`, which the §4
            # contract reserves for the 0-100 confidence score.
            "confidence_level": historical_confidence,
            "sample_size": len(FEATURE_NAMES),
            "schema_version": SCHEMA_VERSION,
            "data_source": DATA_SOURCE,
            "historical_context": historical_phrases,
        }


# The implementation in use today. Swapping in a trained model later is a
# change here (or an argument at the call site) and nowhere else -- every
# caller depends on the PsychologyReadinessModel interface.
DEFAULT_MODEL: PsychologyReadinessModel = HeuristicReadinessModel()
