"""
Constants and small pure helpers for the Pre-Match Psychology Intelligence
mental-readiness engine.

Every tunable number this engine uses lives here, so the whole weighting
scheme can be audited in one file and so `SCHEMA_VERSION` below is the single
thing that has to change when any of them does (an assessment computed under
different constants is not comparable to a new one -- the same reasoning as
PlayerMetric.schema_version in backend/database/models.py).

WHAT THIS IS AND IS NOT
-----------------------
This is a self-report mental-READINESS estimation. It is not emotion
detection, not a psychological or clinical assessment, and not a diagnosis.
Nothing here observes a player; every number traces back to an answer the
player typed on a 13-item form about how ready they feel. The vocabulary is
deliberately constrained to performance language ("mental readiness",
"pressure response", "focus proxy") and never clinical language -- see
FACTOR_PHRASES in model_interface.py, which is the only place this engine
puts words to a number.

Layout of this package (documented once, here, mirroring the sibling
ai/performance_ai/match_readiness_predictor/ engine):

    constants.py          <- you are here: bounds, weights, thresholds
    feature_extraction.py <- raw 1-10 answers -> normalized 0-100 features
    model_interface.py    <- PsychologyReadinessModel ABC + heuristic impl
    <metric>/score.py     <- one PlayerMetric-shaped scorer per domain, per
                             the repo's ai/<domain>/<metric_name>/score.py
                             convention

Note ai/psychology_ai/emotional_state_analysis/ is deliberately left empty.
This module must never claim to detect emotion, so that directory stays a
.gitkeep rather than becoming a feature.
"""

from __future__ import annotations

# Defined once in ai/common/scoring.py; re-exported so existing imports work.
from ai.common.metrics import Confidence, Method
from ai.common.scoring import clamp, invert, scale_1_to_10  # noqa: F401

SCHEMA_VERSION = "v1"

# This module names its own scoring tier as a plain string rather than
# importing backend.database.models.MetricMethod, because ai/ must not import
# backend/ (it has to stay framework-agnostic and unit-testable in isolation).
# The backend router maps this string onto the real MetricMethod enum. Matches
# how every existing ai/ score.py reports "method": "heuristic_proxy".
METHOD_HEURISTIC: Method = "heuristic_proxy"

# Mirrors MetricConfidence in backend/database/models.py, again as plain
# strings for the same reason. The backend maps them onto the enum.
CONFIDENCE_NORMAL: Confidence = "normal"
CONFIDENCE_LOW_SAMPLE: Confidence = "low_sample"
CONFIDENCE_LOW_UPSTREAM: Confidence = "low_upstream_confidence"

# Self-reported answers are the only input that moves a number here. Surfaced
# on every assessment so no consumer mistakes these for observed or sensor-
# derived measurements.
DATA_SOURCE = "self_reported"

# ---------------------------------------------------------------------------
# Questionnaire bounds
#
# Single source of truth: backend/api/schemas.py imports these for its
# Pydantic Field(ge=/le=) constraints (the ones that produce the 422), and
# feature_extraction.py validates against them for callers using the ai/
# engine directly. Declaring them once means the API's rejection rule and the
# engine's rejection rule cannot drift apart.
# ---------------------------------------------------------------------------

SCALE_MIN = 1   # every 1-10 self-rating: 1 = lowest
SCALE_MAX = 10  # 10 = highest

# Item 12 is categorical, not a scale: does high pressure usually improve or
# reduce this player's performance?
PRESSURE_EFFECT_IMPROVES = "improves"
PRESSURE_EFFECT_NO_CHANGE = "no_change"
PRESSURE_EFFECT_REDUCES = "reduces"
PRESSURE_EFFECT_VALUES = (
    PRESSURE_EFFECT_IMPROVES,
    PRESSURE_EFFECT_NO_CHANGE,
    PRESSURE_EFFECT_REDUCES,
)

# The 13 questionnaire items, grouped as the product spec groups them. Used by
# feature_extraction.py for validation and by the tests to assert the form and
# the engine agree on the item set.
SCALE_ITEMS = (
    # focus
    "concentration_level",
    "focus_maintenance",
    "mental_clarity_raw",
    # stress
    "pre_match_stress",
    "importance_pressure",
    "nervousness",
    # confidence
    "performance_confidence",
    "tactical_confidence_raw",
    # motivation
    "match_motivation",
    "competitive_motivation_raw",
    # pressure response
    "mistake_recovery_speed",
    "post_error_calm",
)
CATEGORICAL_ITEMS = ("pressure_performance_effect",)
QUESTIONNAIRE_ITEMS = SCALE_ITEMS + CATEGORICAL_ITEMS

# ---------------------------------------------------------------------------
# pressure_sensitivity
#
# The one derived feature that is not a straight scaling of its inputs, so its
# formula is stated in full here and again in feature_extraction.py:
#
#     pressure_sensitivity =
#         PRESSURE_EFFECT_BASE[pressure_performance_effect] * PRESSURE_EFFECT_WEIGHT
#       + (importance_pressure * 10)                        * PRESSURE_CONTEXT_WEIGHT
#
# Higher means MORE sensitive to pressure (i.e. worse). The categorical answer
# dominates because it is the player's own read on how pressure affects them;
# match importance is the situational context that decides how much that
# disposition is about to be tested. A player who says pressure improves them,
# before a low-stakes match, lands near 16; one who says it reduces them,
# before a high-stakes match, lands near 87.
# ---------------------------------------------------------------------------

PRESSURE_EFFECT_BASE = {
    PRESSURE_EFFECT_IMPROVES: 20.0,    # thrives under pressure -> low sensitivity
    PRESSURE_EFFECT_NO_CHANGE: 50.0,   # neutral -> mid scale
    PRESSURE_EFFECT_REDUCES: 80.0,     # performance drops -> high sensitivity
}
PRESSURE_EFFECT_WEIGHT = 0.65
PRESSURE_CONTEXT_WEIGHT = 0.35

# ---------------------------------------------------------------------------
# Composite weights (each dict sums to 1.0 -- asserted in the tests)
# ---------------------------------------------------------------------------

# Headline focus. Sustained attention over the match is weighted above the
# snapshot of clarity right now, because the score is about the 90 minutes
# ahead rather than the moment the form was filled in.
FOCUS_WEIGHTS = {"focus": 0.60, "mental_clarity": 0.40}

# Headline confidence. Belief in one's own performance leads; confidence in
# executing the tactical brief is the second, more specific component.
CONFIDENCE_WEIGHTS = {"performance": 0.60, "tactical": 0.40}

# Headline stress. HIGHER IS WORSE, unlike every other headline score here --
# the API contract in §4 names it "stress", not "calm", so it keeps the
# direction its questions have and the readiness formula inverts it.
STRESS_WEIGHTS = {"stress": 0.60, "nervousness": 0.40}

# Motivation composite (not a headline score, but a readiness input).
MOTIVATION_WEIGHTS = {"match": 0.55, "competitive": 0.45}

# Mental readiness: the headline aggregate. Focus and confidence lead because
# they are the two dimensions most directly about executing the next match;
# stress enters inverted; error recovery and motivation are real but smaller
# contributors. Sums to 1.0 before the pressure-sensitivity penalty.
READINESS_WEIGHTS = {
    "focus": 0.25,
    "confidence": 0.25,
    "inverse_stress": 0.20,
    "motivation": 0.15,
    "error_recovery": 0.15,
}

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Pressure-sensitivity penalty applied to mental_readiness, shaped exactly
# like _training_load_penalty in the sibling readiness engine: nothing is
# deducted below the knee, then linear, reaching (100 - knee) * slope = 10
# points at a maxed-out sensitivity of 100.
#
# This penalty is why it exists at all: without it, pressure_performance_effect
# (item 12) would move pressure_risk and nothing else, making it a collected
# answer that never reaches the headline number. That is the kind of
# decorative input the "every score traces back to a specific input" rule
# exists to prevent. It is deliberately gentle -- match importance is already
# inside stress_score, and double-counting it at full weight would let one
# dimension dominate.
PRESSURE_PENALTY_KNEE = 60.0
PRESSURE_PENALTY_SLOPE = 0.25

# pressure_risk bands. Computed on a blend of how sensitive the player says
# they are to pressure and how much stress they are actually reporting, so
# neither alone can drive it: a player who says pressure reduces them but
# reports no stress is not the same risk as one reporting both.
PRESSURE_RISK_WEIGHTS = {"sensitivity": 0.60, "stress": 0.40}
PRESSURE_RISK_HIGH_MIN = 65.0      # >= 65 -> "high"
PRESSURE_RISK_MODERATE_MIN = 45.0  # 45-64 -> "moderate", < 45 -> "low"

# mental_performance_risk bands on mental_readiness. Same cutoffs the sibling
# engine uses for performance_risk on physical_readiness, so a coach reading
# both tabs reads "moderate" the same way in each.
PERFORMANCE_RISK_LOW_MIN = 70.0       # >= 70 -> "low"
PERFORMANCE_RISK_MODERATE_MIN = 45.0  # 45-69 -> "moderate", < 45 -> "high"

# Factor classification thresholds, applied to each dimension's 0-100
# "goodness" sub-score (stress and pressure sensitivity are inverted to
# goodness first, so one pair of thresholds covers every dimension).
#
# Same values as the sibling engine's FACTOR_POSITIVE_MIN/FACTOR_NEGATIVE_MAX,
# and the reason the §4 contract can say factors are "derived from the same
# thresholds used for the headline scores": PRESSURE_RISK_MODERATE_MIN and
# PERFORMANCE_RISK_MODERATE_MIN are both 45 too, so "negative factor" and
# "not low risk" cannot disagree about where the line is.
FACTOR_POSITIVE_MIN = 65.0
FACTOR_NEGATIVE_MAX = 45.0

RISK_LOW = "low"
RISK_MODERATE = "moderate"
RISK_HIGH = "high"

FACTOR_POSITIVE = "positive"
FACTOR_NEGATIVE = "negative"
FACTOR_NEUTRAL = "neutral"

# Minimum number of answered items a per-domain scorer needs before it will
# emit a value. Every domain here is backed by 2-3 questionnaire items and the
# API always submits all 13, so this bites only for a caller using the ai/
# engine directly with a partial feature dict -- which is exactly when a
# silently-computed score would be most misleading.
MIN_SAMPLE_ITEMS = 2


def mean(*values: float) -> float:
    """Arithmetic mean. Present so the composite formulas below read the way
    the spec states them (`mean(a, b) * 10`) rather than as inlined division."""
    return float(sum(values)) / len(values)
