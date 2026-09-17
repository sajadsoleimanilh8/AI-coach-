"""
Constants and small pure helpers for the Pre-Match Health Intelligence
readiness engine.

Every tunable number the engine uses lives here so a reviewer can audit the
whole weighting scheme in one file, and so `schema_version` below is the one
thing that has to change when any of them does (a stored assessment computed
under different constants is not comparable to a new one -- same reasoning as
PlayerMetric.schema_version in backend/database/models.py).

Layout of this module (documented once, here, per the spec's "pick one clear
layout" requirement -- the whole engine is in this one package, nothing is
scattered across the sibling readiness_score/fatigue_prediction/
recovery_analysis/hydration_score stub folders):

    constants.py     <- you are here: bounds, targets, weights, thresholds
    questionnaire.py <- validated input dataclass (the raw self-report)
    features.py      <- questionnaire -> normalized PreMatchFeatures
    scorer.py        <- ReadinessScorer interface + HeuristicReadinessScorer
    score.py         <- score_match_readiness() facade (repo's ai/ convention)

Same house style as ai/computer_vision/tactical_analysis/constants.py, which
likewise mixes range constants with a small pure helper (pressure_level).
"""

from __future__ import annotations

# Defined once in ai/common/scoring.py; re-exported so existing imports work.
from ai.common.metrics import Method
from ai.common.scoring import clamp, invert, scale_1_to_10  # noqa: F401

SCHEMA_VERSION = "v1"

# The one place this module names its own scoring tier. Kept as a plain
# string rather than importing backend.database.models.MetricMethod because
# ai/ must not import backend/ (framework-agnostic, unit-testable in
# isolation) -- the backend router maps this string onto the real
# MetricMethod enum. Matches how every existing ai/ score.py reports
# "method": "heuristic_proxy".
METHOD_HEURISTIC: Method = "heuristic_proxy"

# Self-reported questionnaire answers are the only input. This is surfaced
# on every assessment so no consumer can mistake it for a sensor/wearable
# measurement (which is what frontend/web/src/components/TabHealthProxies.jsx
# is about -- a genuinely different, currently-unbuilt data source).
DATA_SOURCE = "self_reported"

# ---------------------------------------------------------------------------
# Questionnaire bounds
#
# Single source of truth: backend/api/schemas.py imports these for its
# Pydantic Field(ge=/le=) constraints (the ones that produce the 422), and
# questionnaire.py validates against them for callers using the ai/ engine
# directly. Declaring them once means the API's rejection rule and the
# engine's rejection rule cannot drift apart.
# ---------------------------------------------------------------------------

SCALE_MIN = 1   # every 1-10 self-rating: 1 = worst/lowest
SCALE_MAX = 10  # 10 = best/highest

SLEEP_HOURS_MIN_EXCLUSIVE = 0.0  # must be > 0: "zero hours of sleep" is not a
                                 # value this questionnaire can represent
                                 # honestly, and 0 would silently sail through
                                 # a ge=0 check as a real reading
SLEEP_HOURS_MAX = 24.0

# ---------------------------------------------------------------------------
# Normalization targets -- what counts as "100" for each 0-100 sub-score
# ---------------------------------------------------------------------------

# Sleep duration scales linearly between a floor and a target, capping at
# both ends: <= 3h -> 0, 8h -> 100, and anything above the target stays 100
# (capped, not extrapolated -- there is no basis in a single self-report for
# treating 11h as better than 8h).
#
# The floor is why this is not just `hours / target`. Scaling linearly from
# zero rates 4.5h of sleep at 56/100, which then reads as "adequate" once
# it is composited with quality and awakenings -- so a player on four and a
# half broken hours came out neutral rather than negative, which is not a
# defensible thing to tell a coach before a match. Anchoring 0 at 3h makes
# the deficit range where it actually matters (3-8h) use the full scale.
SLEEP_TARGET_HOURS = 8.0
SLEEP_FLOOR_HOURS = 3.0

# Each reported night awakening costs this much off the awakenings
# sub-score, floored at 0 (so 7+ awakenings all read as "0", rather than
# going negative and swamping the rest of the sleep composite).
AWAKENING_PENALTY_PER_EVENT = 15.0

# Hydration: litres consumed against a general daily target. Capped at the
# target -- drinking past it is not scored as increasingly better.
HYDRATION_TARGET_LITERS = 3.0

# A training session at or beyond this duration contributes a full
# duration-component to recent training load.
TRAINING_DURATION_FULL_LOAD_MIN = 120.0

# Recent training load decays linearly to zero this many hours after the
# last session, when hours_since_last_training is supplied.
TRAINING_RECENCY_DECAY_HOURS = 72.0

# Residual weight still carried by a session that happened in the 24-48h
# window rather than the last 24h.
TRAINING_RECENCY_48H_FACTOR = 0.6

# Flat addition (before recency scaling) for a session flagged as
# containing sprint/high-intensity work.
HIGH_INTENSITY_LOAD_BONUS = 15.0

# Meal timing: eating inside this window before a match is treated as
# neutral-to-ideal; either side of it is scored down.
MEAL_TIMING_IDEAL_MIN_HOURS = 1.5
MEAL_TIMING_IDEAL_MAX_HOURS = 4.0
MEAL_TIMING_RECENT_FLOOR = 60.0        # score at hours_since_last_meal == 0
MEAL_TIMING_LATE_PENALTY_PER_HOUR = 12.5  # per hour beyond the ideal window

# ---------------------------------------------------------------------------
# Composite weights (each dict sums to 1.0 -- asserted in the tests)
# ---------------------------------------------------------------------------

# Sleep composite. Duration-dominant on purpose: the spec's fatigue formula
# calls for a "sleep_debt_component that grows as sleep_duration_hours falls
# below ~8h", and this composite is what supplies it (as 100 - sleep_score).
# Quality and awakenings are folded in at lower weight rather than left out
# so that two genuinely different nights -- 8h of broken, poor-quality sleep
# vs. 8h of unbroken good sleep -- do not produce an identical fatigue
# score. Without this, sleep_quality and night_awakenings would be collected
# from the player and then never affect any number, which is exactly the
# kind of decorative input the "every number traces back to a field" rule
# exists to prevent.
SLEEP_WEIGHTS = {"duration": 0.55, "quality": 0.35, "awakenings": 0.10}

# Session load: how hard the last session itself was, before recency.
SESSION_LOAD_WEIGHTS = {"duration": 0.45, "intensity": 0.55}

# Recovery (spec formula, unmodified).
RECOVERY_WEIGHTS = {
    "fatigue": 0.30,
    "muscle_soreness": 0.25,
    "pain": 0.25,
    "perceived_readiness": 0.20,
}

# Fatigue (spec formula, unmodified).
FATIGUE_WEIGHTS = {
    "self_reported_fatigue": 0.40,
    "recent_training_load": 0.35,
    "sleep_debt": 0.15,
    "high_intensity": 0.10,
}

# Nutrition: what was eaten (self-rated quality) dominates; when it was
# eaten is a smaller timing correction.
NUTRITION_WEIGHTS = {"quality": 0.75, "timing": 0.25}

# Physical readiness (spec formula, unmodified). Weights sum to 1.0 before
# the training-load penalty is subtracted.
READINESS_WEIGHTS = {
    "recovery": 0.30,
    "inverse_fatigue": 0.25,
    "hydration": 0.15,
    "nutrition": 0.10,
    "perceived_readiness": 0.20,
}

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Training-load penalty applied to physical_readiness. Nothing is deducted
# below the knee point; past it the deduction grows linearly, reaching
# (100 - knee) * slope = 10 points at a maxed-out load of 100.
TRAINING_LOAD_PENALTY_KNEE = 60.0
TRAINING_LOAD_PENALTY_SLOPE = 0.25

# performance_risk bands on physical_readiness.
PERFORMANCE_RISK_LOW_MIN = 70.0       # >= 70 -> "low"
PERFORMANCE_RISK_MODERATE_MIN = 45.0  # 45-69 -> "moderate", < 45 -> "high"

# workload_risk bands on recent_training_load.
WORKLOAD_RISK_HIGH_LOAD_MIN = 70.0
WORKLOAD_RISK_LOW_LOAD_MAX = 40.0
WORKLOAD_RISK_INTENSITY_MIN = 8  # with trained_last_24h + high_intensity_activity

# Factor classification thresholds, applied to each dimension's 0-100
# "goodness" sub-score (load/fatigue dimensions are inverted to goodness
# first, so the same two thresholds work for every dimension).
FACTOR_POSITIVE_MIN = 65.0
FACTOR_NEGATIVE_MAX = 45.0

RISK_LOW = "low"
RISK_MODERATE = "moderate"
RISK_HIGH = "high"

FACTOR_POSITIVE = "positive"
FACTOR_NEGATIVE = "negative"
FACTOR_NEUTRAL = "neutral"


