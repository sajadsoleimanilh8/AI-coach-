"""
Pre-Match Health Intelligence -- public entry point.

Follows the repo's ai/ convention (ai/<domain>/<metric_name>/score.py holds
the callable a consumer imports): backend/api/prematch_health.py imports
score_match_readiness() directly, in-process, the same way
backend/pipeline/runner.py imports the player_intelligence scorers. No HTTP
between backend/ and ai/.

    from ai.performance_ai.match_readiness_predictor.score import (
        score_match_readiness,
    )

Nothing in this package imports FastAPI, SQLAlchemy, or httpx, so the whole
engine runs and is testable with no server and no database.
"""

from __future__ import annotations

from dataclasses import replace

from ai.performance_ai.match_readiness_predictor.features import (
    PreMatchFeatures,
    extract_features,
)
from ai.performance_ai.match_readiness_predictor.questionnaire import (
    PreMatchQuestionnaireInput,
)
from ai.performance_ai.match_readiness_predictor.scorer import (
    HealthAssessment,
    HeuristicReadinessScorer,
    ReadinessScorer,
)

# The implementation in use today. Swapping in an MLReadinessScorer later is
# a change here (or an argument at the call site) and nowhere else --
# backend/api/prematch_health.py depends on the ReadinessScorer interface.
DEFAULT_SCORER: ReadinessScorer = HeuristicReadinessScorer()


def score_match_readiness(
    questionnaire: PreMatchQuestionnaireInput,
    scorer: ReadinessScorer | None = None,
) -> HealthAssessment:
    """Validated questionnaire -> features -> assessment.

    Deterministic end to end: the same questionnaire always yields the same
    assessment, because every step is arithmetic over the submitted answers
    with no clock, no randomness, and no external state.

    `caffeine_or_supplement_notes` is attached to the result here, AFTER
    scoring, and is never handed to the scorer -- so free text cannot reach
    any formula even by accident.
    """
    features = extract_features(questionnaire)
    assessment = (scorer or DEFAULT_SCORER).score(features)
    return replace(assessment, notes=questionnaire.caffeine_or_supplement_notes)


__all__ = [
    "DEFAULT_SCORER",
    "HealthAssessment",
    "HeuristicReadinessScorer",
    "PreMatchFeatures",
    "PreMatchQuestionnaireInput",
    "ReadinessScorer",
    "extract_features",
    "score_match_readiness",
]
