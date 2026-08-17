"""
Pre-Match Health Intelligence -- public entry point.
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

DEFAULT_SCORER: ReadinessScorer = HeuristicReadinessScorer()


def score_match_readiness(
    questionnaire: PreMatchQuestionnaireInput,
    scorer: ReadinessScorer | None = None,
) -> HealthAssessment:
    """Validated questionnaire -> features -> assessment."""
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
