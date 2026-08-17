"""
Mental Readiness -- the composed headline assessment, and this package's
public entry point.
"""

from __future__ import annotations

from ai.psychology_ai.confidence_score.score import score_confidence
from ai.psychology_ai.constants import (
    METHOD_HEURISTIC,
    READINESS_WEIGHTS,
    SCHEMA_VERSION,
)
from ai.psychology_ai.feature_extraction import extract_features
from ai.psychology_ai.focus_score.score import score_focus
from ai.psychology_ai.model_interface import (
    DEFAULT_MODEL,
    PsychologyReadinessModel,
)
from ai.psychology_ai.motivation_score.score import score_motivation
from ai.psychology_ai.pressure_index.score import score_pressure_index
from ai.psychology_ai.scoring_utils import score_from_components
from ai.psychology_ai.stress_analysis.score import score_stress

METRIC_NAME = "mental_readiness"

PROXY_NAMES = (
    "performance_consistency",
    "pressure_response_proxy",
    "focus_proxy",
    "confidence_proxy",
)


def score_mental_readiness(features: dict, historical: dict | None = None) -> dict:
    """PlayerMetric-shaped mental readiness, for symmetry with the sibling
    scorers and for anything that wants this metric on its own.
    """
    stress = features.get("stress_score")
    return score_from_components(
        METRIC_NAME,
        {
            "focus": (features.get("focus_score"), READINESS_WEIGHTS["focus"]),
            "confidence": (features.get("confidence_score"), READINESS_WEIGHTS["confidence"]),
            "inverse_stress": (
                None if stress is None else 100.0 - stress,
                READINESS_WEIGHTS["inverse_stress"],
            ),
            "motivation": (features.get("motivation_score"), READINESS_WEIGHTS["motivation"]),
            "error_recovery": (
                features.get("error_recovery"),
                READINESS_WEIGHTS["error_recovery"],
            ),
        },
        historical=historical,
        proxy_names=PROXY_NAMES,
    )


def score_component_metrics(features: dict, historical: dict | None = None) -> list[dict]:
    """One PlayerMetric-shaped row per domain, in report order."""
    return [
        score_focus(features, historical),
        score_confidence(features, historical),
        score_stress(features, historical),
        score_motivation(features, historical),
        score_pressure_index(features, historical),
        score_mental_readiness(features, historical),
    ]


def assess_psychology(
    responses: dict,
    model: PsychologyReadinessModel | None = None,
    historical: dict | None = None,
) -> dict:
    """Validated questionnaire -> features -> assessment. The entry point."""
    features = extract_features(responses)
    assessment = (model or DEFAULT_MODEL).predict(features, historical)

    return {
        **assessment,
        "features": features,
        "component_metrics": score_component_metrics(features, historical),
    }


__all__ = [
    "METHOD_HEURISTIC",
    "METRIC_NAME",
    "SCHEMA_VERSION",
    "assess_psychology",
    "extract_features",
    "score_component_metrics",
    "score_mental_readiness",
]
