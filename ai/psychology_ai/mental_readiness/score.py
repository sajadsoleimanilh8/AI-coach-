"""
Mental Readiness -- the composed headline assessment, and this package's
public entry point.

Follows the repo's ai/ convention (ai/<domain>/<metric_name>/score.py holds the
callable a consumer imports): backend/api/psychology.py imports
assess_psychology() directly, in-process, the same way backend/pipeline/runner.py
imports the player_intelligence scorers. No HTTP between backend/ and ai/.

    from ai.psychology_ai.mental_readiness.score import assess_psychology

Nothing in this package imports FastAPI, SQLAlchemy, or httpx, so the whole
engine runs and is testable with no server and no database.

THE PIPELINE, IN ORDER
----------------------
    questionnaire responses
        -> feature_extraction.extract_features   (raw 1-10 -> 0-100 features)
        -> PsychologyReadinessModel.predict      (features -> assessment)
        -> assessment dict

The five sibling domain scorers (focus, confidence, stress, motivation,
pressure index) run alongside the model over the same features and are
returned as `component_metrics` -- PlayerMetric-shaped rows, one per domain.
They are the per-domain audit trail behind the headline numbers, not a second
way of computing them: the headline scores come from the model, exactly once.

A self-report mental-READINESS estimate. Not emotion detection, not a
psychological or clinical assessment, not a diagnosis.
"""

from __future__ import annotations

from ai.common.metrics import MetricResult
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

# Every proxy the composed assessment can draw on -- the union of what the
# domain scorers use, since this metric sits above all of them.
PROXY_NAMES = (
    "performance_consistency",
    "pressure_response_proxy",
    "focus_proxy",
    "confidence_proxy",
)


def score_mental_readiness(features: dict, historical: dict | None = None) -> MetricResult:
    """PlayerMetric-shaped mental readiness, for symmetry with the sibling
    scorers and for anything that wants this metric on its own.

    Deliberately NOT how the API computes the headline number -- that goes
    through the swappable model interface (see assess_psychology below), which
    also applies the pressure-sensitivity penalty and the risk banding. This
    function is the plain weighted composite of the five domain sub-scores,
    which is what a per-metric row means everywhere else in this repo.
    """
    stress = features.get("stress_score")
    return score_from_components(
        METRIC_NAME,
        {
            "focus": (features.get("focus_score"), READINESS_WEIGHTS["focus"]),
            "confidence": (features.get("confidence_score"), READINESS_WEIGHTS["confidence"]),
            # Inverted to a goodness scale so it points the same way as the
            # rest of this weighted mean.
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
    """One PlayerMetric-shaped row per domain, in report order.

    Each gates independently: a domain missing its inputs reports
    value=None + low_sample without affecting the others.
    """
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
    """Validated questionnaire -> features -> assessment. The entry point.

    Deterministic end to end: the same responses always yield the same
    assessment, because every step is arithmetic over the submitted answers
    with no clock, no randomness, and no external state.

    `model` defaults to the heuristic implementation and is injectable so the
    API can swap a future trained model in at one composition point.
    `historical` is optional CV-derived corroboration; omitting it (the common
    case) is not a degraded state -- self-report alone is a complete input.

    Raises QuestionnaireValidationError if an answer is missing or out of range.
    """
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
