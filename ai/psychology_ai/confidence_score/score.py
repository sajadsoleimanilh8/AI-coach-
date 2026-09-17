"""
Confidence Score -- self-reported belief in performing and in executing the
tactical brief.

Returns a PlayerMetric-shaped dict per the repo's
ai/<domain>/<metric_name>/score.py convention.

A confidence PROXY from a self-report. Not a personality measure and not a
psychological assessment.
"""

from __future__ import annotations

from ai.common.metrics import MetricResult
from ai.psychology_ai.constants import CONFIDENCE_WEIGHTS
from ai.psychology_ai.scoring_utils import score_from_components

METRIC_NAME = "confidence_score"

PROXY_NAMES = ("confidence_proxy",)


def score_confidence(features: dict, historical: dict | None = None) -> MetricResult:
    """Weighted mean of general performance confidence and the more specific
    confidence in carrying out tactical responsibilities."""
    return score_from_components(
        METRIC_NAME,
        {
            "performance": (features.get("confidence_score"), CONFIDENCE_WEIGHTS["performance"]),
            "tactical": (features.get("tactical_confidence"), CONFIDENCE_WEIGHTS["tactical"]),
        },
        historical=historical,
        proxy_names=PROXY_NAMES,
    )
