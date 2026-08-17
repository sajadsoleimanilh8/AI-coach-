"""
Confidence Score -- self-reported belief in performing and in executing the
tactical brief.
"""

from __future__ import annotations

from ai.psychology_ai.constants import CONFIDENCE_WEIGHTS
from ai.psychology_ai.scoring_utils import score_from_components

METRIC_NAME = "confidence_score"

PROXY_NAMES = ("confidence_proxy",)


def score_confidence(features: dict, historical: dict | None = None) -> dict:
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
