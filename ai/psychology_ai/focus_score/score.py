"""
Focus Score -- self-reported attention readiness for the upcoming match.
"""

from __future__ import annotations

from ai.psychology_ai.constants import FOCUS_WEIGHTS
from ai.psychology_ai.scoring_utils import score_from_components

METRIC_NAME = "focus_score"

PROXY_NAMES = ("focus_proxy",)


def score_focus(features: dict, historical: dict | None = None) -> dict:
    """Weighted mean of sustained focus and current mental clarity."""
    return score_from_components(
        METRIC_NAME,
        {
            "focus": (features.get("focus_score"), FOCUS_WEIGHTS["focus"]),
            "mental_clarity": (features.get("mental_clarity"), FOCUS_WEIGHTS["mental_clarity"]),
        },
        historical=historical,
        proxy_names=PROXY_NAMES,
    )
