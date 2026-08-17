"""
Motivation Score -- self-reported drive for this match and general competitive
drive.
"""

from __future__ import annotations

from ai.psychology_ai.constants import MOTIVATION_WEIGHTS
from ai.psychology_ai.scoring_utils import score_from_components

METRIC_NAME = "motivation_score"

PROXY_NAMES: tuple[str, ...] = ()


def score_motivation(features: dict, historical: dict | None = None) -> dict:
    """Weighted mean of match-specific and general competitive motivation."""
    return score_from_components(
        METRIC_NAME,
        {
            "match": (features.get("motivation_score"), MOTIVATION_WEIGHTS["match"]),
            "competitive": (
                features.get("competitive_motivation"),
                MOTIVATION_WEIGHTS["competitive"],
            ),
        },
        historical=historical,
        proxy_names=PROXY_NAMES,
    )
