"""
Stress Analysis -- self-reported pre-match stress and nervousness.
"""

from __future__ import annotations

from ai.psychology_ai.constants import STRESS_WEIGHTS
from ai.psychology_ai.scoring_utils import score_from_components

METRIC_NAME = "stress_score"

PROXY_NAMES: tuple[str, ...] = ()


def score_stress(features: dict, historical: dict | None = None) -> dict:
    """Weighted mean of general pre-match stress (itself already the mean of
    the stress and match-importance items) and reported nervousness.
    """
    return score_from_components(
        METRIC_NAME,
        {
            "stress": (features.get("stress_score"), STRESS_WEIGHTS["stress"]),
            "nervousness": (features.get("nervousness_score"), STRESS_WEIGHTS["nervousness"]),
        },
        historical=historical,
        proxy_names=PROXY_NAMES,
        extra_sub_scores={"direction": "higher_is_worse"},
    )
