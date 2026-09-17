"""
Stress Analysis -- self-reported pre-match stress and nervousness.

Returns a PlayerMetric-shaped dict per the repo's
ai/<domain>/<metric_name>/score.py convention.

SCALE DIRECTION: this metric's `value` is HIGHER-IS-WORSE, unlike every other
psychology scorer here. It keeps the direction its questions have (the §4 API
contract names the field "stress", not "calm"), and the readiness model
inverts it where it needs a goodness scale. Flipping it here would make the
stored number disagree with the questionnaire it came from.

WHAT THIS IS NOT: an anxiety measure, a clinical screen, or a mental-health
indicator. It is what a player reported about feeling keyed up before a match,
on a 1-10 scale, and it must never be described in clinical terms.
"""

from __future__ import annotations

from ai.common.metrics import MetricResult
from ai.psychology_ai.constants import STRESS_WEIGHTS
from ai.psychology_ai.scoring_utils import score_from_components

METRIC_NAME = "stress_score"

# No CV-derived stress proxy exists and none is invented -- there is no
# observable on-pitch quantity that honestly stands in for reported stress.
PROXY_NAMES: tuple[str, ...] = ()


def score_stress(features: dict, historical: dict | None = None) -> MetricResult:
    """Weighted mean of general pre-match stress (itself already the mean of
    the stress and match-importance items) and reported nervousness.

    Higher values mean more reported stress.
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
