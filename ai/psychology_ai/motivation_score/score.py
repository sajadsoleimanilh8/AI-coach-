"""
Motivation Score -- self-reported drive for this match and general competitive
drive.

Returns a PlayerMetric-shaped dict per the repo's
ai/<domain>/<metric_name>/score.py convention.

New sibling directory alongside the scaffolded psychology metrics: the
questionnaire covers motivation, so it gets its own scorer rather than being
folded invisibly into another domain's sub-scores.
"""

from __future__ import annotations

from ai.common.metrics import MetricResult
from ai.psychology_ai.constants import MOTIVATION_WEIGHTS
from ai.psychology_ai.scoring_utils import score_from_components

METRIC_NAME = "motivation_score"

# No CV-derived motivation proxy exists, and none is invented: there is no
# observable on-pitch quantity in this repo that honestly stands in for
# motivation. This scorer is self-report only, which is why PROXY_NAMES is
# empty rather than pointing at a loosely-related running metric.
PROXY_NAMES: tuple[str, ...] = ()


def score_motivation(features: dict, historical: dict | None = None) -> MetricResult:
    """Weighted mean of match-specific and general competitive motivation.

    `historical` is accepted for interface symmetry with the sibling scorers
    but cannot change this result -- see PROXY_NAMES above.
    """
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
