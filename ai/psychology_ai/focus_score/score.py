"""
Focus Score -- self-reported attention readiness for the upcoming match.

Follows the repo's ai/<domain>/<metric_name>/score.py convention and returns a
PlayerMetric-shaped dict, the same shape
ai/player_intelligence/decision_making_score/score.py returns.

A focus PROXY from a self-report, not a measurement of attention. Nothing here
observes the player.
"""

from __future__ import annotations

from ai.common.metrics import MetricResult
from ai.psychology_ai.constants import FOCUS_WEIGHTS
from ai.psychology_ai.scoring_utils import score_from_components

METRIC_NAME = "focus_score"

# The CV-derived proxy that corroborates this one, when history is supplied.
# scanning_behavior_score is the closest observable analogue to "was this
# player attending to what was around them", which is why it is the focus
# proxy rather than, say, a passing metric.
PROXY_NAMES = ("focus_proxy",)


def score_focus(features: dict, historical: dict | None = None) -> MetricResult:
    """Weighted mean of sustained focus and current mental clarity.

    Gates to value=None + low_sample when fewer than two of its inputs are
    present, and to low_upstream_confidence when a requested focus proxy came
    back unusable (the value still stands -- history only corroborates).
    """
    return score_from_components(
        METRIC_NAME,
        {
            "focus": (features.get("focus_score"), FOCUS_WEIGHTS["focus"]),
            "mental_clarity": (features.get("mental_clarity"), FOCUS_WEIGHTS["mental_clarity"]),
        },
        historical=historical,
        proxy_names=PROXY_NAMES,
    )
