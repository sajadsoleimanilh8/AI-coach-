"""
Shared scoring helper for the per-domain psychology scorers.

Exists for the same reason ai/computer_vision/tactical_analysis/utils.py does:
six sibling score.py modules need the identical weighted-average and
confidence-gating logic, and six copies of it would be six places for the
gating rule to drift.

The gating contract implemented here is the repo's existing one, matching
ai/player_intelligence/*/score.py:

  * value=None + confidence="low_sample"
        Genuinely could not be computed -- too few of the answers this score
        depends on were supplied. Never silently defaulted to 0 or to a
        midpoint guess.
  * value=<number> + confidence="low_upstream_confidence"
        Computed fine from the self-report, but the CV-derived history that
        was explicitly requested came back unusable. The value stands; only
        the label degrades. This is the same "a value with a caveat" state
        PlayerMetric already uses.
  * value=<number> + confidence="normal"
        Fully computed.

Historical data can only ever reach the third and second states -- it never
produces low_sample, because self-report alone is always a complete input
(§5: history corroborates, it does not gate).
"""

from __future__ import annotations

from ai.common.metrics import Confidence, MetricResult, metric_result
from ai.psychology_ai.constants import (
    CONFIDENCE_LOW_SAMPLE,
    CONFIDENCE_LOW_UPSTREAM,
    CONFIDENCE_NORMAL,
    METHOD_HEURISTIC,
    MIN_SAMPLE_ITEMS,
    SCHEMA_VERSION,
    clamp,
)


def historical_confidence(historical: dict | None, proxy_names: tuple[str, ...]) -> Confidence:
    """normal, unless history was asked for and every named proxy is unusable.

    `historical=None` -- the common case, no CV data for this player -- is not
    a degraded state and reports normal. A proxy counts as unusable when it has
    no value, or when the CV pipeline itself flagged it low_upstream_confidence
    (the same rule SportsMetric.is_available applies in nexus/sports/adapter.py).
    """
    if not historical:
        return CONFIDENCE_NORMAL

    relevant = [historical.get(name) for name in proxy_names]
    present = [proxy for proxy in relevant if isinstance(proxy, dict)]
    if not present:
        # History was supplied, but none of it relates to this metric. That is
        # not evidence of anything being wrong upstream.
        return CONFIDENCE_NORMAL

    usable = [
        proxy
        for proxy in present
        if proxy.get("value") is not None
        and proxy.get("confidence", CONFIDENCE_NORMAL) != CONFIDENCE_LOW_UPSTREAM
    ]
    return CONFIDENCE_NORMAL if usable else CONFIDENCE_LOW_UPSTREAM


def score_from_components(
    metric_name: str,
    components: dict[str, tuple[float | None, float]],
    *,
    historical: dict | None = None,
    proxy_names: tuple[str, ...] = (),
    extra_sub_scores: dict | None = None,
) -> MetricResult:
    """Weighted mean over whichever components are present -> PlayerMetric dict.

    `components` maps a sub-score name to (value, weight), exactly the shape
    ai/player_intelligence/decision_making_score/score.py uses. Weights are
    renormalized over the present components, so a missing one shifts weight to
    the others rather than dragging the result toward zero.
    """
    valid = {name: (value, weight) for name, (value, weight) in components.items() if value is not None}
    sample_size = len(valid)

    if sample_size < MIN_SAMPLE_ITEMS:
        value = None
        confidence = CONFIDENCE_LOW_SAMPLE
    else:
        weight_sum = sum(weight for _, weight in valid.values())
        value = clamp(sum(v * w for v, w in valid.values()) / weight_sum)
        confidence = historical_confidence(historical, proxy_names)

    sub_scores: dict = {
        name: round(component_value, 2) if component_value is not None else None
        for name, (component_value, _) in components.items()
    }
    if extra_sub_scores:
        sub_scores.update(extra_sub_scores)

    return metric_result(
        metric_name,
        round(value, 1) if value is not None else None,
        method=METHOD_HEURISTIC,
        confidence=confidence,
        sample_size=sample_size,
        sub_scores=sub_scores,
        schema_version=SCHEMA_VERSION,
    )
