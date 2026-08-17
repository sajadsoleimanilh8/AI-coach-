"""
Shared scoring helper for the per-domain psychology scorers.
"""

from __future__ import annotations

from ai.psychology_ai.constants import (
    CONFIDENCE_LOW_SAMPLE,
    CONFIDENCE_LOW_UPSTREAM,
    CONFIDENCE_NORMAL,
    METHOD_HEURISTIC,
    MIN_SAMPLE_ITEMS,
    SCHEMA_VERSION,
    clamp,
)


def historical_confidence(historical: dict | None, proxy_names: tuple[str, ...]) -> str:
    """normal, unless history was asked for and every named proxy is unusable."""
    if not historical:
        return CONFIDENCE_NORMAL

    relevant = [historical.get(name) for name in proxy_names]
    present = [proxy for proxy in relevant if isinstance(proxy, dict)]
    if not present:
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
) -> dict:
    """Weighted mean over whichever components are present -> PlayerMetric dict."""
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

    return {
        "metric_name": metric_name,
        "value": round(value, 1) if value is not None else None,
        "method": METHOD_HEURISTIC,
        "confidence": confidence,
        "sample_size": sample_size,
        "sub_scores": sub_scores,
        "schema_version": SCHEMA_VERSION,
    }
