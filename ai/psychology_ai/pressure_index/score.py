"""
Pressure Index -- how exposed this player reports being to match pressure.

Returns a PlayerMetric-shaped dict per the repo's
ai/<domain>/<metric_name>/score.py convention.

SCALE DIRECTION: HIGHER-IS-WORSE, like stress_analysis. A high pressure index
means the player says pressure tends to reduce their performance and/or is
reporting high stress going into a match they rate as important.

This is a pressure-RESPONSE proxy built from three self-reported answers
(pressure_performance_effect, importance_pressure, and the stress items). It
says nothing about how the player will actually perform, and nothing clinical.
"""

from __future__ import annotations

from ai.common.metrics import MetricResult
from ai.psychology_ai.constants import (
    PRESSURE_RISK_HIGH_MIN,
    PRESSURE_RISK_MODERATE_MIN,
    PRESSURE_RISK_WEIGHTS,
    RISK_HIGH,
    RISK_LOW,
    RISK_MODERATE,
)
from ai.psychology_ai.scoring_utils import score_from_components

METRIC_NAME = "pressure_index"

# press_resistance_score is the observable analogue: how the player actually
# held up when opponents closed them down. Labelled a "pressure response
# proxy" and never as a measure of what they feel.
PROXY_NAMES = ("pressure_response_proxy",)


def pressure_band(index: float) -> str:
    """The low/moderate/high band for a computed index.

    Exported so callers band an index the same way the model does, against the
    same constants -- there is exactly one set of cutoffs for pressure risk in
    this engine.
    """
    if index >= PRESSURE_RISK_HIGH_MIN:
        return RISK_HIGH
    if index >= PRESSURE_RISK_MODERATE_MIN:
        return RISK_MODERATE
    return RISK_LOW


def score_pressure_index(features: dict, historical: dict | None = None) -> MetricResult:
    """Weighted blend of stated pressure sensitivity and reported stress.

    Both are needed: a player who says pressure reduces them but reports no
    stress is genuinely not in the same position as one reporting both, and
    either input alone would flatten that difference.

    The resulting band is carried in sub_scores rather than being a second
    return value, so the PlayerMetric shape stays exactly like its siblings'.
    """
    result = score_from_components(
        METRIC_NAME,
        {
            "sensitivity": (
                features.get("pressure_sensitivity"),
                PRESSURE_RISK_WEIGHTS["sensitivity"],
            ),
            "stress": (features.get("stress_score"), PRESSURE_RISK_WEIGHTS["stress"]),
        },
        historical=historical,
        proxy_names=PROXY_NAMES,
        extra_sub_scores={"direction": "higher_is_worse"},
    )
    # Only banded when there is a value to band -- an ungated None must not
    # acquire a "low" label on the way out.
    result["sub_scores"]["pressure_risk"] = (
        pressure_band(result["value"]) if result["value"] is not None else None
    )
    return result
