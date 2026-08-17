"""
Pressure Index -- how exposed this player reports being to match pressure.
"""

from __future__ import annotations

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

PROXY_NAMES = ("pressure_response_proxy",)


def pressure_band(index: float) -> str:
    """The low/moderate/high band for a computed index."""
    if index >= PRESSURE_RISK_HIGH_MIN:
        return RISK_HIGH
    if index >= PRESSURE_RISK_MODERATE_MIN:
        return RISK_MODERATE
    return RISK_LOW


def score_pressure_index(features: dict, historical: dict | None = None) -> dict:
    """Weighted blend of stated pressure sensitivity and reported stress."""
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
    result["sub_scores"]["pressure_risk"] = (
        pressure_band(result["value"]) if result["value"] is not None else None
    )
    return result
