"""
Utility math and missing-data functions for tactical analysis and player intelligence scoring.
"""

from __future__ import annotations

import math


def safe_ratio(numerator: float | None, denominator: float | None, default=None) -> float | None:
    """Zero-denominator guard per docs/data_analysis.md §0.7."""
    if numerator is None or denominator is None or denominator == 0:
        return default
    return numerator / denominator

def clip(val: float, min_val: float, max_val: float) -> float:
    """Clips val between min_val and max_val."""
    return max(min_val, min(max_val, val))

def gaussian_score(val: float | None, ideal: float, tolerance: float) -> float:
    """Returns 100 * exp(-0.5 * ((val - ideal) / tolerance)^2). Peak 100 at ideal."""
    if val is None or tolerance == 0:
        return 0.0
    diff = (val - ideal) / tolerance
    return 100.0 * math.exp(-0.5 * diff * diff)
