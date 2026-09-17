"""Score-range helpers shared by the scoring engines.

These were defined identically in ai/psychology_ai/constants.py and
ai/performance_ai/match_readiness_predictor/constants.py, and the two copies
had already started to drift (one typed `scale_1_to_10` as taking an int, the
other a float). Both constants modules re-export these names, so existing
imports keep working.
"""

from __future__ import annotations


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Bounds a score to [low, high]. Every sub-score and headline score
    passes through here, so no formula can emit an out-of-range number for
    the API/DB layer to store."""
    return float(min(max(value, low), high))


def scale_1_to_10(rating: float) -> float:
    """Maps a 1-10 self-rating onto 0-100 as `10 * rating`, so 1 -> 10 and
    10 -> 100.

    This deliberately never reaches 0: the questionnaire's own floor is 1, and
    pretending a "1" means "zero" would overstate what was reported. That
    documented floor of 10 is what the boundary tests assert.
    """
    return float(10.0 * rating)


def invert(score_0_100: float) -> float:
    """Turns a 'higher is worse' score (fatigue, stress, training load) into a
    'higher is better' goodness score, so factor classification can use one
    direction for every dimension."""
    return clamp(100.0 - score_0_100)
