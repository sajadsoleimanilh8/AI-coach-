from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from nexus.personal.dimensions import INVERTED_DIMENSIONS

_SECONDS_PER_DAY = 86400.0
_DEFAULT_MIN_SAMPLES = 4
_DEFAULT_STABLE_EPSILON = 0.01


@dataclass
class Trend:
    dimension: str
    direction: Literal["improving", "declining", "stable"]
    slope: float
    confidence: float
    sample_count: int


def compute_trend(
    dimension: str,
    points: list[tuple[float, float]],
    *,
    min_samples: int = _DEFAULT_MIN_SAMPLES,
    stable_epsilon: float = _DEFAULT_STABLE_EPSILON,
) -> Trend | None:
    """Hand-rolled least-squares linear fit over (timestamp_seconds, value)
    points — arithmetic, not a model call (principle 4). `confidence` is the
    fit's R^2: a trend line that barely explains the scatter shouldn't be
    reported with the same confidence as a clean one, even at equal
    """
    if len(points) < min_samples:
        return None

    n = len(points)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    s_xx = sum((x - mean_x) ** 2 for x in xs)
    s_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope_per_second = 0.0 if s_xx == 0 else s_xy / s_xx
    intercept = mean_y - slope_per_second * mean_x

    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    if ss_tot == 0:
        confidence = 1.0
    else:
        ss_res = sum((y - (intercept + slope_per_second * x)) ** 2 for x, y in zip(xs, ys))
        confidence = max(0.0, 1 - ss_res / ss_tot)

    slope_per_day = slope_per_second * _SECONDS_PER_DAY

    if abs(slope_per_day) < stable_epsilon:
        direction: Literal["improving", "declining", "stable"] = "stable"
    else:
        rising = slope_per_day > 0
        is_inverted = dimension in INVERTED_DIMENSIONS
        if rising:
            direction = "declining" if is_inverted else "improving"
        else:
            direction = "improving" if is_inverted else "declining"

    return Trend(
        dimension=dimension,
        direction=direction,
        slope=slope_per_day,
        confidence=confidence,
        sample_count=n,
    )
