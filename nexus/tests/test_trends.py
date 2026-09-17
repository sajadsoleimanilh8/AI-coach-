from __future__ import annotations

import time

import pytest

from nexus.personal.trends import compute_trend

_DAY = 86400.0


def _rising_points(now: float) -> list[tuple[float, float]]:
    return [
        (now - 3 * _DAY, 0.2),
        (now - 2 * _DAY, 0.4),
        (now - 1 * _DAY, 0.6),
        (now, 0.8),
    ]


def _falling_points(now: float) -> list[tuple[float, float]]:
    return [
        (now - 3 * _DAY, 0.8),
        (now - 2 * _DAY, 0.6),
        (now - 1 * _DAY, 0.4),
        (now, 0.2),
    ]


def _flat_points(now: float) -> list[tuple[float, float]]:
    return [(now - i * _DAY, 0.5) for i in range(4)]


def test_rising_values_are_improving_for_a_normal_dimension() -> None:
    now = time.time()
    trend = compute_trend("physical.energy", _rising_points(now))

    assert trend is not None
    assert trend.direction == "improving"
    assert trend.slope == pytest.approx(0.2, abs=1e-6)
    assert trend.confidence == pytest.approx(1.0, abs=1e-6)
    assert trend.sample_count == 4


def test_falling_values_are_declining_for_a_normal_dimension() -> None:
    now = time.time()
    trend = compute_trend("physical.energy", _falling_points(now))

    assert trend is not None
    assert trend.direction == "declining"
    assert trend.slope == pytest.approx(-0.2, abs=1e-6)


def test_rising_values_are_declining_for_an_inverted_dimension() -> None:
    now = time.time()
    trend = compute_trend("mental.stress", _rising_points(now))

    assert trend is not None
    assert trend.direction == "declining"
    assert trend.slope == pytest.approx(0.2, abs=1e-6)


def test_falling_values_are_improving_for_an_inverted_dimension() -> None:
    now = time.time()
    trend = compute_trend("mental.fatigue", _falling_points(now))

    assert trend is not None
    assert trend.direction == "improving"


def test_flat_series_is_stable_regardless_of_inversion() -> None:
    now = time.time()
    normal_trend = compute_trend("physical.energy", _flat_points(now))
    inverted_trend = compute_trend("mental.stress", _flat_points(now))

    assert normal_trend is not None and normal_trend.direction == "stable"
    assert inverted_trend is not None and inverted_trend.direction == "stable"
    assert normal_trend.slope == pytest.approx(0.0, abs=1e-6)
    assert normal_trend.confidence == pytest.approx(1.0, abs=1e-6)  # flat = certain "stable"


def test_below_min_samples_returns_none() -> None:
    now = time.time()
    points = _rising_points(now)[:3]  # 3 points, default min_samples is 4

    assert compute_trend("physical.energy", points) is None


def test_small_slope_within_epsilon_reads_as_stable() -> None:
    now = time.time()
    points = [
        (now - 3 * _DAY, 0.500),
        (now - 2 * _DAY, 0.501),
        (now - 1 * _DAY, 0.502),
        (now, 0.503),
    ]

    trend = compute_trend("physical.energy", points)

    assert trend is not None
    assert trend.direction == "stable"
