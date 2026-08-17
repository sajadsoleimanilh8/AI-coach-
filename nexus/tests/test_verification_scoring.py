from __future__ import annotations

import pytest

from nexus.verification.types import (
    CheckResult,
    CheckStatus,
    ConfidenceBand,
    band_for_score,
    score_checks,
)


def _check(status: CheckStatus, weight: float = 1.0) -> CheckResult:
    return CheckResult(name="x", status=status, weight=weight, detail="detail")


def test_all_inconclusive_yields_none_score_never_a_default() -> None:
    checks = [_check(CheckStatus.INCONCLUSIVE), _check(CheckStatus.INCONCLUSIVE, weight=5.0)]
    assert score_checks(checks) is None


def test_all_inconclusive_yields_unverified_band_never_high() -> None:
    score = score_checks([_check(CheckStatus.INCONCLUSIVE)])
    band = band_for_score(score)
    assert band == ConfidenceBand.UNVERIFIED
    assert band != ConfidenceBand.HIGH


def test_empty_checks_list_is_also_unverified() -> None:
    assert score_checks([]) is None
    assert band_for_score(score_checks([])) == ConfidenceBand.UNVERIFIED


def test_inconclusive_checks_excluded_from_denominator_entirely() -> None:
    checks = [_check(CheckStatus.PASS, weight=1.0), _check(CheckStatus.INCONCLUSIVE, weight=100.0)]
    assert score_checks(checks) == pytest.approx(1.0)


def test_inconclusive_does_not_reward_or_punish() -> None:
    only_fail = score_checks([_check(CheckStatus.FAIL, weight=1.0)])
    fail_plus_inconclusive = score_checks(
        [_check(CheckStatus.FAIL, weight=1.0), _check(CheckStatus.INCONCLUSIVE, weight=1.0)]
    )
    assert only_fail == fail_plus_inconclusive == pytest.approx(0.0)


def test_score_is_weighted_pass_fraction() -> None:
    checks = [
        _check(CheckStatus.PASS, weight=2.0),
        _check(CheckStatus.PASS, weight=1.0),
        _check(CheckStatus.FAIL, weight=1.0),
    ]
    assert score_checks(checks) == pytest.approx(0.75)


@pytest.mark.parametrize(
    "score,expected_band",
    [
        (1.0, ConfidenceBand.HIGH),
        (0.85, ConfidenceBand.HIGH),
        (0.84, ConfidenceBand.MEDIUM),
        (0.65, ConfidenceBand.MEDIUM),
        (0.64, ConfidenceBand.LOW),
        (0.45, ConfidenceBand.LOW),
        (0.44, ConfidenceBand.UNCERTAIN),
        (0.0, ConfidenceBand.UNCERTAIN),
    ],
)
def test_band_thresholds_map_correctly(score: float, expected_band: ConfidenceBand) -> None:
    assert band_for_score(score) == expected_band


def test_band_for_none_is_always_unverified() -> None:
    assert band_for_score(None) == ConfidenceBand.UNVERIFIED
