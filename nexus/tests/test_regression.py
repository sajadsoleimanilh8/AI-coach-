from __future__ import annotations

import pytest

from nexus.evaluation.regression import ProviderModeMismatchError, compare_runs
from nexus.evaluation.types import CaseOutcome, EvalRun, ProviderMode, SuiteResult


def _outcome(case_id: str, *, passed: bool, cost: float = 0.0, latency: float = 0.0) -> CaseOutcome:
    return CaseOutcome(
        case_id=case_id, passed=passed, score=1.0 if passed else 0.0, actual={}, detail="",
        latency_seconds=latency, cost_usd=cost,
    )


def _run(
    run_id: str, suites: list[SuiteResult], *, provider_mode: ProviderMode = "fake"
) -> EvalRun:
    return EvalRun(
        run_id=run_id, started_at=0.0, finished_at=1.0, suites=suites,
        config_snapshot={}, git_sha=None, provider_mode=provider_mode,
    )


def test_suite_pass_rate_drop_beyond_tolerance_is_flagged() -> None:
    baseline = _run(
        "baseline",
        [SuiteResult.from_outcomes("routing", [_outcome("a", passed=True), _outcome("b", passed=True)])],
    )
    candidate = _run(
        "candidate",
        [SuiteResult.from_outcomes("routing", [_outcome("a", passed=True), _outcome("b", passed=False)])],
    )

    findings = compare_runs(baseline, candidate, pass_rate_tolerance=0.02)

    assert any(f.kind == "suite_pass_rate_drop" and f.suite == "routing" for f in findings)


def test_pass_rate_drop_within_tolerance_is_not_flagged_at_suite_level() -> None:
    outcomes = [_outcome(str(i), passed=True) for i in range(100)]
    baseline = _run("baseline", [SuiteResult.from_outcomes("routing", outcomes)])

    candidate_outcomes = [_outcome(str(i), passed=(i != 0)) for i in range(100)]
    candidate = _run("candidate", [SuiteResult.from_outcomes("routing", candidate_outcomes)])

    findings = compare_runs(baseline, candidate, pass_rate_tolerance=0.02)

    assert not any(f.kind == "suite_pass_rate_drop" for f in findings)


def test_individual_case_regression_is_always_flagged() -> None:
    baseline = _run("baseline", [SuiteResult.from_outcomes("routing", [_outcome("a", passed=True)])])
    candidate = _run("candidate", [SuiteResult.from_outcomes("routing", [_outcome("a", passed=False)])])

    findings = compare_runs(baseline, candidate)

    case_findings = [f for f in findings if f.kind == "case_regressed"]
    assert len(case_findings) == 1
    assert case_findings[0].case_id == "a"
    assert case_findings[0].suite == "routing"


def test_case_that_was_already_failing_is_not_a_new_regression() -> None:
    baseline = _run("baseline", [SuiteResult.from_outcomes("routing", [_outcome("a", passed=False)])])
    candidate = _run("candidate", [SuiteResult.from_outcomes("routing", [_outcome("a", passed=False)])])

    findings = compare_runs(baseline, candidate)

    assert not any(f.kind == "case_regressed" for f in findings)


def test_cost_spike_beyond_tolerance_is_flagged() -> None:
    baseline = _run("baseline", [SuiteResult.from_outcomes("rag", [_outcome("a", passed=True, cost=1.0)])])
    candidate = _run("candidate", [SuiteResult.from_outcomes("rag", [_outcome("a", passed=True, cost=1.5)])])

    findings = compare_runs(baseline, candidate, cost_tolerance=0.25)

    assert any(f.kind == "cost_spike" for f in findings)


def test_cost_increase_within_tolerance_is_not_flagged() -> None:
    baseline = _run("baseline", [SuiteResult.from_outcomes("rag", [_outcome("a", passed=True, cost=1.0)])])
    candidate = _run("candidate", [SuiteResult.from_outcomes("rag", [_outcome("a", passed=True, cost=1.1)])])

    findings = compare_runs(baseline, candidate, cost_tolerance=0.25)

    assert not any(f.kind == "cost_spike" for f in findings)


def test_latency_spike_beyond_tolerance_is_flagged() -> None:
    baseline = _run("baseline", [SuiteResult.from_outcomes("rag", [_outcome("a", passed=True, latency=1.0)])])
    candidate = _run("candidate", [SuiteResult.from_outcomes("rag", [_outcome("a", passed=True, latency=2.0)])])

    findings = compare_runs(baseline, candidate, latency_tolerance=0.5)

    assert any(f.kind == "latency_spike" for f in findings)


def test_safety_suite_pass_rate_drop_is_flagged_even_within_normal_tolerance() -> None:
    baseline_outcomes = [_outcome(str(i), passed=True) for i in range(100)]
    baseline = _run("baseline", [SuiteResult.from_outcomes("safety", baseline_outcomes)])

    candidate_outcomes = [_outcome(str(i), passed=(i != 0)) for i in range(100)]
    candidate = _run("candidate", [SuiteResult.from_outcomes("safety", candidate_outcomes)])

    findings = compare_runs(baseline, candidate, pass_rate_tolerance=0.02)

    assert any(f.kind == "suite_pass_rate_drop" and f.suite == "safety" for f in findings)


def test_safety_case_regression_is_flagged() -> None:
    baseline = _run("baseline", [SuiteResult.from_outcomes("safety", [_outcome("s1", passed=True)])])
    candidate = _run("candidate", [SuiteResult.from_outcomes("safety", [_outcome("s1", passed=False)])])

    findings = compare_runs(baseline, candidate)

    assert any(f.kind == "case_regressed" and f.suite == "safety" for f in findings)


def test_no_regressions_when_nothing_changed() -> None:
    outcomes = [_outcome("a", passed=True, cost=0.5, latency=1.0)]
    baseline = _run("baseline", [SuiteResult.from_outcomes("routing", outcomes)])
    candidate = _run("candidate", [SuiteResult.from_outcomes("routing", outcomes)])

    findings = compare_runs(baseline, candidate)

    assert findings == []


def test_suite_absent_from_baseline_is_skipped_not_errored() -> None:
    baseline = _run("baseline", [])
    candidate = _run("candidate", [SuiteResult.from_outcomes("routing", [_outcome("a", passed=True)])])

    findings = compare_runs(baseline, candidate)

    assert findings == []


def test_comparing_a_fake_baseline_to_a_real_candidate_raises_rather_than_diffing() -> None:
    baseline = _run(
        "fake-baseline",
        [SuiteResult.from_outcomes("safety", [_outcome("s1", passed=True)])],
        provider_mode="fake",
    )
    candidate = _run(
        "real-candidate",
        [SuiteResult.from_outcomes("safety", [_outcome("s1", passed=False)])],
        provider_mode="real",
    )

    with pytest.raises(ProviderModeMismatchError) as excinfo:
        compare_runs(baseline, candidate)

    message = str(excinfo.value)
    assert "'fake'" in message and "'real'" in message


def test_comparison_is_refused_in_either_direction() -> None:
    real = _run("real", [SuiteResult.from_outcomes("routing", [_outcome("a", passed=True)])], provider_mode="real")
    fake = _run("fake", [SuiteResult.from_outcomes("routing", [_outcome("a", passed=True)])], provider_mode="fake")

    with pytest.raises(ProviderModeMismatchError):
        compare_runs(real, fake)


@pytest.mark.parametrize(
    ("baseline_mode", "candidate_mode"),
    [("unknown", "real"), ("real", "unknown"), ("unknown", "unknown")],
)
def test_untagged_runs_are_never_assumed_comparable(
    baseline_mode: ProviderMode, candidate_mode: ProviderMode
) -> None:
    suites = [SuiteResult.from_outcomes("routing", [_outcome("a", passed=True)])]
    baseline = _run("baseline", suites, provider_mode=baseline_mode)
    candidate = _run("candidate", suites, provider_mode=candidate_mode)

    with pytest.raises(ProviderModeMismatchError):
        compare_runs(baseline, candidate)


def test_two_real_runs_compare_normally() -> None:
    baseline = _run(
        "real-baseline",
        [SuiteResult.from_outcomes("routing", [_outcome("a", passed=True)])],
        provider_mode="real",
    )
    candidate = _run(
        "real-candidate",
        [SuiteResult.from_outcomes("routing", [_outcome("a", passed=False)])],
        provider_mode="real",
    )

    findings = compare_runs(baseline, candidate)

    assert any(f.kind == "case_regressed" and f.case_id == "a" for f in findings)
