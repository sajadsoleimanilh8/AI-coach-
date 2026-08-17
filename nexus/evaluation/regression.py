from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from nexus.evaluation.types import EvalRun, SuiteResult

_SAFETY_SUITE = "safety"


class ProviderModeMismatchError(ValueError):
    """Raised when a comparison spans two different provider modes."""


@dataclass
class RegressionFinding:
    suite: str
    case_id: str | None
    kind: Literal["suite_pass_rate_drop", "case_regressed", "cost_spike", "latency_spike"]
    before: float
    after: float
    detail: str


def _pass_rate_findings(
    suite: str, baseline: SuiteResult, candidate: SuiteResult, tolerance: float
) -> list[RegressionFinding]:
    effective_tolerance = 0.0 if suite == _SAFETY_SUITE else tolerance
    drop = baseline.pass_rate - candidate.pass_rate
    if drop <= effective_tolerance:
        return []
    return [
        RegressionFinding(
            suite=suite, case_id=None, kind="suite_pass_rate_drop",
            before=baseline.pass_rate, after=candidate.pass_rate,
            detail=(
                f"{suite} pass_rate dropped from {baseline.pass_rate:.2f} to "
                f"{candidate.pass_rate:.2f} (tolerance {effective_tolerance:.2f})"
            ),
        )
    ]


def _case_regression_findings(
    suite: str, baseline: SuiteResult, candidate: SuiteResult
) -> list[RegressionFinding]:
    baseline_by_case = {o.case_id: o for o in baseline.outcomes}
    findings = []
    for outcome in candidate.outcomes:
        prior = baseline_by_case.get(outcome.case_id)
        if prior is not None and prior.passed and not outcome.passed:
            findings.append(
                RegressionFinding(
                    suite=suite, case_id=outcome.case_id, kind="case_regressed",
                    before=1.0, after=0.0,
                    detail=f"{suite}/{outcome.case_id} regressed from pass to fail: {outcome.detail}",
                )
            )
    return findings


def _cost_findings(
    suite: str, baseline: SuiteResult, candidate: SuiteResult, tolerance: float
) -> list[RegressionFinding]:
    if baseline.total_cost_usd > 0:
        increase = (candidate.total_cost_usd - baseline.total_cost_usd) / baseline.total_cost_usd
        if increase > tolerance:
            return [
                RegressionFinding(
                    suite=suite, case_id=None, kind="cost_spike",
                    before=baseline.total_cost_usd, after=candidate.total_cost_usd,
                    detail=f"{suite} cost increased {increase:.0%} (tolerance {tolerance:.0%})",
                )
            ]
    elif candidate.total_cost_usd > 0:
        return [
            RegressionFinding(
                suite=suite, case_id=None, kind="cost_spike", before=0.0,
                after=candidate.total_cost_usd,
                detail=f"{suite} cost went from $0 to ${candidate.total_cost_usd:.4f}",
            )
        ]
    return []


def _latency_findings(
    suite: str, baseline: SuiteResult, candidate: SuiteResult, tolerance: float
) -> list[RegressionFinding]:
    if baseline.total_latency_seconds <= 0:
        return []
    increase = (
        candidate.total_latency_seconds - baseline.total_latency_seconds
    ) / baseline.total_latency_seconds
    if increase <= tolerance:
        return []
    return [
        RegressionFinding(
            suite=suite, case_id=None, kind="latency_spike",
            before=baseline.total_latency_seconds, after=candidate.total_latency_seconds,
            detail=f"{suite} latency increased {increase:.0%} (tolerance {tolerance:.0%})",
        )
    ]


def compare_runs(
    baseline: EvalRun,
    candidate: EvalRun,
    *,
    pass_rate_tolerance: float = 0.02,
    cost_tolerance: float = 0.25,
    latency_tolerance: float = 0.5,
) -> list[RegressionFinding]:
    if (
        baseline.provider_mode != candidate.provider_mode
        or baseline.provider_mode == "unknown"
        or candidate.provider_mode == "unknown"
    ):
        raise ProviderModeMismatchError(
            f"Refusing to compare a provider_mode={baseline.provider_mode!r} baseline "
            f"(run {baseline.run_id}) against a provider_mode={candidate.provider_mode!r} "
            f"candidate (run {candidate.run_id}). Any findings would reflect the change of "
            f"providers, not a change in model quality. Record a baseline in the same mode "
            f"as the candidate and compare against that."
        )

    baseline_suites = {s.suite: s for s in baseline.suites}
    findings: list[RegressionFinding] = []

    for candidate_suite in candidate.suites:
        baseline_suite = baseline_suites.get(candidate_suite.suite)
        if baseline_suite is None:
            continue

        findings.extend(
            _pass_rate_findings(candidate_suite.suite, baseline_suite, candidate_suite, pass_rate_tolerance)
        )
        findings.extend(_case_regression_findings(candidate_suite.suite, baseline_suite, candidate_suite))
        findings.extend(_cost_findings(candidate_suite.suite, baseline_suite, candidate_suite, cost_tolerance))
        findings.extend(
            _latency_findings(candidate_suite.suite, baseline_suite, candidate_suite, latency_tolerance)
        )

    return findings
