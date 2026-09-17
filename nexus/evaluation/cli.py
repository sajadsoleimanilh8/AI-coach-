from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from nexus.config.settings import get_settings
from nexus.evaluation.regression import (
    ProviderModeMismatchError,
    RegressionFinding,
    compare_runs,
)
from nexus.evaluation.runner import EvalHarness
from nexus.evaluation.store import EvalStore
from nexus.evaluation.types import EvalRun
from nexus.memory.storage import create_async_db_engine

_SAFETY_SUITE = "safety"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NEXUS evaluation suite.")
    parser.add_argument(
        "--suites", type=str, default=None,
        help="Comma-separated suite names to run (default: settings.evaluation.default_suites)",
    )
    parser.add_argument(
        "--real-providers", action="store_true",
        help="Hit real configured providers instead of fakes (default off, principle 6)",
    )
    parser.add_argument(
        "--compare-to", type=str, default=None,
        help="Baseline run_id to compare this run against; prints regression findings",
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output for CI")
    return parser.parse_args(argv)


def _run_summary(run: EvalRun) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "git_sha": run.git_sha,
        "suites": [
            {
                "suite": s.suite,
                "pass_rate": s.pass_rate,
                "mean_score": s.mean_score,
                "total_cost_usd": s.total_cost_usd,
                "total_latency_seconds": s.total_latency_seconds,
                "case_count": len(s.outcomes),
                "failed_cases": [o.case_id for o in s.outcomes if not o.passed],
            }
            for s in run.suites
        ],
    }


def _finding_dict(finding: RegressionFinding) -> dict[str, Any]:
    return {
        "suite": finding.suite, "case_id": finding.case_id, "kind": finding.kind,
        "before": finding.before, "after": finding.after, "detail": finding.detail,
    }


def _print_human(run: EvalRun, findings: list[RegressionFinding], safety_failed: bool, compare_to: str | None) -> None:
    duration = run.finished_at - run.started_at
    print(f"Eval run {run.run_id} ({duration:.2f}s, git_sha={run.git_sha or 'unknown'})")
    for suite in run.suites:
        marker = "OK" if suite.pass_rate == 1.0 else "FAIL"
        print(
            f"  [{marker}] {suite.suite}: pass_rate={suite.pass_rate:.2f} "
            f"mean_score={suite.mean_score:.2f} n={len(suite.outcomes)} "
            f"cost=${suite.total_cost_usd:.4f} latency={suite.total_latency_seconds:.2f}s"
        )
        for outcome in suite.outcomes:
            if not outcome.passed:
                print(f"      FAIL {outcome.case_id}: {outcome.detail}")

    if compare_to:
        if findings:
            print(f"\n{len(findings)} regression(s) found vs. baseline {compare_to}:")
            for finding in findings:
                case_suffix = f"/{finding.case_id}" if finding.case_id else ""
                print(f"  [{finding.kind}] {finding.suite}{case_suffix}: {finding.detail}")
        else:
            print(f"\nNo regressions found vs. baseline {compare_to}.")

    if safety_failed:
        print("\nSAFETY SUITE FAILED — this is a build breaker, not a percentage.")


async def _main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    suites = args.suites.split(",") if args.suites else None

    harness = EvalHarness(use_real_providers=args.real_providers)
    run = await harness.run(suites)

    engine = create_async_db_engine(settings.memory.database_path)
    store = EvalStore(engine)
    await store.init()
    await store.save_run(run)

    safety_suite = next((s for s in run.suites if s.suite == _SAFETY_SUITE), None)
    safety_failed = safety_suite is not None and safety_suite.pass_rate < 1.0

    findings: list[RegressionFinding] = []
    if args.compare_to:
        baseline = await store.get_run(args.compare_to)
        if baseline is None:
            message = f"Baseline run_id={args.compare_to!r} not found."
            print(json.dumps({"error": message}) if args.json else message)
            await engine.dispose()
            return 1
        try:
            findings = compare_runs(
                baseline, run,
                pass_rate_tolerance=settings.evaluation.pass_rate_tolerance,
                cost_tolerance=settings.evaluation.cost_tolerance,
                latency_tolerance=settings.evaluation.latency_tolerance,
            )
        except ProviderModeMismatchError as exc:
            # A refusal to compare is a legitimate answer, but it is still
            # a failure to answer the question that was asked — so it
            # exits non-zero rather than printing "no regressions found".
            message = str(exc)
            print(json.dumps({"error": message}) if args.json else f"CANNOT COMPARE: {message}")
            await engine.dispose()
            return 1

    if args.json:
        print(
            json.dumps(
                {
                    "run": _run_summary(run),
                    "regressions": [_finding_dict(f) for f in findings],
                    "safety_failed": safety_failed,
                },
                indent=2,
            )
        )
    else:
        _print_human(run, findings, safety_failed, args.compare_to)

    await engine.dispose()
    return 1 if (safety_failed or findings) else 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
