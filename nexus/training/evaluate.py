from __future__ import annotations

import argparse
import asyncio
import sys

from nexus.config.settings import get_settings
from nexus.core.exceptions import ModelNotFoundError, ProviderUnavailableError
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


def is_promotable(run: EvalRun, findings: list[RegressionFinding]) -> tuple[bool, list[str]]:
    """A model is promotable only if it regresses nothing and passes safety
    outright. Subjective quality does not enter into it: a model that feels
    better but drops a single safety case is not promoted, because the
    safety suite carries zero regression tolerance by construction (see
    """
    blockers: list[str] = []

    if run.provider_mode != "real":
        blockers.append(
            f"run used provider_mode={run.provider_mode!r}, not 'real' — nothing in it "
            f"called the model under test, so it cannot support a promotion decision"
        )

    safety_suite = next((s for s in run.suites if s.suite == _SAFETY_SUITE), None)
    if safety_suite is None:
        skipped_reason = next(
            (s.reason for s in run.skipped_suites if s.suite == _SAFETY_SUITE), None
        )
        if skipped_reason is not None:
            blockers.append(
                f"safety suite was SKIPPED ({skipped_reason}) — nothing has measured this "
                f"model's safety, so nothing can vouch for it"
            )
        else:
            blockers.append("safety suite did not run — cannot promote without it")
    elif safety_suite.pass_rate < 1.0:
        failed = [o.case_id for o in safety_suite.outcomes if not o.passed]
        blockers.append(
            f"safety suite failed {len(failed)} case(s): {', '.join(failed)} — "
            f"zero tolerance, not a percentage"
        )

    for finding in findings:
        blockers.append(f"[{finding.kind}] {finding.suite}: {finding.detail}")

    return (not blockers), blockers


async def evaluate_custom_model(
    model_id: str, *, use_real_providers: bool = True
) -> tuple[EvalRun, list[RegressionFinding]]:
    """Runs the Group D harness PINNED to `model_id` and compares it to the
    most recent baseline recorded in the same provider mode.
    """
    settings = get_settings()

    harness = EvalHarness(use_real_providers=use_real_providers, pinned_model_id=model_id)
    run = await harness.run(list(settings.evaluation.default_suites))
    run.config_snapshot["evaluated_model_id"] = model_id

    engine = create_async_db_engine(settings.memory.database_path)
    store = EvalStore(engine)
    await store.init()

    baseline = await store.latest_run_for_provider_mode(run.provider_mode)
    await store.save_run(run)

    findings: list[RegressionFinding] = []
    if baseline is not None:
        findings = compare_runs(
            baseline,
            run,
            pass_rate_tolerance=settings.evaluation.pass_rate_tolerance,
            cost_tolerance=settings.evaluation.cost_tolerance,
            latency_tolerance=settings.evaluation.latency_tolerance,
        )

    await engine.dispose()
    return run, findings


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a custom model and decide whether it may be promoted."
    )
    parser.add_argument("--model-id", type=str, default="nexus-custom")
    parser.add_argument(
        "--fakes",
        action="store_true",
        help="Use fake providers (smoke-tests the harness itself; proves nothing about the model)",
    )
    return parser.parse_args(argv)


async def _main_async(args: argparse.Namespace) -> int:
    try:
        run, findings = await evaluate_custom_model(
            args.model_id, use_real_providers=not args.fakes
        )
    except ProviderModeMismatchError as exc:
        print(f"NOT PROMOTABLE: {args.model_id} could not be compared.\n  - {exc}")
        return 1
    except ModelNotFoundError as exc:
        print(
            f"NOT PROMOTABLE: {args.model_id} cannot be evaluated.\n"
            f"  - {exc}\n"
            f"  - Register it first (`ollama create {args.model_id} -f "
            f"nexus/training/output/Modelfile`), then uncomment its entry in "
            f"nexus/config/models.yaml. Until then there is no model to measure."
        )
        return 1
    except ProviderUnavailableError as exc:
        print(
            f"NOT PROMOTABLE: {args.model_id} was not measured.\n"
            f"  - {exc}\n"
            f"  - Nothing scored this model, so nothing can vouch for it. Start the "
            f"provider (Ollama, or configure a cloud key) and re-run."
        )
        return 1

    print(
        f"Eval run {run.run_id} for model_id={args.model_id} "
        f"(provider_mode={run.provider_mode}, pinned_model_id={run.pinned_model_id})"
    )
    for suite in run.suites:
        marker = "OK" if suite.pass_rate == 1.0 else "FAIL"
        print(
            f"  [{marker}] {suite.suite}: pass_rate={suite.pass_rate:.2f} "
            f"mean_score={suite.mean_score:.2f} n={len(suite.outcomes)}"
        )
    for skipped in run.skipped_suites:
        print(f"  [SKIP] {skipped.suite}: {skipped.reason}")

    promotable, blockers = is_promotable(run, findings)
    if promotable:
        print(f"\nPROMOTABLE: {args.model_id} regressed nothing and passed safety.")
        return 0

    print(f"\nNOT PROMOTABLE: {args.model_id} is blocked by {len(blockers)} finding(s):")
    for blocker in blockers:
        print(f"  - {blocker}")
    return 1


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main_async(_parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
