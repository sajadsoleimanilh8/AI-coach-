from __future__ import annotations

from nexus.core.router import ModelRouter
from nexus.core.types import TaskType, Usage
from nexus.verification.checks import DETERMINISTIC_CHECKS, CheckContext
from nexus.verification.fact_checker import FactChecker
from nexus.verification.judge import MultiModelJudge
from nexus.verification.types import (
    CheckResult,
    CheckStatus,
    ConfidenceBand,
    VerificationReport,
    band_for_score,
    score_checks,
)

_JUDGE_CHECK_NAME = "multi_model_judge"


def _add_usage(a: Usage, b: Usage) -> Usage:
    return Usage(
        prompt_tokens=a.prompt_tokens + b.prompt_tokens,
        completion_tokens=a.completion_tokens + b.completion_tokens,
    )


class VerificationEngine:
    def __init__(
        self,
        router: ModelRouter,
        fact_checker: FactChecker,
        judge: MultiModelJudge,
        *,
        escalate_below: float = 0.65,
        enable_fact_check: bool = True,
        enable_judge: bool = True,
    ) -> None:
        self._router = router
        self._fact_checker = fact_checker
        self._judge = judge
        self._escalate_below = escalate_below
        self._enable_fact_check = enable_fact_check
        self._enable_judge = enable_judge

    async def verify(
        self,
        *,
        question: str,
        answer: str,
        model_id: str,
        task_type: TaskType,
        evidence=None,
    ) -> VerificationReport:
        # Step 1: deterministic checks always run — they're free (no LLM
        # call), so there's no cost reason to gate them behind anything.
        context = CheckContext(text=answer, evidence=evidence)
        checks: list[CheckResult] = [check(context) for check in DETERMINISTIC_CHECKS]

        extra_usage = Usage()
        escalated = False

        # Step 2: model-based claim extraction + checking, opt-in (costs tokens).
        if self._enable_fact_check:
            fact_check_results, fact_check_usage = await self._fact_checker.run(answer, evidence)
            checks.extend(fact_check_results)
            extra_usage = _add_usage(extra_usage, fact_check_usage)

        # Step 3: composite score/band from everything computed so far.
        score = score_checks(checks)
        band = band_for_score(score)

        # Step 4: escalate to a second model only when confidence is low
        # (or entirely unverified) — this is the expensive step, so it's
        # gated on actually needing it, not run unconditionally.
        if self._enable_judge and (score is None or score < self._escalate_below):
            escalated = True
            verdict = await self._judge.judge(
                question=question, answer=answer, original_model_id=model_id, task_type=task_type
            )
            if verdict is None:
                checks.append(
                    CheckResult(
                        name=_JUDGE_CHECK_NAME, status=CheckStatus.INCONCLUSIVE, weight=1.0,
                        detail="No distinct healthy model was available to consult for a "
                               "second opinion.",
                    )
                )
            else:
                extra_usage = _add_usage(extra_usage, verdict.usage)
                if verdict.agrees:
                    checks.append(
                        CheckResult(
                            name=_JUDGE_CHECK_NAME, status=CheckStatus.PASS, weight=1.0,
                            detail=f"Judge model {verdict.judge_model} agreed with the answer.",
                        )
                    )
                else:
                    detail = verdict.disagreement_summary or (
                        f"Judge model {verdict.judge_model} disagreed with the answer."
                    )
                    checks.append(
                        CheckResult(
                            name=_JUDGE_CHECK_NAME, status=CheckStatus.FAIL, weight=1.0,
                            detail=detail,
                            evidence=[verdict.disagreement_summary] if verdict.disagreement_summary else [],
                        )
                    )
            # Recomputed AFTER the judge's check lands — escalation exists
            # to move the final verdict, not to sit alongside a stale
            # pre-escalation score.
            score = score_checks(checks)
            band = band_for_score(score)

        uncertainty_notes = [
            c.detail for c in checks if c.status in (CheckStatus.FAIL, CheckStatus.INCONCLUSIVE)
        ]

        return VerificationReport(
            checks=checks,
            score=score,
            band=band,
            summary=_summarize(band, score, checks),
            uncertainty_notes=uncertainty_notes,
            escalated=escalated,
            extra_usage=extra_usage,
        )


def _summarize(band: ConfidenceBand, score: float | None, checks: list[CheckResult]) -> str:
    passed = sum(1 for c in checks if c.status == CheckStatus.PASS)
    failed = sum(1 for c in checks if c.status == CheckStatus.FAIL)
    inconclusive = sum(1 for c in checks if c.status == CheckStatus.INCONCLUSIVE)
    score_text = f"{score:.2f}" if score is not None else "n/a"
    return (
        f"{band.value} confidence (score={score_text}): {passed} check(s) passed, "
        f"{failed} failed, {inconclusive} inconclusive."
    )
