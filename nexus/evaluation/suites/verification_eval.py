from __future__ import annotations

import time

from nexus.core.types import TaskType
from nexus.core.vector_store import RetrievedChunk
from nexus.evaluation.runner import EvalHarness, Evaluator
from nexus.evaluation.types import CaseOutcome, EvalCase
from nexus.verification.types import CheckStatus, ConfidenceBand


def precision_recall(outcomes: list[CaseOutcome]) -> tuple[float, float]:
    """Reported alongside (not instead of) the standard pass_rate — a
    checker that flags every single answer would score a perfect recall
    while being useless (near-zero precision), and pass_rate alone
    doesn't separate those two failure modes. Computed post-hoc from each
    """
    true_positive = sum(1 for o in outcomes if o.actual.get("should_flag") and o.actual.get("flagged"))
    false_positive = sum(1 for o in outcomes if not o.actual.get("should_flag") and o.actual.get("flagged"))
    false_negative = sum(1 for o in outcomes if o.actual.get("should_flag") and not o.actual.get("flagged"))
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 1.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 1.0
    return precision, recall


class VerificationEvaluator(Evaluator):
    """Runs VerificationEngine against answers with known planted errors
    (must be flagged) and clean answers (must NOT be false-positived).
    Uses enable_fact_check=False/enable_judge=False by default (see
    EvalHarness) so this suite exercises only the deterministic checks —
    """

    suite = "verification"

    supports_model_pinning = True

    async def run_case(self, case: EvalCase, harness: EvalHarness) -> CaseOutcome:
        start = time.monotonic()
        answer = case.input["answer"]
        evidence_raw = case.input.get("evidence") or []
        evidence = (
            [
                RetrievedChunk(
                    doc_id="eval", source_name="eval", chunk_text=e["chunk_text"],
                    score=1.0, cosine_score=1.0,
                )
                for e in evidence_raw
            ]
            or None
        )

        report = await harness.verification_engine.verify(
            question="eval", answer=answer, model_id="eval-model", task_type=TaskType.GENERAL,
            evidence=evidence,
        )

        flagged = report.band != ConfidenceBand.HIGH or any(
            c.status == CheckStatus.FAIL for c in report.checks
        )
        expected_flag = case.expected["should_flag"]
        passed = flagged == expected_flag

        actual = {
            "should_flag": expected_flag, "flagged": flagged,
            "band": report.band.value, "score": report.score,
        }
        detail = (
            f"expected should_flag={expected_flag}, got flagged={flagged} "
            f"(band={report.band.value}, score={report.score})"
        )

        return CaseOutcome(
            case_id=case.id, passed=passed, score=1.0 if passed else 0.0, actual=actual,
            detail=detail, latency_seconds=time.monotonic() - start, cost_usd=0.0,
        )
