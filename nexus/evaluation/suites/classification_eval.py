from __future__ import annotations

import time

from nexus.evaluation.runner import EvalHarness, Evaluator
from nexus.evaluation.types import CaseOutcome, EvalCase


class ClassificationEvaluator(Evaluator):
    suite = "classification"

    async def run_case(self, case: EvalCase, harness: EvalHarness) -> CaseOutcome:
        start = time.monotonic()
        query = case.input["query"]

        task_classification = harness.task_classifier.classify(query)
        privacy_classification = harness.privacy_classifier.classify(query)

        actual = {
            "task_type": task_classification.task_type.value,
            "privacy_level": privacy_classification.level,
        }
        expected_task_type = case.expected.get("task_type")
        expected_privacy = case.expected.get("privacy_level")

        task_ok = expected_task_type is None or actual["task_type"] == expected_task_type
        privacy_ok = expected_privacy is None or actual["privacy_level"] == expected_privacy
        passed = task_ok and privacy_ok
        matched = sum([task_ok, privacy_ok])
        checked = sum([expected_task_type is not None, expected_privacy is not None])
        score = (matched / checked) if checked else 1.0

        detail = (
            f"expected task_type={expected_task_type!r} got {actual['task_type']!r}; "
            f"expected privacy_level={expected_privacy!r} got {actual['privacy_level']!r}"
        )

        return CaseOutcome(
            case_id=case.id, passed=passed, score=score, actual=actual, detail=detail,
            latency_seconds=time.monotonic() - start, cost_usd=0.0,
        )
