from __future__ import annotations

import time

from nexus.core.types import RoutingPolicy
from nexus.evaluation.runner import EvalHarness, Evaluator
from nexus.evaluation.types import CaseOutcome, EvalCase


class RoutingEvaluator(Evaluator):
    """No LLM needed — asserts ModelRouter's actual decision (task_type
    classification, and optionally which provider class it lands on)
    matches expectation. Pure and fast."""

    suite = "routing"

    supports_model_pinning = False

    async def run_case(self, case: EvalCase, harness: EvalHarness) -> CaseOutcome:
        start = time.monotonic()
        query = case.input["query"]
        policy = RoutingPolicy(case.input.get("policy", "BALANCED"))

        classification = harness.task_classifier.classify(query)
        actual_task_type = classification.task_type.value
        expected_task_type = case.expected.get("task_type")
        task_type_ok = expected_task_type is None or actual_task_type == expected_task_type

        actual_provider_class: str | None = None
        provider_class_ok = True
        if "provider_class" in case.expected:
            decision, _provider = await harness.router.route_with_failover(
                task_type=classification.task_type, policy=policy
            )
            actual_provider_class = "local" if decision.provider_name == "local" else "cloud"
            provider_class_ok = actual_provider_class == case.expected["provider_class"]

        passed = task_type_ok and provider_class_ok
        actual = {"task_type": actual_task_type}
        if actual_provider_class is not None:
            actual["provider_class"] = actual_provider_class

        detail = f"expected task_type={expected_task_type!r}, got {actual_task_type!r}"
        if "provider_class" in case.expected:
            detail += (
                f"; expected provider_class={case.expected['provider_class']!r}, "
                f"got {actual_provider_class!r}"
            )

        return CaseOutcome(
            case_id=case.id, passed=passed, score=1.0 if passed else 0.0, actual=actual,
            detail=detail, latency_seconds=time.monotonic() - start, cost_usd=0.0,
        )
