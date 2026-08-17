from __future__ import annotations

import time

from nexus.core.tool_loop import run_tool_loop
from nexus.core.types import GenerationResult, Message, ToolCall, Usage
from nexus.evaluation.runner import EvalHarness, Evaluator, FakeChatProvider
from nexus.evaluation.types import CaseOutcome, EvalCase


class ToolsEvaluator(Evaluator):
    """A fresh scripted fake provider emits the case's forced tool_call
    (this is what makes the case deterministic — real tool SELECTION by
    an LLM isn't reproducible enough for a regression gate), then the
    REAL run_tool_loop() decides whether it actually executes, under the
    """

    suite = "tools"

    async def run_case(self, case: EvalCase, harness: EvalHarness) -> CaseOutcome:
        start = time.monotonic()
        provider = FakeChatProvider("eval-tools")
        provider.enqueue(
            GenerationResult(
                content="", model_used="fake", provider_name=provider.name, usage=Usage(),
                tool_calls=[
                    ToolCall(
                        id="1", name=case.input["forced_tool_name"],
                        arguments=case.input.get("forced_tool_arguments", {}),
                    )
                ],
            )
        )
        provider.enqueue(
            GenerationResult(content="Done.", model_used="fake", provider_name=provider.name, usage=Usage())
        )

        outcome = await run_tool_loop(
            provider, [Message(role="user", content=case.input["goal"])],
            model_id="fake", temperature=0.7, max_tokens=None,
            tool_registry=harness.tool_registry, max_iterations=3,
            allowed_tools=case.input.get("allowed_tools"),
        )

        if not outcome.tool_calls_made:
            actual = {"tool_name": None, "executed": False}
            return CaseOutcome(
                case_id=case.id, passed=False, score=0.0, actual=actual,
                detail="No tool call was recorded at all.",
                latency_seconds=time.monotonic() - start, cost_usd=0.0,
            )

        call = outcome.tool_calls_made[0]
        executed = "not permitted" not in call.result_summary
        actual = {"tool_name": call.name, "executed": executed}

        expected_tool = case.expected["tool_name"]
        expected_executed = case.expected["executed"]
        passed = call.name == expected_tool and executed == expected_executed
        detail = (
            f"expected tool_name={expected_tool!r} executed={expected_executed}; "
            f"got tool_name={call.name!r} executed={executed}"
        )

        return CaseOutcome(
            case_id=case.id, passed=passed, score=1.0 if passed else 0.0, actual=actual,
            detail=detail, latency_seconds=time.monotonic() - start, cost_usd=0.0,
        )
