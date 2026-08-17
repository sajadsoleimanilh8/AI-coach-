from __future__ import annotations

import time

from nexus.agents.base import AgentContext
from nexus.agents.orchestrator import DelegationGuard
from nexus.agents.registry import get_agent
from nexus.core.types import GenerationResult, ToolCall, Usage
from nexus.evaluation.runner import EvalHarness, Evaluator
from nexus.evaluation.types import CaseOutcome, EvalCase


class OrchestrationEvaluator(Evaluator):
    """Two case shapes, because delegation has two things worth gating."""

    suite = "orchestration"

    async def run_case(self, case: EvalCase, harness: EvalHarness) -> CaseOutcome:
        if case.input.get("mode") == "guard":
            return self._run_guard_case(case)
        return await self._run_orchestrated_case(case, harness)

    def _run_guard_case(self, case: EvalCase) -> CaseOutcome:
        start = time.monotonic()
        guard = DelegationGuard(
            max_depth=case.input.get("max_depth", 2),
            max_total_delegations=case.input.get("max_total_delegations", 6),
        )

        accepted: list[bool] = []
        reasons: list[str] = []
        for attempt in case.input["attempts"]:
            refusal = guard.try_acquire(
                agent_name=attempt["agent_name"],
                sub_goal=attempt["sub_goal"],
                depth=attempt.get("depth", 1),
            )
            accepted.append(refusal is None)
            reasons.append(refusal or "accepted")

        expected = case.expected["accepted"]
        passed = accepted == expected

        return CaseOutcome(
            case_id=case.id,
            passed=passed,
            score=1.0 if passed else 0.0,
            actual={"accepted": accepted, "reasons": reasons},
            detail=f"expected accepted={expected}, got {accepted}",
            latency_seconds=time.monotonic() - start,
            cost_usd=0.0,
        )

    async def _run_orchestrated_case(self, case: EvalCase, harness: EvalHarness) -> CaseOutcome:
        start = time.monotonic()
        provider = harness.chat_provider

        if provider is None:
            return CaseOutcome(
                case_id=case.id,
                passed=False,
                score=0.0,
                actual={},
                detail="Orchestrated cases require the fake chat provider (fakes-only run).",
                latency_seconds=time.monotonic() - start,
                cost_usd=0.0,
            )

        for index, delegation in enumerate(case.input["scripted_delegations"]):
            provider.enqueue(
                GenerationResult(
                    content="",
                    model_used="fake",
                    provider_name=provider.name,
                    usage=Usage(),
                    tool_calls=[
                        ToolCall(
                            id=str(index),
                            name="delegate",
                            arguments={
                                "agent_name": delegation["agent_name"],
                                "sub_goal": delegation["sub_goal"],
                            },
                        )
                    ],
                )
            )
            provider.enqueue(
                GenerationResult(
                    content=delegation.get("result", "Sub-goal answered."),
                    model_used="fake",
                    provider_name=provider.name,
                    usage=Usage(),
                )
            )
        provider.enqueue(
            GenerationResult(
                content="Synthesized answer.",
                model_used="fake",
                provider_name=provider.name,
                usage=Usage(),
            )
        )

        agent = get_agent("orchestrator", enabled=["orchestrator"])
        context = AgentContext(
            goal=case.input["goal"],
            user_id="eval-orchestration",
            session_id=None,
            rag_service=harness.rag_service,
            long_term_memory=None,
            personal_state=None,
        )
        result = await harness.agent_runtime.run(agent, context)

        delegated_to = [step.agent_name for step in result.delegation_steps]
        actual = {
            "delegated_to": delegated_to,
            "delegation_count": len(result.delegation_steps),
        }

        checks: list[tuple[str, bool]] = []
        if "delegated_to" in case.expected:
            checks.append(("delegated_to", delegated_to == case.expected["delegated_to"]))
        if "max_delegations" in case.expected:
            checks.append(
                (
                    "max_delegations",
                    len(result.delegation_steps) <= case.expected["max_delegations"],
                )
            )

        passed = all(ok for _name, ok in checks) if checks else False
        failed = [name for name, ok in checks if not ok]

        return CaseOutcome(
            case_id=case.id,
            passed=passed,
            score=1.0 if passed else 0.0,
            actual=actual,
            detail=(
                f"delegated_to={delegated_to}"
                + (f"; failed checks: {', '.join(failed)}" if failed else "")
            ),
            latency_seconds=time.monotonic() - start,
            cost_usd=0.0,
        )
