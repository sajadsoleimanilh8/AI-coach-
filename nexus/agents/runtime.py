from __future__ import annotations

from nexus.agents.base import Agent, AgentContext, AgentResult, AgentStep
from nexus.agents.orchestrator import (
    DELEGATE_TOOL_NAME,
    DelegateTool,
    DelegationGuard,
    DelegationStep,
)
from nexus.core.cost_tracker import CostTracker
from nexus.core.router import ModelRouter
from nexus.core.tool_loop import ToolCallSummaryData, run_tool_loop
from nexus.core.types import Message, Usage
from nexus.models.registry import get_model
from nexus.tools.registry import ToolRegistry
from nexus.verification.engine import VerificationEngine

_DEFAULT_TEMPERATURE = 0.7

PLANNING_PREAMBLE = (
    "You are an autonomous agent working toward a specific goal. Break the "
    "goal into concrete steps. Use the tools available to you to gather real "
    "evidence rather than guessing or fabricating information — only state "
    "something as fact if a tool call (or your own sound reasoning) actually "
    "supports it. When you are uncertain, say so explicitly rather than "
    "presenting a guess as settled. Once you have enough evidence, stop "
    "calling tools and give a clear, direct final answer that addresses the "
    "goal."
)


class AgentRuntime:
    """Executes an Agent against a goal. Routing goes through ModelRouter
    with require_tool_calling=True; execution goes through the shared
    run_tool_loop() with the agent's allowed_tools — the same two
    components chat.py's tool-calling path already uses, so agents get
    """

    def __init__(
        self,
        router: ModelRouter,
        tool_registry: ToolRegistry,
        *,
        max_iterations: int,
        cost_tracker: CostTracker | None = None,
        verification_engine: VerificationEngine | None = None,
        agent_factory=None,
        enabled_agents: list[str] | None = None,
        orchestration_max_depth: int = 2,
        orchestration_max_total_delegations: int = 6,
    ) -> None:
        self._router = router
        self._tool_registry = tool_registry
        self._max_iterations = max_iterations
        self._cost_tracker = cost_tracker
        self._verification_engine = verification_engine
        self._agent_factory = agent_factory
        self._enabled_agents = enabled_agents or []
        self._orchestration_max_depth = orchestration_max_depth
        self._orchestration_max_total_delegations = orchestration_max_total_delegations

    async def run(
        self,
        agent: Agent,
        context: AgentContext,
        *,
        depth: int = 0,
        guard: DelegationGuard | None = None,
    ) -> AgentResult:
        system_content = f"{PLANNING_PREAMBLE}\n\n{agent.system_prompt(context)}"
        messages: list[Message] = [Message(role="system", content=system_content)]
        messages.extend(await agent.prepare_context(context))
        messages.append(Message(role="user", content=context.goal))

        decision, provider = await self._router.route_with_failover(
            task_type=agent.task_type,
            policy=agent.default_policy,
            require_tool_calling=True,
        )

        max_iterations = context.extra.get("max_iterations") or self._max_iterations

        active_guard = guard or DelegationGuard(
            max_depth=self._orchestration_max_depth,
            max_total_delegations=self._orchestration_max_total_delegations,
        )
        tool_registry, delegate_tool = self._registry_for(agent, context, depth, active_guard)

        conversation = list(messages)
        evidence_texts: list[str] = []
        all_tool_calls: list[ToolCallSummaryData] = []
        total_usage = Usage()
        final_content = ""
        hit_iteration_cap = False
        iterations_used = 0
        rounds_used = 0

        for round_index in range(max(1, agent.max_rounds)):
            outcome = await run_tool_loop(
                provider,
                conversation,
                model_id=decision.model_id,
                temperature=_DEFAULT_TEMPERATURE,
                max_tokens=None,
                tool_registry=tool_registry,
                max_iterations=max_iterations,
                allowed_tools=agent.allowed_tools,
            )

            rounds_used = round_index + 1
            all_tool_calls.extend(outcome.tool_calls_made)
            evidence_texts.extend(call.result_summary for call in outcome.tool_calls_made)
            total_usage = Usage(
                prompt_tokens=total_usage.prompt_tokens + outcome.result.usage.prompt_tokens,
                completion_tokens=total_usage.completion_tokens
                + outcome.result.usage.completion_tokens,
            )
            final_content = outcome.result.content
            hit_iteration_cap = outcome.hit_iteration_cap
            iterations_used += outcome.iterations_used

            if rounds_used >= max(1, agent.max_rounds):
                break

            next_message = agent.next_round_message(
                round_index=round_index, evidence_texts=evidence_texts, context=context
            )
            if next_message is None:
                break

            conversation = conversation + [
                Message(role="assistant", content=final_content),
                next_message,
            ]

        final_answer = agent.finalize_answer(
            final_content, evidence_texts=evidence_texts, context=context
        )

        if agent.output_filter is not None:
            verdict = agent.output_filter(final_answer)
            if not verdict.allowed:
                final_answer = verdict.rewritten_text or final_answer

        verification = None
        if agent.verify_output and self._verification_engine is not None:
            verification = await self._verification_engine.verify(
                question=context.goal,
                answer=final_answer,
                model_id=decision.model_id,
                task_type=agent.task_type,
                evidence=context.extra.get("retrieved_evidence"),
            )

        steps = [
            AgentStep(
                index=i,
                thought=call.thought,
                tool_name=call.name,
                tool_arguments=call.arguments,
                tool_output=call.result_summary,
                timestamp=call.timestamp,
            )
            for i, call in enumerate(all_tool_calls)
        ]

        cost_usd: float | None = None
        if self._cost_tracker is not None:
            model_info = get_model(decision.model_id)
            cost_usd = await self._cost_tracker.record(
                session_id=context.session_id or f"agent:{agent.name}:{context.user_id}",
                provider_name=decision.provider_name,
                model_id=decision.model_id,
                usage=total_usage,
                model_info=model_info,
            )

        return AgentResult(
            goal=context.goal,
            final_answer=final_answer,
            steps=steps,
            completed=not hit_iteration_cap,
            iterations_used=iterations_used,
            usage=total_usage,
            cost_usd=cost_usd,
            model_used=decision.model_id,
            provider_name=decision.provider_name,
            verification=verification,
            delegation_steps=list(delegate_tool.steps) if delegate_tool else [],
            rounds_used=rounds_used,
        )

    def _registry_for(
        self,
        agent: Agent,
        context: AgentContext,
        depth: int,
        guard: DelegationGuard,
    ) -> tuple[ToolRegistry, DelegateTool | None]:
        """Agents that don't delegate get the shared registry untouched.
        One that does gets a per-run copy with a `delegate` tool bound to
        this run's guard and depth — which is why it can't be registered
        once at startup."""
        if DELEGATE_TOOL_NAME not in agent.allowed_tools or self._agent_factory is None:
            return self._tool_registry, None

        delegatable = [
            name
            for name in self._enabled_agents
            if name not in (agent.name, "orchestrator")
        ]
        context.extra["delegatable_agents"] = delegatable

        delegate_tool = DelegateTool(
            guard=guard,
            depth=depth,
            run_sub_agent=self._make_sub_agent_runner(context, guard),
            available_agents=delegatable,
        )
        merged = self._tool_registry.tools()
        merged[DELEGATE_TOOL_NAME] = delegate_tool
        return ToolRegistry(merged), delegate_tool

    def _make_sub_agent_runner(self, parent_context: AgentContext, guard: DelegationGuard):
        async def run_sub_agent(agent_name: str, sub_goal: str, depth: int):
            sub_agent = self._agent_factory(agent_name)
            sub_context = AgentContext(
                goal=sub_goal,
                user_id=parent_context.user_id,
                session_id=parent_context.session_id,
                rag_service=parent_context.rag_service,
                long_term_memory=parent_context.long_term_memory,
                personal_state=parent_context.personal_state,
                extra=dict(parent_context.extra),
            )
            result = await self.run(sub_agent, sub_context, depth=depth, guard=guard)
            return result.final_answer, result.usage

        return run_sub_agent


__all__ = ["AgentRuntime", "PLANNING_PREAMBLE", "DelegationStep"]
