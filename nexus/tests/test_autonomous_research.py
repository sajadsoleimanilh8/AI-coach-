from __future__ import annotations

import pytest

from nexus.agents.autonomous_research import (
    AutonomousResearchAgent,
    assess_coverage,
    format_coverage,
    split_sub_questions,
)
from nexus.agents.base import AgentContext
from nexus.agents.runtime import AgentRuntime
from nexus.core.providers import AIProvider
from nexus.core.router import RoutingDecision
from nexus.core.types import (
    GenerationChunk,
    GenerationResult,
    ModelInfo,
    RoutingPolicy,
    TaskType,
    ToolCall,
    Usage,
)
from nexus.tools.registry import Tool, ToolRegistry, ToolResult


class _SearchTool(Tool):
    name = "web_search"
    description = "Searches."
    parameters_schema = {"type": "object", "properties": {"query": {"type": "string"}}}

    def __init__(self, output: str) -> None:
        self._output = output
        self.calls = 0

    async def execute(self, arguments) -> ToolResult:
        self.calls += 1
        return ToolResult(success=True, output=self._output)


class _CountingProvider(AIProvider):
    name = "fake"

    def __init__(self, *, emit_tool_call: bool) -> None:
        self._emit_tool_call = emit_tool_call
        self.call_count = 0

    async def generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        self.call_count += 1
        if self._emit_tool_call and self.call_count % 2 == 1:
            return GenerationResult(
                content="Searching.", model_used=model_id, provider_name=self.name, usage=Usage(),
                tool_calls=[ToolCall(id=str(self.call_count), name="web_search", arguments={"query": "x"})],
            )
        return GenerationResult(
            content="Here is what I found.", model_used=model_id, provider_name=self.name,
            usage=Usage(prompt_tokens=5, completion_tokens=5),
        )

    async def stream_generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        yield GenerationChunk(delta="", done=True, usage=Usage())

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> bool:
        return True

    def count_tokens(self, text: str, *, model_id: str) -> int:
        return len(text)


class _FakeRouter:
    def __init__(self, provider: AIProvider) -> None:
        self._provider = provider

    async def route_with_failover(self, **kwargs):
        return (
            RoutingDecision(
                provider_name=self._provider.name, model_id="fake-model",
                task_type=kwargs.get("task_type", TaskType.GENERAL),
                policy=RoutingPolicy.BALANCED, reason="fake",
            ),
            self._provider,
        )


class _StubRag:
    async def retrieve(self, *, user_id, query, top_k):
        return []


def _context(goal: str) -> AgentContext:
    return AgentContext(
        goal=goal, user_id="u1", session_id=None,
        rag_service=_StubRag(), long_term_memory=None, personal_state=None,
    )


# --- the deterministic half -----------------------------------------------


def test_sub_questions_are_split_deterministically() -> None:
    parts = split_sub_questions(
        "What causes thermal throttling and how do I detect it in a training run?"
    )

    assert len(parts) >= 2
    assert split_sub_questions("same goal text") == split_sub_questions("same goal text")


def test_a_goal_with_no_separators_stays_one_sub_question() -> None:
    assert split_sub_questions("Explain gradient checkpointing") == [
        "Explain gradient checkpointing"
    ]


def test_coverage_marks_a_well_supported_question_known() -> None:
    coverage = assess_coverage(
        ["thermal throttling laptop"],
        ["Thermal throttling on a laptop reduces sustained clocks under load."],
    )

    assert coverage.items[0].tier == "known"


def test_coverage_marks_an_unsupported_question_unknown() -> None:
    coverage = assess_coverage(["quantum entanglement basics"], ["Something about gardening."])

    assert coverage.items[0].tier == "unknown"
    assert coverage.items[0].overlap == 0.0


def test_coverage_with_no_evidence_at_all_is_unknown() -> None:
    coverage = assess_coverage(["anything here"], [])

    assert coverage.items[0].tier == "unknown"
    assert not coverage.fully_covered


def test_partial_support_lands_between_the_extremes() -> None:
    coverage = assess_coverage(
        ["alpha beta gamma delta"], ["alpha beta appear but the rest does not"]
    )

    assert coverage.items[0].tier in ("likely", "uncertain")


def test_unresolved_collects_uncertain_and_unknown() -> None:
    coverage = assess_coverage(
        ["alpha beta", "zeta omega"], ["alpha beta are fully covered here"]
    )

    assert [item.sub_question for item in coverage.unresolved] == ["zeta omega"]


def test_output_separates_all_four_tiers() -> None:
    coverage = assess_coverage(
        ["alpha beta", "alpha zeta", "alpha omega kappa lambda", "sigma tau"],
        ["alpha beta gamma zeta"],
    )

    rendered = format_coverage(coverage)

    assert "Known" in rendered
    assert "Unknown" in rendered
    assert "Research coverage" in rendered


def test_format_coverage_is_empty_for_no_items() -> None:
    from nexus.agents.autonomous_research import Coverage

    assert format_coverage(Coverage(items=[])) == ""


# --- the loop -------------------------------------------------------------


def test_agent_declares_multiple_rounds_and_inherits_verification() -> None:
    agent = AutonomousResearchAgent()

    assert agent.max_rounds == 3
    assert agent.verify_output is True  # inherited from ResearchAgent


def test_next_round_message_is_none_once_coverage_is_complete() -> None:
    agent = AutonomousResearchAgent()
    context = _context("thermal throttling")

    result = agent.next_round_message(
        round_index=0,
        evidence_texts=["thermal throttling explained in detail"],
        context=context,
    )

    assert result is None  # the CODE decided to stop, not the model


def test_next_round_message_names_what_is_still_missing() -> None:
    agent = AutonomousResearchAgent()
    context = _context("quantum entanglement basics")

    message = agent.next_round_message(
        round_index=0, evidence_texts=["unrelated gardening notes"], context=context
    )

    assert message is not None
    assert "quantum entanglement basics" in message.content
    assert "Round 1" in message.content


@pytest.mark.asyncio
async def test_loop_terminates_on_coverage_before_max_rounds() -> None:
    tool = _SearchTool("thermal throttling on a laptop reduces sustained clocks")
    provider = _CountingProvider(emit_tool_call=True)
    runtime = AgentRuntime(_FakeRouter(provider), ToolRegistry({"web_search": tool}), max_iterations=4)

    result = await runtime.run(AutonomousResearchAgent(), _context("thermal throttling laptop"))

    assert result.rounds_used == 1  # coverage satisfied after round one


@pytest.mark.asyncio
async def test_loop_terminates_on_max_rounds_when_coverage_never_completes() -> None:
    tool = _SearchTool("entirely unrelated content about gardening")
    provider = _CountingProvider(emit_tool_call=True)
    runtime = AgentRuntime(_FakeRouter(provider), ToolRegistry({"web_search": tool}), max_iterations=4)

    result = await runtime.run(
        AutonomousResearchAgent(), _context("quantum chromodynamics lattice calculations")
    )

    # Never satisfied, so the hard cap is what stops it — not an infinite loop.
    assert result.rounds_used == 3


@pytest.mark.asyncio
async def test_final_answer_carries_the_coverage_breakdown() -> None:
    tool = _SearchTool("some findings")
    provider = _CountingProvider(emit_tool_call=True)
    runtime = AgentRuntime(_FakeRouter(provider), ToolRegistry({"web_search": tool}), max_iterations=4)

    result = await runtime.run(AutonomousResearchAgent(), _context("obscure unmatched topic here"))

    assert "Research coverage" in result.final_answer
    assert "Unknown" in result.final_answer


@pytest.mark.asyncio
async def test_a_single_round_agent_is_unaffected_by_the_loop() -> None:
    """Every pre-existing agent has max_rounds=1 and must behave exactly as
    it did before rounds existed."""
    from nexus.agents.research import ResearchAgent

    provider = _CountingProvider(emit_tool_call=False)
    runtime = AgentRuntime(_FakeRouter(provider), ToolRegistry({}), max_iterations=4)

    result = await runtime.run(ResearchAgent(), _context("anything at all"))

    assert result.rounds_used == 1
    assert result.final_answer == "Here is what I found."  # no coverage block appended
