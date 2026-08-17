from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from nexus.core.types import Message, RoutingPolicy, TaskType, Usage
from nexus.health.safety import SafetyVerdict
from nexus.memory.long_term import LongTermMemoryStore
from nexus.personal.state import PersonalStateEngine
from nexus.rag.service import RagService
from nexus.verification.types import VerificationReport


@dataclass
class AgentStep:
    index: int
    thought: str
    tool_name: str | None
    tool_arguments: dict[str, Any] | None
    tool_output: str | None
    timestamp: float


@dataclass
class AgentResult:
    goal: str
    final_answer: str
    steps: list[AgentStep]
    completed: bool
    iterations_used: int
    usage: Usage
    cost_usd: float | None = None
    model_used: str | None = None
    provider_name: str | None = None
    verification: VerificationReport | None = None
    delegation_steps: list[Any] = field(default_factory=list)
    rounds_used: int = 1


@dataclass
class AgentContext:
    goal: str
    user_id: str
    session_id: str | None
    rag_service: RagService
    long_term_memory: LongTermMemoryStore
    personal_state: PersonalStateEngine
    extra: dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    """A goal-driven role that runs through AgentRuntime: a system prompt,
    an explicit tool allowlist (enforced by run_tool_loop, not just
    documented here — principle 2), and a routing task_type/policy.
    """

    name: str
    description: str
    allowed_tools: list[str]
    task_type: TaskType
    default_policy: RoutingPolicy | None = None
    output_filter: Callable[[str], SafetyVerdict] | None = None
    verify_output: bool = False
    max_rounds: int = 1

    @abstractmethod
    def system_prompt(self, context: AgentContext) -> str:
        """Agent-specific instructions, composed with the shared planning
        preamble by AgentRuntime."""

    async def prepare_context(self, context: AgentContext) -> list[Message]:
        """Extra grounding messages inserted before the goal. Default: none
        — override for agent-specific context injection (e.g. RAG chunks,
        personal state)."""
        return []

    def next_round_message(
        self, *, round_index: int, evidence_texts: list[str], context: AgentContext
    ) -> Message | None:
        """Called by AgentRuntime after each round when max_rounds > 1.
        Return a message to run another round, or None to stop. The DEFAULT
        stops immediately, so an agent that never overrides this behaves
        exactly as it did before rounds existed."""
        return None

    def finalize_answer(
        self, answer: str, *, evidence_texts: list[str], context: AgentContext
    ) -> str:
        """Last chance to shape the answer before output_filter and
        verification run. Default is identity."""
        return answer
