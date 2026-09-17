from __future__ import annotations

from nexus.agents.base import Agent, AgentContext
from nexus.core.types import Message, TaskType

_RAG_TOP_K = 5


class ResearchAgent(Agent):
    name = "research"
    description = "Investigates a question using web search and document retrieval."
    allowed_tools = ["web_search", "files"]
    task_type = TaskType.RESEARCH
    # The one agent whose output is most citation-dependent and most
    # load-bearing for the user — every other agent is unchanged
    # (verify_output defaults to False on the base class).
    verify_output = True

    def system_prompt(self, context: AgentContext) -> str:
        return (
            "You are a research agent. Search multiple independent sources before "
            "concluding anything. For every factual claim in your final answer, cite "
            "the source it came from. Explicitly separate what is VERIFIED (directly "
            "supported by a tool result), what is INFERRED (your own reasoning from "
            "verified facts), and what remains UNKNOWN — never present an inference "
            "or a guess as a verified fact."
        )

    async def prepare_context(self, context: AgentContext) -> list[Message]:
        retrieved = await context.rag_service.retrieve(
            user_id=context.user_id, query=context.goal, top_k=_RAG_TOP_K
        )
        if not retrieved:
            return []

        # Stashed for AgentRuntime to pick up as verification evidence
        # (verify_output=True above) — AgentContext's fixed fields don't
        # carry RAG results, so context.extra is the one channel available
        # to hand this forward to the runtime after prepare_context() returns.
        context.extra["retrieved_evidence"] = retrieved

        # Same delimited format chat.py's _apply_rag() uses, so retrieved
        # chunks read consistently whether they reach the model via /api/chat
        # or via this agent.
        context_lines = "\n".join(
            f"[{i + 1}] {chunk.chunk_text}" for i, chunk in enumerate(retrieved)
        )
        return [
            Message(
                role="system",
                content=f"Relevant context from your documents:\n\n{context_lines}",
            )
        ]
