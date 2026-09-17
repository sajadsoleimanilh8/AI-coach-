from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from typing import Any

from nexus.core.providers import AIProvider
from nexus.core.types import GenerationResult, Message, Usage
from nexus.tools.registry import ToolRegistry

_TRUNCATE_CHAR_LIMIT = 300


def _truncate(text: str, limit: int = _TRUNCATE_CHAR_LIMIT) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


@dataclass
class ToolCallSummaryData:
    """Plain-dataclass mirror of api.schemas.ToolCallSummary — core/ must
    not depend on api/, so the pydantic conversion happens at the API
    boundary (chat.py, agents.py), not here.

    `thought` and `timestamp` are extra fields AgentRuntime needs to build
    its AgentStep trace (principle 6: explainability) that ToolCallSummary
    deliberately does NOT expose over the /api/chat wire format — chat.py's
    mapping only reads name/arguments/result_summary, so adding fields here
    is invisible to existing chat.py behavior/tests.
    """

    name: str
    arguments: dict[str, Any]
    result_summary: str
    thought: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolLoopOutcome:
    result: GenerationResult
    tool_calls_made: list[ToolCallSummaryData]
    hit_iteration_cap: bool
    iterations_used: int = 0


async def run_tool_loop(
    provider: AIProvider,
    messages: list[Message],
    *,
    model_id: str,
    temperature: float,
    max_tokens: int | None,
    tool_registry: ToolRegistry,
    max_iterations: int,
    allowed_tools: list[str] | None = None,
) -> ToolLoopOutcome:
    """Shared tool-calling loop used by both /api/chat (use_tools=True) and
    AgentRuntime. Call the model with tool schemas attached -> if it returns
    tool calls, execute each via tool_registry and feed results back as
    role="tool" messages -> repeat until a plain answer or max_iterations.

    `allowed_tools=None` exposes every registered tool (today's chat.py
    behavior, unchanged). A given list restricts both what the model is
    shown (schemas are pre-filtered, so a permitted-only model can't even
    "discover" a disallowed tool) and what actually executes: a tool_call
    for a name outside the allowlist is refused with a role="tool" message
    explaining why, never silently run — an agent's declared permissions
    must hold even if the underlying model tries to call something else.
    """
    all_schemas = tool_registry.enabled_tool_schemas()
    if allowed_tools is None:
        tool_schemas = all_schemas
        allowed_set: set[str] | None = None
    else:
        allowed_set = set(allowed_tools)
        tool_schemas = [schema for schema in all_schemas if schema["name"] in allowed_set]

    conversation = list(messages)
    tool_calls_made: list[ToolCallSummaryData] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    result: GenerationResult | None = None
    hit_iteration_cap = True
    iterations_used = 0

    for _ in range(max_iterations):
        iterations_used += 1
        result = await provider.generate(
            conversation,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tool_schemas,
        )
        total_prompt_tokens += result.usage.prompt_tokens
        total_completion_tokens += result.usage.completion_tokens

        if not result.tool_calls:
            hit_iteration_cap = False
            break

        # Every tool_call returned alongside this same generate() response
        # shares this iteration's assistant text as its "thought" — that
        # text is the model's stated reasoning for the whole batch of calls,
        # not any single one of them.
        iteration_thought = result.content or ""

        for call in result.tool_calls:
            if allowed_set is not None and call.name not in allowed_set:
                output = f"Tool '{call.name}' is not permitted for this agent and was not executed."
                tool_calls_made.append(
                    ToolCallSummaryData(
                        name=call.name,
                        arguments=call.arguments,
                        result_summary=_truncate(output),
                        thought=iteration_thought,
                    )
                )
                conversation.append(
                    Message(
                        role="tool",
                        content=json.dumps({"tool_call_id": call.id, "output": output}),
                    )
                )
                continue

            tool_result = await tool_registry.execute(call.name, call.arguments)
            output = tool_result.output if tool_result.success else (tool_result.error or "")
            tool_calls_made.append(
                ToolCallSummaryData(
                    name=call.name,
                    arguments=call.arguments,
                    result_summary=_truncate(output),
                    thought=iteration_thought,
                )
            )
            conversation.append(
                Message(
                    role="tool", content=json.dumps({"tool_call_id": call.id, "output": output})
                )
            )

    assert result is not None  # max_iterations >= 1 always, so the loop runs at least once
    combined_usage = Usage(
        prompt_tokens=total_prompt_tokens, completion_tokens=total_completion_tokens
    )
    return ToolLoopOutcome(
        result=replace(result, usage=combined_usage),
        tool_calls_made=tool_calls_made,
        hit_iteration_cap=hit_iteration_cap,
        iterations_used=iterations_used,
    )
