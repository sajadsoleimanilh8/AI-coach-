from __future__ import annotations

from nexus.agents.base import Agent, AgentContext
from nexus.core.types import TaskType


class CodingAgent(Agent):
    name = "coding"
    description = "Reads and modifies code, verifying changes by executing them."
    allowed_tools = ["files", "python", "github"]
    task_type = TaskType.CODING

    def system_prompt(self, context: AgentContext) -> str:
        return (
            "You are a coding agent. Read the actual code (via the files or github "
            "tools) before proposing any change — never speculate about what code "
            "does without looking at it. When debugging, identify the root cause, "
            "not just a symptom that happens to make the visible error go away. "
            "Where possible, verify your proposed change actually works by executing "
            "it with the python tool before presenting it as the final answer."
        )
