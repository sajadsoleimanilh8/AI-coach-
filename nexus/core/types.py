from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


class TaskType(str, Enum):
    GENERAL = "general"
    CODING = "coding"
    MATH = "mathematics"
    RESEARCH = "research"
    DOCUMENT_ANALYSIS = "document_analysis"
    LONG_CONTEXT = "long_context_reasoning"
    VISION = "vision"
    AUDIO = "audio"
    TRANSLATION = "translation"
    HEALTH = "health"
    FITNESS = "fitness"
    SPORTS = "sports"
    DATA_ANALYSIS = "data_analysis"
    PLANNING = "planning"
    AGENT_EXECUTION = "agent_execution"
    PRIVATE_PERSONAL = "private_personal_data"
    COMPLEX_REASONING = "complex_reasoning"


class RoutingPolicy(str, Enum):
    MAX_QUALITY = "MAX_QUALITY"
    BALANCED = "BALANCED"
    LOW_COST = "LOW_COST"
    LOW_LATENCY = "LOW_LATENCY"
    LOCAL_ONLY = "LOCAL_ONLY"


@dataclass
class Message:
    role: Role
    content: str
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class GenerationResult:
    content: str
    model_used: str
    provider_name: str
    usage: Usage
    finish_reason: str | None = None
    latency_seconds: float | None = None
    raw: dict[str, Any] | None = None
    tool_calls: list[ToolCall] | None = None


@dataclass
class GenerationChunk:
    delta: str
    done: bool = False
    usage: Usage | None = None
    finish_reason: str | None = None


@dataclass
class ModelInfo:
    id: str
    provider: str
    kind: Literal["chat", "embedding"] = "chat"
    context_window: int = 4096
    capabilities: dict[str, float] = field(default_factory=dict)
    cost_per_1k_input_tokens: float | None = None
    cost_per_1k_output_tokens: float | None = None
    supports_tool_calling: bool = False


def new_session_id() -> str:
    return uuid.uuid4().hex


ChunkStream = AsyncIterator[GenerationChunk]
