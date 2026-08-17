from __future__ import annotations

from nexus.config.settings import load_model_registry
from nexus.core.types import ModelInfo


def list_models() -> list[ModelInfo]:
    return [
        ModelInfo(
            id=entry["id"],
            provider=entry.get("provider", "local"),
            kind=entry.get("kind", "chat"),
            context_window=entry.get("context_window", 4096),
            capabilities=entry.get("capabilities", {}),
            cost_per_1k_input_tokens=entry.get("cost_per_1k_input_tokens"),
            cost_per_1k_output_tokens=entry.get("cost_per_1k_output_tokens"),
            supports_tool_calling=entry.get("supports_tool_calling", False),
        )
        for entry in load_model_registry()
    ]


def get_model(model_id: str) -> ModelInfo | None:
    for model in list_models():
        if model.id == model_id:
            return model
    return None
