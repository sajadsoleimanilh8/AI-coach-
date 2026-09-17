from __future__ import annotations

import asyncio

from nexus.core.exceptions import ModelNotFoundError, ProviderUnavailableError
from nexus.core.providers import AIProvider
from nexus.logging_setup.logger import get_logger
from nexus.models.registry import get_model

logger = get_logger("core.provider_manager")


class ProviderManager:
    """Owns all registered AIProvider instances (local + cloud).

    Dict keys are the model registry's provider identifiers ("local",
    "openai", "anthropic" — see models.yaml), not necessarily each
    provider's own `.name` attribute (OllamaRuntime keeps `name = "ollama"`
    from Phase 1), so routing decisions and registry lookups always agree on
    the same vocabulary.
    """

    def __init__(self, providers: dict[str, AIProvider]) -> None:
        self._providers = providers

    def get(self, provider_name: str) -> AIProvider:
        try:
            return self._providers[provider_name]
        except KeyError as exc:
            raise ProviderUnavailableError(
                f"Provider '{provider_name}' is not registered."
            ) from exc

    def has_provider(self, provider_name: str) -> bool:
        return provider_name in self._providers

    async def health_check_all(self) -> dict[str, bool]:
        names = list(self._providers.keys())
        results = await asyncio.gather(
            *(self._providers[name].health_check() for name in names),
            return_exceptions=True,
        )
        statuses: dict[str, bool] = {}
        for name, result in zip(names, results):
            if isinstance(result, BaseException):
                logger.warning("health_check raised for provider=%s: %s", name, result)
                statuses[name] = False
            else:
                statuses[name] = bool(result)
        return statuses

    def provider_for_model(self, model_id: str) -> AIProvider:
        model_info = get_model(model_id)
        if model_info is None:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry.")
        return self.get(model_info.provider)
