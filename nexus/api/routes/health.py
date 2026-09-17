from __future__ import annotations

from fastapi import APIRouter, Request

from nexus.api.schemas import HealthResponse
from nexus.api.services import get_services
from nexus.core.provider_manager import ProviderManager

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    provider_manager: ProviderManager = get_services(request).provider_manager
    provider_statuses = await provider_manager.health_check_all()

    local_models: list[str] = []
    if provider_statuses.get("local"):
        local_provider = provider_manager.get("local")
        local_models = [model.id for model in await local_provider.list_models()]

    status = "ok" if provider_statuses and all(provider_statuses.values()) else "degraded"

    return HealthResponse(status=status, providers=provider_statuses, local_models=local_models)
