from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from nexus.api.routes import agents as agents_routes
from nexus.api.routes import chat, documents, health
from nexus.api.routes import evaluation as evaluation_routes
from nexus.api.routes import feedback as feedback_routes
from nexus.api.routes import generation as generation_routes
from nexus.api.routes import health_intel as health_intel_routes
from nexus.api.routes import memory as memory_routes
from nexus.api.routes import personal as personal_routes
from nexus.api.routes import psychology as psychology_routes
from nexus.api.routes import sports as sports_routes
from nexus.api.security import require_api_key
from nexus.api.services import build_services
from nexus.config.settings import get_settings
from nexus.logging_setup.logger import configure_logging, get_logger

logger = get_logger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(level=settings.server.log_level, json_output=settings.server.log_json)

    services = await build_services(settings)
    app.state.services = services
    yield

    await services.engine.dispose()


# The dashboard fetches NEXUS directly from the browser (:3000 -> :8100), so
# every request is cross-origin. Without CORSMiddleware, fetch() throws a
# generic TypeError and the UI reports NEXUS as unreachable even when it is
# running. See backend/api/main.py for the full explanation.
# allow_credentials=True requires an explicit origin list, not "*".
_default_cors_origins = "http://localhost:3000,http://127.0.0.1:3000"
NEXUS_CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("NEXUS_CORS_ALLOWED_ORIGINS", _default_cors_origins).split(",")
    if origin.strip()
]


# Explicit lists rather than "*": allow_credentials=True means a wildcard here
# would let any origin named in NEXUS_CORS_ALLOWED_ORIGINS drive authenticated,
# cookie-bearing requests with arbitrary headers.
NEXUS_CORS_ALLOWED_METHODS = ["GET", "POST", "PATCH", "DELETE", "OPTIONS"]
NEXUS_CORS_ALLOWED_HEADERS = ["Accept", "Authorization", "Content-Type", "X-API-Key"]


def create_app() -> FastAPI:
    app = FastAPI(
        title="NEXUS",
        version="0.7.0",
        lifespan=lifespan,
        # Applied on the constructor, not per-router, so a router added later
        # cannot accidentally ship unauthenticated. This surface can reach the
        # tool layer, so it matters more here than on the backend.
        dependencies=[Depends(require_api_key)],
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=NEXUS_CORS_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=NEXUS_CORS_ALLOWED_METHODS,
        allow_headers=NEXUS_CORS_ALLOWED_HEADERS,
    )
    app.include_router(chat.router, prefix="/api")
    app.include_router(health.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")
    app.include_router(memory_routes.router, prefix="/api")
    app.include_router(agents_routes.router, prefix="/api")
    app.include_router(personal_routes.router, prefix="/api")
    app.include_router(health_intel_routes.router, prefix="/api")
    app.include_router(generation_routes.router, prefix="/api")
    app.include_router(sports_routes.router, prefix="/api")
    app.include_router(psychology_routes.router, prefix="/api")
    app.include_router(evaluation_routes.router, prefix="/api")
    app.include_router(feedback_routes.router, prefix="/api")
    return app


app = create_app()
