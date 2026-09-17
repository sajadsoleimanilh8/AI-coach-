"""Shared-secret API key authentication.

The key is read from the environment on every request rather than captured at
import time, so tests (and an operator rotating the secret) can change it
without rebuilding the app object.

When NEXUS_API_KEY is unset, authentication is DISABLED and every request is let
through. That keeps the documented local-dev flow in RUN.md working untouched.
Setting the variable is what turns enforcement on; there is deliberately no
partially-enforced mode.

This deliberately mirrors backend/auth/api_key.py rather than importing it.
The two services are independently deployable -- nexus/Dockerfile copies only
nexus/, backend/Dockerfile copies only backend/ and ai/ -- so a cross-package
import would break both images. They also want separate secrets: NEXUS_API_KEY
gates the LLM surface, SSC_API_KEY gates the video/metrics surface. If they are
ever merged into one deployable, collapse these two modules.
"""

from __future__ import annotations

import os
import secrets

from fastapi import HTTPException, Request, status
from fastapi.security import APIKeyHeader

API_KEY_ENV = "NEXUS_API_KEY"
API_KEY_HEADER = "X-API-Key"

# auto_error=False: a missing header must fall through to our own check so that
# the key-not-configured case can return 200 rather than 403.
_header_scheme = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)

# Kept open even when a key is configured, so container HEALTHCHECKs and load
# balancer probes do not need to carry the secret. /health returns no data.
PUBLIC_PATHS = frozenset({"/api/health"})


def auth_enabled() -> bool:
    return bool(os.getenv(API_KEY_ENV, "").strip())


def require_api_key(request: Request) -> None:
    """App-wide dependency. Raises 401 unless the request carries the key.

    Registered once on the FastAPI constructor rather than per-router, so a new
    router cannot be added later and accidentally ship unauthenticated.
    """
    expected = os.getenv(API_KEY_ENV, "").strip()
    if not expected:
        return
    if request.url.path in PUBLIC_PATHS:
        return

    presented = request.headers.get(API_KEY_HEADER)
    # compare_digest over a constant-time comparison of equal-length inputs;
    # the None guard is separate so we never hand it a non-str.
    if presented is None or not secrets.compare_digest(presented, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing or invalid {API_KEY_HEADER} header.",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )


__all__ = ["API_KEY_ENV", "API_KEY_HEADER", "PUBLIC_PATHS", "auth_enabled", "require_api_key", "_header_scheme"]
