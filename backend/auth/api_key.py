"""Shared-secret API key authentication.

The key is read from the environment on every request rather than captured at
import time, so tests (and an operator rotating the secret) can change it
without rebuilding the app object.

When SSC_API_KEY is unset, authentication is DISABLED and every request is let
through. That keeps the documented local-dev flow in RUN.md working untouched.
Setting the variable is what turns enforcement on; there is deliberately no
partially-enforced mode.
"""

from __future__ import annotations

import os
import re
import secrets

from fastapi import HTTPException, Request, status
from fastapi.security import APIKeyHeader

API_KEY_ENV = "SSC_API_KEY"
API_KEY_HEADER = "X-API-Key"

# auto_error=False: a missing header must fall through to our own check so that
# the key-not-configured case can return 200 rather than 403.
_header_scheme = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)

# Kept open even when a key is configured, so container HEALTHCHECKs and load
# balancer probes do not need to carry the secret. /health returns no data.
PUBLIC_PATHS = frozenset({"/health"})

# <video src="..."> cannot attach request headers, so the two media routes the
# dashboard plays directly also accept the key as ?api_key=. Only GET, only
# these paths: everywhere else the header is required, so the key is not
# routinely written into access logs.
MEDIA_PATH_RE = re.compile(r"^/api/videos/[^/]+/(file|processed)$")
MEDIA_QUERY_PARAM = "api_key"


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
    if presented is None and request.method == "GET" and MEDIA_PATH_RE.match(request.url.path):
        presented = request.query_params.get(MEDIA_QUERY_PARAM)
    # compare_digest over a constant-time comparison of equal-length inputs;
    # the None guard is separate so we never hand it a non-str.
    if presented is None or not secrets.compare_digest(presented, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing or invalid {API_KEY_HEADER} header.",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )


__all__ = ["API_KEY_ENV", "API_KEY_HEADER", "MEDIA_QUERY_PARAM", "PUBLIC_PATHS", "auth_enabled", "require_api_key"]
