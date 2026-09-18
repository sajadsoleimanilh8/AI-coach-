"""
Redis caching layer for PlayerMetric and TeamMetric queries.
Implementation Spec §5.2.

The cache is optional: every function here degrades to "no cache" on any
Redis problem and never fails the request. Degrading also has to be CHEAP.
Previously each call tried to reach Redis with the client's default timeouts,
and via "localhost" (which tries IPv6 first on Windows) a refused connection
cost ~4 s -- twice per metrics request (read + write), so a stopped Redis made
the whole API ~8 s slower per call without any visible error. Now a
connection failure marks Redis down for REDIS_RETRY_AFTER_S seconds, during
which calls skip it immediately, and the outage is logged once, not per call.
"""

from __future__ import annotations

import json
import logging
import os
import time

import redis

logger = logging.getLogger(__name__)

# 127.0.0.1, not localhost: see CLAUDE.md -- localhost resolves to ::1 first
# on Windows and every connection attempt to a stopped server pays for both.
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
REDIS_TIMEOUT_S = float(os.getenv("REDIS_TIMEOUT_S", "0.25"))
REDIS_RETRY_AFTER_S = float(os.getenv("REDIS_RETRY_AFTER_S", "30"))

_redis_client = None
_down_until = 0.0

_UNAVAILABLE = (redis.ConnectionError, redis.TimeoutError, OSError)


def get_redis():
    """The client, or None while Redis is known to be unreachable."""
    global _redis_client
    if time.monotonic() < _down_until:
        return None
    if _redis_client is None:
        try:
            _redis_client = redis.Redis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=REDIS_TIMEOUT_S,
                socket_timeout=REDIS_TIMEOUT_S,
            )
        except Exception as e:  # noqa: BLE001 - cache is optional; any Redis failure degrades to no-cache
            logger.warning("Invalid Redis configuration %s: %s", REDIS_URL, e)
            _mark_down()
            return None
    return _redis_client


def _mark_down(error: Exception | None = None) -> None:
    """Skip Redis for REDIS_RETRY_AFTER_S; log only on the transition to down."""
    global _down_until
    was_up = time.monotonic() >= _down_until
    _down_until = time.monotonic() + REDIS_RETRY_AFTER_S
    if was_up and error is not None:
        logger.warning(
            "Redis at %s unavailable (%s); serving uncached for the next %.0fs",
            REDIS_URL, error, REDIS_RETRY_AFTER_S,
        )


def get_cached_metrics(match_id: str, scope: str, schema_version: str = "v3") -> dict | list | None:
    r = get_redis()
    if not r:
        return None
    key = f"metric_cache:{match_id}:{scope}:{schema_version}"
    try:
        data = r.get(key)
        if data:
            return json.loads(data)
    except _UNAVAILABLE as e:
        _mark_down(e)
    except Exception as e:  # noqa: BLE001 - cache miss on any Redis error, never fail the request
        logger.warning("Redis get error for key %s: %s", key, e)
    return None


def set_cached_metrics(match_id: str, scope: str, value: dict | list, schema_version: str = "v3") -> None:
    r = get_redis()
    if not r:
        return
    key = f"metric_cache:{match_id}:{scope}:{schema_version}"
    try:
        r.set(key, json.dumps(value, default=str))
    except _UNAVAILABLE as e:
        _mark_down(e)
    except Exception as e:  # noqa: BLE001 - cache write is best-effort
        logger.warning("Redis set error for key %s: %s", key, e)


def invalidate_match_cache(match_id: str) -> None:
    r = get_redis()
    if not r:
        return
    try:
        keys = r.keys(f"metric_cache:{match_id}:*")
        if keys:
            r.delete(*keys)
    except _UNAVAILABLE as e:
        _mark_down(e)
    except Exception as e:  # noqa: BLE001 - cache invalidation is best-effort
        logger.warning("Redis invalidation error for match_id %s: %s", match_id, e)
