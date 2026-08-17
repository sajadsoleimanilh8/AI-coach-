"""
Redis caching layer for PlayerMetric and TeamMetric queries.
Implementation Spec §5.2.
"""

from __future__ import annotations
import json
import logging
import os
import redis

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_redis_client = None

def get_redis():
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        except Exception as e:
            logger.warning(f"Failed to connect to Redis at {REDIS_URL}: {e}")
            _redis_client = None
    return _redis_client

def get_cached_metrics(match_id: str, scope: str, schema_version: str = "v3") -> dict | list | None:
    r = get_redis()
    if not r:
        return None
    key = f"metric_cache:{match_id}:{scope}:{schema_version}"
    try:
        data = r.get(key)
        if data:
            return json.loads(data)
    except Exception as e:
        logger.warning(f"Redis get error for key {key}: {e}")
    return None

def set_cached_metrics(match_id: str, scope: str, value: dict | list, schema_version: str = "v3") -> None:
    r = get_redis()
    if not r:
        return
    key = f"metric_cache:{match_id}:{scope}:{schema_version}"
    try:
        r.set(key, json.dumps(value, default=str))
    except Exception as e:
        logger.warning(f"Redis set error for key {key}: {e}")

def invalidate_match_cache(match_id: str) -> None:
    r = get_redis()
    if not r:
        return
    try:
        keys = r.keys(f"metric_cache:{match_id}:*")
        if keys:
            r.delete(*keys)
    except Exception as e:
        logger.warning(f"Redis invalidation error for match_id {match_id}: {e}")
