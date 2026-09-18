"""The metrics cache must degrade CHEAPLY when Redis is down.

Regression: each call tried Redis with default timeouts, and a refused
connection via "localhost" took ~4 s. Metrics routes read and write the cache,
so a stopped Redis added ~8 s to every metrics request with no visible error.
"""

import time

import pytest
import redis

from backend.api import cache


class _DeadRedis:
    """Stands in for a client whose server is down; counts attempts."""

    def __init__(self):
        self.calls = 0

    def _fail(self, *args, **kwargs):
        self.calls += 1
        raise redis.ConnectionError("connection refused")

    get = set = keys = delete = _fail


@pytest.fixture
def dead_redis(monkeypatch):
    fake = _DeadRedis()
    monkeypatch.setattr(cache, "_redis_client", fake)
    monkeypatch.setattr(cache, "_down_until", 0.0)
    return fake


def test_a_down_redis_is_a_cache_miss_not_an_error(dead_redis):
    assert cache.get_cached_metrics("m", "scope") is None
    cache.set_cached_metrics("m", "scope", {"a": 1})  # must not raise
    cache.invalidate_match_cache("m")  # must not raise


def test_after_one_failure_redis_is_skipped_entirely(dead_redis):
    """The outage costs one attempt, then calls return immediately."""
    cache.get_cached_metrics("m", "scope")
    assert dead_redis.calls == 1
    for _ in range(20):
        cache.get_cached_metrics("m", "scope")
        cache.set_cached_metrics("m", "scope", {"a": 1})
    assert dead_redis.calls == 1, "every call after the first should skip Redis"


def test_redis_is_retried_after_the_backoff_window(dead_redis, monkeypatch):
    monkeypatch.setattr(cache, "REDIS_RETRY_AFTER_S", 0.05)
    cache.get_cached_metrics("m", "scope")
    time.sleep(0.1)
    cache.get_cached_metrics("m", "scope")
    assert dead_redis.calls == 2, "Redis should be tried again once the window passes"


def test_outage_is_logged_once_not_per_request(dead_redis, caplog):
    with caplog.at_level("WARNING", logger=cache.logger.name):
        for _ in range(10):
            cache.get_cached_metrics("m", "scope")
    assert len([r for r in caplog.records if "unavailable" in r.getMessage()]) == 1


def test_a_real_unreachable_server_fails_fast(monkeypatch):
    """End to end with a real client: port 1 on loopback refuses immediately,
    and the configured short timeouts bound the worst case."""
    monkeypatch.setattr(cache, "REDIS_URL", "redis://127.0.0.1:1/0")
    monkeypatch.setattr(cache, "_redis_client", None)
    monkeypatch.setattr(cache, "_down_until", 0.0)
    start = time.monotonic()
    for _ in range(5):
        assert cache.get_cached_metrics("m", "scope") is None
    assert time.monotonic() - start < 3.0
