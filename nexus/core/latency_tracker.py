from __future__ import annotations

import time
from collections import defaultdict, deque

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from nexus.memory.storage import LatencyRecord, init_db, make_session_factory

_DEFAULT_WINDOW_SIZE = 50
_DEFAULT_MIN_SAMPLES = 5


class LatencyTracker:
    """Persists every generation's latency (observability) and keeps a
    bounded in-memory rolling window per (provider_name, model_id) so
    ModelRouter's synchronous route() path can read a p50 estimate without
    an async DB round-trip during routing.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        window_size: int = _DEFAULT_WINDOW_SIZE,
        min_samples: int = _DEFAULT_MIN_SAMPLES,
    ) -> None:
        self._engine = engine
        self._session_factory: async_sessionmaker = make_session_factory(engine)
        self._min_samples = min_samples
        self._windows: dict[tuple[str, str], deque[float]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

    async def init(self) -> None:
        await init_db(self._engine)

    async def record(
        self,
        *,
        session_id: str,
        provider_name: str,
        model_id: str,
        latency_seconds: float,
    ) -> None:
        self._windows[(provider_name, model_id)].append(latency_seconds)
        async with self._session_factory() as db:
            db.add(
                LatencyRecord(
                    session_id=session_id,
                    provider_name=provider_name,
                    model_id=model_id,
                    latency_seconds=latency_seconds,
                    created_at=time.time(),
                )
            )
            await db.commit()

    def p50(self, provider_name: str, model_id: str) -> float | None:
        samples = self._windows.get((provider_name, model_id))
        if samples is None or len(samples) < self._min_samples:
            return None
        ordered = sorted(samples)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2

    def sample_count(self, provider_name: str, model_id: str) -> int:
        samples = self._windows.get((provider_name, model_id))
        return len(samples) if samples is not None else 0
