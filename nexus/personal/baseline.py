from __future__ import annotations

import time
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from nexus.memory.storage import PersonalSignalRecord, make_session_factory

_SECONDS_PER_DAY = 86400.0


@dataclass
class Baseline:
    dimension: str
    value: float
    sample_count: int
    window_days: float


class BaselineCalculator:
    """Unweighted historical mean per dimension, used as the reference
    point WeaknessEngine compares current state against."""

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        window_days: float = 90.0,
        recent_window_days: float = 14.0,
        min_samples: int = 5,
    ) -> None:
        self._session_factory: async_sessionmaker = make_session_factory(engine)
        self._window_days = window_days
        self._recent_window_days = recent_window_days
        self._min_samples = min_samples

    async def get_baselines(self, user_id: str) -> dict[str, Baseline]:
        now = time.time()
        window_start = now - self._window_days * _SECONDS_PER_DAY
        # A baseline built from the same recent_window_days that
        # PersonalStateEngine treats as "current" would compare current
        # state against itself, making every deviation trivially small —
        # so history strictly excludes that trailing window.
        window_end = now - self._recent_window_days * _SECONDS_PER_DAY

        async with self._session_factory() as db:
            result = await db.execute(
                select(PersonalSignalRecord).where(
                    PersonalSignalRecord.user_id == user_id,
                    PersonalSignalRecord.recorded_at >= window_start,
                    PersonalSignalRecord.recorded_at < window_end,
                )
            )
            rows = result.scalars().all()

        by_dimension: dict[str, list[float]] = {}
        for row in rows:
            by_dimension.setdefault(row.dimension, []).append(row.value)

        baselines: dict[str, Baseline] = {}
        for dimension, values in by_dimension.items():
            if len(values) < self._min_samples:
                continue
            baselines[dimension] = Baseline(
                dimension=dimension,
                value=sum(values) / len(values),
                sample_count=len(values),
                window_days=self._window_days,
            )
        return baselines
