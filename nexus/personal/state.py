from __future__ import annotations

import time
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from nexus.core.exceptions import InvalidSignalError
from nexus.memory.storage import PersonalSignalRecord, init_db, make_session_factory
from nexus.personal.dimensions import ALL_DIMENSIONS, DIMENSION_GROUPS, SOURCE_CONFIDENCE

_MIN_SAMPLES_FOR_FULL_CONFIDENCE = 5
_SECONDS_PER_DAY = 86400.0


@dataclass
class DimensionState:
    dimension: str
    value: float
    confidence: float
    sample_count: int
    latest_at: float | None


@dataclass
class PersonalState:
    user_id: str
    dimensions: dict[str, DimensionState]
    computed_at: float

    def group(self, group_name: str) -> dict[str, DimensionState]:
        members = DIMENSION_GROUPS.get(group_name, ())
        return {d: self.dimensions[d] for d in members if d in self.dimensions}


@dataclass
class _RawSignal:
    value: float
    source: str
    recorded_at: float


class PersonalStateEngine:
    """Derives current per-dimension state from append-only
    PersonalSignalRecord rows. Pure deterministic arithmetic — no LLM
    (principle 4): the "model" here is a recency- and source-confidence-
    weighted mean, not anything an LLM decides."""

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        half_life_days: float = 7.0,
        recent_window_days: float = 14.0,
    ) -> None:
        self._engine = engine
        self._session_factory: async_sessionmaker = make_session_factory(engine)
        self._half_life_days = half_life_days
        self._recent_window_days = recent_window_days

    async def init(self) -> None:
        await init_db(self._engine)

    async def record_signal(
        self, *, user_id: str, dimension: str, value: float, source: str, note: str = ""
    ) -> None:
        await self.record_signal_at(
            user_id=user_id,
            dimension=dimension,
            value=value,
            source=source,
            note=note,
            recorded_at=time.time(),
        )

    async def record_signal_at(
        self,
        *,
        user_id: str,
        dimension: str,
        value: float,
        source: str,
        recorded_at: float,
        note: str = "",
    ) -> None:
        """Records a signal with an explicit timestamp, for backfilling
        history that happened before it could be recorded — importing a
        wearable's export, or seeding a known series. record_signal() is
        this with recorded_at=now.
        """
        if dimension not in ALL_DIMENSIONS:
            raise InvalidSignalError(f"Unknown dimension: {dimension!r}")
        if not (0.0 <= value <= 1.0):
            raise InvalidSignalError(f"value must be within [0.0, 1.0], got {value!r}")
        if source not in SOURCE_CONFIDENCE:
            raise InvalidSignalError(
                f"Unknown source: {source!r} (expected one of {sorted(SOURCE_CONFIDENCE)})"
            )

        async with self._session_factory() as db:
            db.add(
                PersonalSignalRecord(
                    user_id=user_id,
                    dimension=dimension,
                    value=value,
                    source=source,
                    note=note,
                    recorded_at=recorded_at,
                )
            )
            await db.commit()

    async def get_state(self, user_id: str) -> PersonalState:
        now = time.time()
        signals_by_dimension = await self._recent_signals_by_dimension(user_id, now=now)

        dimensions: dict[str, DimensionState] = {}
        for dimension, signals in signals_by_dimension.items():
            dimensions[dimension] = self._derive_dimension_state(dimension, signals, now=now)

        return PersonalState(user_id=user_id, dimensions=dimensions, computed_at=now)

    async def get_signal_history(
        self, user_id: str, dimension: str, *, window_days: float
    ) -> list[tuple[float, float]]:
        """(recorded_at, value) pairs for a single dimension over the last
        window_days — the raw material trends.compute_trend() fits a line
        to. Unlike get_state(), this intentionally does NOT stop at
        recent_window_days: a trend needs to see the trajectory leading up
        """
        cutoff = time.time() - window_days * _SECONDS_PER_DAY
        async with self._session_factory() as db:
            result = await db.execute(
                select(PersonalSignalRecord).where(
                    PersonalSignalRecord.user_id == user_id,
                    PersonalSignalRecord.dimension == dimension,
                    PersonalSignalRecord.recorded_at >= cutoff,
                )
            )
            rows = result.scalars().all()
        return [(row.recorded_at, row.value) for row in rows]

    async def delete_all(self, user_id: str) -> None:
        async with self._session_factory() as db:
            await db.execute(
                delete(PersonalSignalRecord).where(PersonalSignalRecord.user_id == user_id)
            )
            await db.commit()

    async def _recent_signals_by_dimension(
        self, user_id: str, *, now: float
    ) -> dict[str, list[_RawSignal]]:
        cutoff = now - self._recent_window_days * _SECONDS_PER_DAY
        async with self._session_factory() as db:
            result = await db.execute(
                select(PersonalSignalRecord).where(
                    PersonalSignalRecord.user_id == user_id,
                    PersonalSignalRecord.recorded_at >= cutoff,
                )
            )
            rows = result.scalars().all()

        by_dimension: dict[str, list[_RawSignal]] = {}
        for row in rows:
            by_dimension.setdefault(row.dimension, []).append(
                _RawSignal(value=row.value, source=row.source, recorded_at=row.recorded_at)
            )
        return by_dimension

    def _derive_dimension_state(
        self, dimension: str, signals: list[_RawSignal], *, now: float
    ) -> DimensionState:
        weighted_sum = 0.0
        total_weight = 0.0
        source_confidences: list[float] = []
        latest_at = max(s.recorded_at for s in signals)

        for signal in signals:
            age_days = (now - signal.recorded_at) / _SECONDS_PER_DAY
            source_confidence = SOURCE_CONFIDENCE[signal.source]
            weight = source_confidence * (0.5 ** (age_days / self._half_life_days))
            weighted_sum += weight * signal.value
            total_weight += weight
            source_confidences.append(source_confidence)

        value = weighted_sum / total_weight
        sample_count = len(signals)
        mean_source_confidence = sum(source_confidences) / sample_count
        confidence = mean_source_confidence * min(
            1.0, sample_count / _MIN_SAMPLES_FOR_FULL_CONFIDENCE
        )

        return DimensionState(
            dimension=dimension,
            value=value,
            confidence=confidence,
            sample_count=sample_count,
            latest_at=latest_at,
        )
