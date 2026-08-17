from __future__ import annotations

import json
import time

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from nexus.core.types import Message
from nexus.memory.storage import InteractionRecord, init_db, make_session_factory

_SECONDS_PER_DAY = 86400.0


class InteractionLogger:
    """Persists full request/response records for later training-data
    mining, following CostTracker's pattern (engine, async init(), async
    record()).
    """

    def __init__(self, engine: AsyncEngine, *, enabled: bool = False, retention_days: int = 180) -> None:
        self._engine = engine
        self._session_factory: async_sessionmaker = make_session_factory(engine)
        self._enabled = enabled
        self._retention_days = retention_days

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def init(self) -> None:
        await init_db(self._engine)

    async def record(
        self,
        *,
        session_id: str,
        user_id: str,
        task_type: str,
        privacy_level: str,
        prompt_messages: list[Message],
        response_text: str,
        model_id: str,
        provider_name: str,
        tools_used: list[str] | None = None,
        verification_band: str = "",
        verification_score: float | None = None,
    ) -> int | None:
        if not self._enabled:
            return None

        record = InteractionRecord(
            session_id=session_id,
            user_id=user_id,
            task_type=task_type,
            privacy_level=privacy_level,
            prompt_json=json.dumps(
                [{"role": m.role, "content": m.content} for m in prompt_messages]
            ),
            response_text=response_text,
            model_id=model_id,
            provider_name=provider_name,
            tools_used_json=json.dumps(tools_used or []),
            verification_band=verification_band,
            verification_score=verification_score,
            user_feedback=None,
            created_at=time.time(),
        )
        async with self._session_factory() as db:
            db.add(record)
            await db.commit()
            return record.id

    async def set_feedback(self, interaction_id: int, feedback: int) -> bool:
        """Returns False when no such interaction exists, so the API can
        answer 404 rather than silently accepting a rating that landed
        nowhere. Feedback is accepted even when logging is disabled — the
        record it points at was written while logging was on, and refusing
        """
        async with self._session_factory() as db:
            existing = await db.get(InteractionRecord, interaction_id)
            if existing is None:
                return False
            await db.execute(
                update(InteractionRecord)
                .where(InteractionRecord.id == interaction_id)
                .values(user_feedback=feedback)
            )
            await db.commit()
        return True

    async def count(self) -> int:
        async with self._session_factory() as db:
            result = await db.execute(select(InteractionRecord.id))
            return len(result.scalars().all())

    async def purge_expired(self) -> int:
        """Deletes records past interaction_retention_days. Returns how many
        were removed."""
        cutoff = time.time() - self._retention_days * _SECONDS_PER_DAY
        async with self._session_factory() as db:
            result = await db.execute(
                select(InteractionRecord).where(InteractionRecord.created_at < cutoff)
            )
            expired = result.scalars().all()
            for row in expired:
                await db.delete(row)
            await db.commit()
        return len(expired)
