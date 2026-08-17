from __future__ import annotations

import json
import time
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from nexus.memory.storage import LongTermFactRecord, init_db, make_session_factory


class LongTermMemoryStore:
    """CRUD over LongTermFactRecord, scoped by user_id."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._session_factory: async_sessionmaker = make_session_factory(engine)

    async def init(self) -> None:
        await init_db(self._engine)

    async def set_fact(self, user_id: str, key: str, value: Any) -> None:
        now = time.time()
        value_json = json.dumps(value)
        async with self._session_factory() as db:
            result = await db.execute(
                select(LongTermFactRecord).where(
                    LongTermFactRecord.user_id == user_id, LongTermFactRecord.key == key
                )
            )
            row = result.scalar_one_or_none()
            if row is None:
                db.add(
                    LongTermFactRecord(
                        user_id=user_id, key=key, value_json=value_json, updated_at=now
                    )
                )
            else:
                row.value_json = value_json
                row.updated_at = now
            await db.commit()

    async def get_facts(self, user_id: str) -> dict[str, Any]:
        async with self._session_factory() as db:
            result = await db.execute(
                select(LongTermFactRecord).where(LongTermFactRecord.user_id == user_id)
            )
            rows = result.scalars().all()
        return {row.key: json.loads(row.value_json) for row in rows}

    async def delete_fact(self, user_id: str, key: str) -> None:
        async with self._session_factory() as db:
            await db.execute(
                delete(LongTermFactRecord).where(
                    LongTermFactRecord.user_id == user_id, LongTermFactRecord.key == key
                )
            )
            await db.commit()

    async def clear(self, user_id: str) -> None:
        async with self._session_factory() as db:
            await db.execute(delete(LongTermFactRecord).where(LongTermFactRecord.user_id == user_id))
            await db.commit()
