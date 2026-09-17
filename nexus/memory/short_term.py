from __future__ import annotations

import time

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from nexus.core.exceptions import SessionNotFoundError
from nexus.core.memory import MemoryStore
from nexus.core.types import Message, new_session_id
from nexus.memory.storage import MessageRecord, SessionRecord, init_db, make_session_factory


class ShortTermMemoryStore(MemoryStore):
    """SQLite-backed conversation memory with a sliding window and TTL expiry.

    Swappable for a Postgres- or Redis-backed store later without touching
    callers, since everything routes through the `MemoryStore` interface.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        window_size: int = 20,
        ttl_seconds: float = 24 * 3600,
    ) -> None:
        self._engine = engine
        self._session_factory: async_sessionmaker = make_session_factory(engine)
        self._window_size = window_size
        self._ttl_seconds = ttl_seconds

    async def init(self) -> None:
        await init_db(self._engine)

    async def create_session(self) -> str:
        session_id = new_session_id()
        now = time.time()
        async with self._session_factory() as db:
            db.add(SessionRecord(id=session_id, created_at=now, last_active_at=now))
            await db.commit()
        return session_id

    async def add_message(self, session_id: str, message: Message) -> None:
        now = time.time()
        async with self._session_factory() as db:
            session_row = await db.get(SessionRecord, session_id)
            if session_row is None:
                session_row = SessionRecord(id=session_id, created_at=now, last_active_at=now)
                db.add(session_row)
            else:
                session_row.last_active_at = now

            db.add(
                MessageRecord(
                    session_id=session_id,
                    role=message.role,
                    content=message.content,
                    created_at=message.created_at,
                )
            )
            await db.commit()

        await self._trim_to_window(session_id)

    async def _trim_to_window(self, session_id: str) -> None:
        async with self._session_factory() as db:
            result = await db.execute(
                select(MessageRecord.id)
                .where(MessageRecord.session_id == session_id)
                # id (autoincrement) breaks ties deterministically when two
                # messages land on the same created_at float — time.time()'s
                # resolution isn't fine enough to guarantee distinct values
                # for fast successive inserts, especially on Windows.
                .order_by(MessageRecord.created_at.desc(), MessageRecord.id.desc())
                .offset(self._window_size)
            )
            stale_ids = [row[0] for row in result.all()]
            if stale_ids:
                await db.execute(delete(MessageRecord).where(MessageRecord.id.in_(stale_ids)))
                await db.commit()

    async def get_history(self, session_id: str) -> list[Message]:
        async with self._session_factory() as db:
            session_row = await db.get(SessionRecord, session_id)
            if session_row is None:
                raise SessionNotFoundError(f"No session found for id={session_id}")

            result = await db.execute(
                select(MessageRecord)
                .where(MessageRecord.session_id == session_id)
                .order_by(MessageRecord.created_at.asc(), MessageRecord.id.asc())
            )
            return [
                Message(role=row.role, content=row.content, created_at=row.created_at)
                for row in result.scalars().all()
            ]

    async def clear_session(self, session_id: str) -> None:
        # Core `delete()` bypasses the ORM's cascade config, and SQLite
        # doesn't enforce FK constraints by default, so messages are
        # removed explicitly to avoid orphaning them.
        async with self._session_factory() as db:
            await db.execute(delete(MessageRecord).where(MessageRecord.session_id == session_id))
            await db.execute(delete(SessionRecord).where(SessionRecord.id == session_id))
            await db.commit()

    async def expire_sessions(self) -> int:
        cutoff = time.time() - self._ttl_seconds
        async with self._session_factory() as db:
            result = await db.execute(
                select(SessionRecord.id).where(SessionRecord.last_active_at < cutoff)
            )
            expired_ids = [row[0] for row in result.all()]
            if expired_ids:
                await db.execute(
                    delete(MessageRecord).where(MessageRecord.session_id.in_(expired_ids))
                )
                await db.execute(delete(SessionRecord).where(SessionRecord.id.in_(expired_ids)))
                await db.commit()
        return len(expired_ids)
