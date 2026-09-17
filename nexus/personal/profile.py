from __future__ import annotations

import json
import time
from typing import Any

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from nexus.memory.storage import PersonalProfileRecord, init_db, make_session_factory


class ProfileStore:
    """CRUD over PersonalProfileRecord — goals/preferences/constraints,
    scoped by user_id. Separate from PersonalStateEngine (which owns the
    append-only signal history) the same way LongTermMemoryStore is kept
    separate from ShortTermMemoryStore: different shape, different update
    pattern (whole-document replace, not append-only evidence)."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._session_factory: async_sessionmaker = make_session_factory(engine)

    async def init(self) -> None:
        await init_db(self._engine)

    async def get_profile(self, user_id: str) -> dict[str, Any]:
        async with self._session_factory() as db:
            row = await db.get(PersonalProfileRecord, user_id)
        return json.loads(row.profile_json) if row is not None else {}

    async def set_profile(self, user_id: str, profile: dict[str, Any]) -> None:
        now = time.time()
        profile_json = json.dumps(profile)
        async with self._session_factory() as db:
            row = await db.get(PersonalProfileRecord, user_id)
            if row is None:
                db.add(
                    PersonalProfileRecord(
                        user_id=user_id, profile_json=profile_json, updated_at=now
                    )
                )
            else:
                row.profile_json = profile_json
                row.updated_at = now
            await db.commit()

    async def delete_profile(self, user_id: str) -> None:
        async with self._session_factory() as db:
            await db.execute(
                delete(PersonalProfileRecord).where(PersonalProfileRecord.user_id == user_id)
            )
            await db.commit()
