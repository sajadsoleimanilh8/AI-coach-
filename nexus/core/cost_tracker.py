from __future__ import annotations

import time

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from nexus.core.types import ModelInfo, Usage
from nexus.memory.storage import CostRecord, init_db, make_session_factory


class CostTracker:
    """Minimal cost ledger, reusing the same SQLite engine as conversation memory."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._session_factory: async_sessionmaker = make_session_factory(engine)

    async def init(self) -> None:
        await init_db(self._engine)

    async def record(
        self,
        *,
        session_id: str,
        provider_name: str,
        model_id: str,
        usage: Usage,
        model_info: ModelInfo | None,
    ) -> float:
        cost_usd = self._compute_cost(usage, model_info)
        async with self._session_factory() as db:
            db.add(
                CostRecord(
                    session_id=session_id,
                    provider_name=provider_name,
                    model_id=model_id,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    cost_usd=cost_usd,
                    created_at=time.time(),
                )
            )
            await db.commit()
        return cost_usd

    async def session_total(self, session_id: str) -> float:
        async with self._session_factory() as db:
            result = await db.execute(
                select(func.sum(CostRecord.cost_usd)).where(CostRecord.session_id == session_id)
            )
            return result.scalar() or 0.0

    @staticmethod
    def _compute_cost(usage: Usage, model_info: ModelInfo | None) -> float:
        if model_info is None:
            return 0.0
        input_rate = model_info.cost_per_1k_input_tokens
        output_rate = model_info.cost_per_1k_output_tokens
        if not input_rate and not output_rate:
            return 0.0
        cost = (usage.prompt_tokens / 1000) * (input_rate or 0.0) + (
            usage.completion_tokens / 1000
        ) * (output_rate or 0.0)
        return round(cost, 8)
