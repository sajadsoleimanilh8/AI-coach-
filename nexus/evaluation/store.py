from __future__ import annotations

import json

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from nexus.evaluation.types import CaseOutcome, EvalRun, SuiteResult
from nexus.memory.storage import EvalCaseRecord, EvalRunRecord, init_db, make_session_factory


class EvalStore:
    """Persists EvalRun results, following CostTracker's exact pattern
    (async init(), same engine/session-factory setup)."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._session_factory: async_sessionmaker = make_session_factory(engine)

    async def init(self) -> None:
        await init_db(self._engine)

    async def save_run(self, run: EvalRun) -> None:
        summary = [
            {
                "suite": s.suite,
                "pass_rate": s.pass_rate,
                "mean_score": s.mean_score,
                "total_cost_usd": s.total_cost_usd,
                "total_latency_seconds": s.total_latency_seconds,
                "case_count": len(s.outcomes),
            }
            for s in run.suites
        ]
        async with self._session_factory() as db:
            db.add(
                EvalRunRecord(
                    run_id=run.run_id,
                    started_at=run.started_at,
                    finished_at=run.finished_at,
                    git_sha=run.git_sha or "",
                    config_json=json.dumps(run.config_snapshot),
                    summary_json=json.dumps(summary),
                    provider_mode=run.provider_mode,
                    pinned_model_id=run.pinned_model_id or "",
                )
            )
            for suite in run.suites:
                for outcome in suite.outcomes:
                    db.add(
                        EvalCaseRecord(
                            run_id=run.run_id,
                            suite=suite.suite,
                            case_id=outcome.case_id,
                            passed=outcome.passed,
                            score=outcome.score,
                            detail=outcome.detail,
                            latency_seconds=outcome.latency_seconds,
                            cost_usd=outcome.cost_usd,
                        )
                    )
            await db.commit()

    async def get_run(self, run_id: str) -> EvalRun | None:
        async with self._session_factory() as db:
            run_row = await db.get(EvalRunRecord, run_id)
            if run_row is None:
                return None
            result = await db.execute(
                select(EvalCaseRecord).where(EvalCaseRecord.run_id == run_id)
            )
            case_rows = result.scalars().all()

        by_suite: dict[str, list[CaseOutcome]] = {}
        for row in case_rows:
            # `actual` (the detailed structured output) is deliberately
            # NOT part of EvalCaseRecord's columns — only the scored
            # summary fields round-trip through storage; a live run's
            # in-memory EvalRun carries the full detail, a reloaded one
            # carries the scoring history, which is what regression
            # comparison actually needs.
            by_suite.setdefault(row.suite, []).append(
                CaseOutcome(
                    case_id=row.case_id, passed=row.passed, score=row.score, actual={},
                    detail=row.detail, latency_seconds=row.latency_seconds, cost_usd=row.cost_usd,
                )
            )

        suites = [SuiteResult.from_outcomes(suite, outcomes) for suite, outcomes in by_suite.items()]
        return EvalRun(
            run_id=run_row.run_id,
            started_at=run_row.started_at,
            finished_at=run_row.finished_at,
            suites=suites,
            config_snapshot=json.loads(run_row.config_json),
            git_sha=run_row.git_sha or None,
            # Rows predating the column read back as NULL; "unknown" is the
            # honest label, and compare_runs() refuses to diff it.
            provider_mode=run_row.provider_mode or "unknown",
            pinned_model_id=run_row.pinned_model_id or None,
        )

    async def latest_run(self, suite: str | None = None) -> EvalRun | None:
        async with self._session_factory() as db:
            result = await db.execute(
                select(EvalRunRecord).order_by(EvalRunRecord.started_at.desc())
            )
            rows = result.scalars().all()

        for row in rows:
            if suite is None:
                return await self.get_run(row.run_id)
            summary = json.loads(row.summary_json)
            if any(entry["suite"] == suite for entry in summary):
                return await self.get_run(row.run_id)
        return None

    async def latest_run_for_provider_mode(self, provider_mode: str) -> EvalRun | None:
        """Newest run recorded in the SAME provider mode, or None.

        Separate from latest_run() because a regression comparison is only
        meaningful within one mode — see compare_runs(), which refuses
        across them. "unknown" never matches, including another "unknown":
        two untagged runs may have come from different modes, and pairing
        them would rebuild exactly the false comparison this avoids.
        """
        if provider_mode == "unknown":
            return None
        async with self._session_factory() as db:
            result = await db.execute(
                select(EvalRunRecord)
                .where(EvalRunRecord.provider_mode == provider_mode)
                .order_by(EvalRunRecord.started_at.desc())
            )
            row = result.scalars().first()
        return await self.get_run(row.run_id) if row is not None else None

    async def list_runs(self, limit: int = 20) -> list[EvalRun]:
        async with self._session_factory() as db:
            result = await db.execute(
                select(EvalRunRecord).order_by(EvalRunRecord.started_at.desc()).limit(limit)
            )
            rows = result.scalars().all()
        runs = []
        for row in rows:
            run = await self.get_run(row.run_id)
            if run is not None:
                runs.append(run)
        return runs

    async def delete_run(self, run_id: str) -> None:
        async with self._session_factory() as db:
            await db.execute(delete(EvalCaseRecord).where(EvalCaseRecord.run_id == run_id))
            await db.execute(delete(EvalRunRecord).where(EvalRunRecord.run_id == run_id))
            await db.commit()
