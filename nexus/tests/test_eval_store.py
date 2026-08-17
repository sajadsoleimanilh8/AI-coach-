from __future__ import annotations

import pytest

from nexus.evaluation.store import EvalStore
from nexus.evaluation.types import CaseOutcome, EvalRun, SuiteResult
from nexus.memory.storage import create_async_db_engine


async def _make_store(tmp_path) -> EvalStore:
    engine = create_async_db_engine(str(tmp_path / "nexus.db"))
    store = EvalStore(engine)
    await store.init()
    return store


def _run(run_id: str, *, started_at: float = 100.0) -> EvalRun:
    outcomes = [
        CaseOutcome(
            case_id="c1", passed=True, score=1.0, actual={"x": 1}, detail="ok",
            latency_seconds=0.1, cost_usd=0.001,
        ),
        CaseOutcome(
            case_id="c2", passed=False, score=0.0, actual={"x": 2}, detail="failed",
            latency_seconds=0.2, cost_usd=0.002,
        ),
    ]
    return EvalRun(
        run_id=run_id, started_at=started_at, finished_at=started_at + 5.0,
        suites=[SuiteResult.from_outcomes("routing", outcomes)],
        config_snapshot={"routing": {"default_policy": "BALANCED"}}, git_sha="abc123",
    )


@pytest.mark.asyncio
async def test_save_then_get_round_trip_preserves_scored_fields(tmp_path) -> None:
    store = await _make_store(tmp_path)
    run = _run("run-1")

    await store.save_run(run)
    loaded = await store.get_run("run-1")

    assert loaded is not None
    assert loaded.run_id == "run-1"
    assert loaded.started_at == run.started_at
    assert loaded.finished_at == run.finished_at
    assert loaded.git_sha == "abc123"
    assert loaded.config_snapshot == run.config_snapshot

    assert len(loaded.suites) == 1
    suite = loaded.suites[0]
    assert suite.suite == "routing"
    assert suite.pass_rate == pytest.approx(0.5)
    assert suite.mean_score == pytest.approx(0.5)
    assert suite.total_cost_usd == pytest.approx(0.003)
    assert suite.total_latency_seconds == pytest.approx(0.3)

    outcomes_by_id = {o.case_id: o for o in suite.outcomes}
    assert outcomes_by_id["c1"].passed is True
    assert outcomes_by_id["c1"].score == pytest.approx(1.0)
    assert outcomes_by_id["c1"].detail == "ok"
    assert outcomes_by_id["c2"].passed is False
    assert outcomes_by_id["c2"].detail == "failed"


@pytest.mark.asyncio
async def test_get_unknown_run_returns_none(tmp_path) -> None:
    store = await _make_store(tmp_path)
    assert await store.get_run("does-not-exist") is None


@pytest.mark.asyncio
async def test_latest_run_returns_most_recent(tmp_path) -> None:
    store = await _make_store(tmp_path)
    await store.save_run(_run("run-old", started_at=100.0))
    await store.save_run(_run("run-new", started_at=200.0))

    latest = await store.latest_run()

    assert latest.run_id == "run-new"


@pytest.mark.asyncio
async def test_latest_run_scoped_to_a_suite(tmp_path) -> None:
    store = await _make_store(tmp_path)
    await store.save_run(_run("run-with-routing", started_at=100.0))

    only_other_suite = EvalRun(
        run_id="run-without-routing", started_at=200.0, finished_at=205.0,
        suites=[SuiteResult.from_outcomes("safety", [])],
        config_snapshot={}, git_sha=None,
    )
    await store.save_run(only_other_suite)

    latest_routing = await store.latest_run(suite="routing")
    assert latest_routing.run_id == "run-with-routing"


@pytest.mark.asyncio
async def test_list_runs_orders_most_recent_first(tmp_path) -> None:
    store = await _make_store(tmp_path)
    await store.save_run(_run("run-1", started_at=100.0))
    await store.save_run(_run("run-2", started_at=200.0))
    await store.save_run(_run("run-3", started_at=300.0))

    runs = await store.list_runs(limit=2)

    assert [r.run_id for r in runs] == ["run-3", "run-2"]


@pytest.mark.asyncio
async def test_delete_run_removes_run_and_cases(tmp_path) -> None:
    store = await _make_store(tmp_path)
    await store.save_run(_run("run-1"))

    await store.delete_run("run-1")

    assert await store.get_run("run-1") is None
