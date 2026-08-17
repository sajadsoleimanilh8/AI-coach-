from __future__ import annotations

import json
import os
import uuid

import pytest

from nexus.config.settings import get_settings
from nexus.evaluation import cli
from nexus.evaluation.runner import EvalHarness
from nexus.evaluation.store import EvalStore
from nexus.evaluation.types import CaseOutcome, EvalRun, SuiteResult
from nexus.memory.storage import create_async_db_engine


@pytest.fixture
def _memory_db_env(tmp_path):
    db_path = str(tmp_path / "nexus.db")
    os.environ["NEXUS_MEMORY__DATABASE_PATH"] = db_path
    get_settings(refresh=True)
    yield db_path
    os.environ.pop("NEXUS_MEMORY__DATABASE_PATH", None)
    get_settings(refresh=True)


def _good_run(run_id: str) -> EvalRun:
    outcomes = [CaseOutcome(case_id="c1", passed=True, score=1.0, actual={}, detail="ok",
                             latency_seconds=0.01, cost_usd=0.0)]
    return EvalRun(
        run_id=run_id, started_at=0.0, finished_at=1.0,
        suites=[SuiteResult.from_outcomes("routing", outcomes)], config_snapshot={}, git_sha=None, provider_mode="fake",
    )


def _bad_safety_run() -> EvalRun:
    outcomes = [CaseOutcome(case_id="fake-1", passed=False, score=0.0, actual={}, detail="forced failure",
                             latency_seconds=0.01, cost_usd=0.0)]
    return EvalRun(
        run_id=f"fake-{uuid.uuid4().hex}", started_at=0.0, finished_at=1.0,
        suites=[SuiteResult.from_outcomes("safety", outcomes)], config_snapshot={}, git_sha=None, provider_mode="fake",
    )


def _regressed_routing_run() -> EvalRun:
    outcomes = [CaseOutcome(case_id="c1", passed=False, score=0.0, actual={}, detail="regressed",
                             latency_seconds=0.01, cost_usd=0.0)]
    return EvalRun(
        run_id=f"fake-{uuid.uuid4().hex}", started_at=0.0, finished_at=1.0,
        suites=[SuiteResult.from_outcomes("routing", outcomes)], config_snapshot={}, git_sha=None, provider_mode="fake",
    )


def test_clean_run_exits_zero_and_json_parses(_memory_db_env, capsys) -> None:
    exit_code = cli.main(["--suites", "safety", "--json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["run"]["run_id"]
    assert output["run"]["suites"][0]["suite"] == "safety"
    assert output["regressions"] == []
    assert output["safety_failed"] is False


def test_human_readable_output_mentions_the_suite(_memory_db_env, capsys) -> None:
    exit_code = cli.main(["--suites", "safety"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "safety" in out
    assert "No regressions found" in out or "pass_rate=1.00" in out


def test_exit_code_one_when_safety_suite_fails(_memory_db_env, monkeypatch, capsys) -> None:
    async def _fake_run(self, suites=None):
        return _bad_safety_run()

    monkeypatch.setattr(EvalHarness, "run", _fake_run)

    exit_code = cli.main(["--suites", "safety", "--json"])

    assert exit_code == 1
    output = json.loads(capsys.readouterr().out)
    assert output["safety_failed"] is True


async def _seed_baseline(db_path: str, run: EvalRun) -> None:
    engine = create_async_db_engine(db_path)
    store = EvalStore(engine)
    await store.init()
    await store.save_run(run)
    await engine.dispose()


def test_exit_code_one_on_regression_against_baseline(_memory_db_env, monkeypatch, capsys) -> None:
    import asyncio

    baseline = _good_run("baseline-run")
    asyncio.run(_seed_baseline(_memory_db_env, baseline))

    async def _fake_run(self, suites=None):
        return _regressed_routing_run()

    monkeypatch.setattr(EvalHarness, "run", _fake_run)

    exit_code = cli.main(["--suites", "routing", "--compare-to", "baseline-run", "--json"])

    assert exit_code == 1
    output = json.loads(capsys.readouterr().out)
    assert len(output["regressions"]) >= 1
    assert any(f["kind"] == "case_regressed" for f in output["regressions"])


def test_exit_code_zero_when_compare_to_shows_no_regression(_memory_db_env, monkeypatch, capsys) -> None:
    import asyncio

    baseline = _good_run("baseline-run-2")
    asyncio.run(_seed_baseline(_memory_db_env, baseline))

    async def _fake_run(self, suites=None):
        return _good_run("candidate-matches-baseline")

    monkeypatch.setattr(EvalHarness, "run", _fake_run)

    exit_code = cli.main(["--suites", "routing", "--compare-to", "baseline-run-2", "--json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["regressions"] == []


def test_unknown_baseline_run_id_is_an_error(_memory_db_env, capsys) -> None:
    exit_code = cli.main(["--suites", "safety", "--compare-to", "does-not-exist", "--json"])

    assert exit_code == 1
    output = json.loads(capsys.readouterr().out)
    assert "error" in output


def _real_provider_run() -> EvalRun:
    outcomes = [CaseOutcome(case_id="c1", passed=True, score=1.0, actual={}, detail="ok",
                             latency_seconds=0.01, cost_usd=0.0)]
    return EvalRun(
        run_id=f"real-{uuid.uuid4().hex}", started_at=0.0, finished_at=1.0,
        suites=[SuiteResult.from_outcomes("routing", outcomes)], config_snapshot={}, git_sha=None,
        provider_mode="real",
    )


def test_comparing_across_provider_modes_reports_a_refusal_not_a_traceback(
    _memory_db_env, monkeypatch, capsys
) -> None:
    import asyncio

    asyncio.run(_seed_baseline(_memory_db_env, _good_run("fake-mode-baseline")))

    async def _fake_run(self, suites=None):
        return _real_provider_run()

    monkeypatch.setattr(EvalHarness, "run", _fake_run)

    exit_code = cli.main(["--suites", "routing", "--compare-to", "fake-mode-baseline", "--json"])

    assert exit_code == 1
    output = json.loads(capsys.readouterr().out)
    assert "provider_mode" in output["error"]
