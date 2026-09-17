from __future__ import annotations

import os

import pytest

from nexus.config.settings import get_settings
from nexus.evaluation.runner import EvalHarness


@pytest.fixture(scope="module")
def _memory_db_env(tmp_path_factory) -> str:
    db_path = str(tmp_path_factory.mktemp("nexus-eval-harness") / "nexus.db")
    os.environ["NEXUS_MEMORY__DATABASE_PATH"] = db_path
    get_settings(refresh=True)
    yield db_path
    os.environ.pop("NEXUS_MEMORY__DATABASE_PATH", None)
    get_settings(refresh=True)


@pytest.mark.asyncio
async def test_run_produces_a_well_formed_eval_run(_memory_db_env: str) -> None:
    harness = EvalHarness(use_real_providers=False)

    run = await harness.run(["safety"])

    assert run.run_id
    assert run.finished_at >= run.started_at
    assert len(run.suites) == 1
    assert run.suites[0].suite == "safety"
    assert len(run.suites[0].outcomes) > 0
    assert run.config_snapshot  # non-empty settings snapshot


@pytest.mark.asyncio
async def test_default_suites_run_when_none_specified(_memory_db_env: str) -> None:
    harness = EvalHarness(use_real_providers=False)

    run = await harness.run()

    suite_names = {s.suite for s in run.suites}
    assert suite_names == {
        "routing",
        "classification",
        "rag",
        "tools",
        "safety",
        "verification",
        "graph_rag",
        "orchestration",
        "forecast",
    }


@pytest.mark.asyncio
async def test_no_real_network_calls_by_default(_memory_db_env: str) -> None:
    # If this ran against real providers it would either hang or raise a
    # connection error against a non-existent local Ollama/cloud endpoint
    # within a normal test timeout — completing quickly with all suites
    # populated is itself evidence no real network call was attempted.
    harness = EvalHarness(use_real_providers=False)

    run = await harness.run(["routing", "tools"])

    for suite in run.suites:
        assert len(suite.outcomes) > 0


@pytest.mark.asyncio
async def test_config_snapshot_never_carries_secrets(_memory_db_env: str) -> None:
    os.environ["OPENAI_API_KEY"] = "sk-should-not-leak-into-snapshot"
    get_settings(refresh=True)
    try:
        harness = EvalHarness(use_real_providers=False)
        run = await harness.run(["safety"])

        openai_section = run.config_snapshot.get("cloud", {}).get("openai", {})
        assert "api_key" not in openai_section
    finally:
        os.environ.pop("OPENAI_API_KEY", None)
        get_settings(refresh=True)


@pytest.mark.asyncio
async def test_run_against_a_small_custom_fixture_dataset(tmp_path, _memory_db_env: str) -> None:
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    (datasets_dir / "routing.jsonl").write_text(
        '{"id": "fixture-001", "suite": "routing", "input": {"query": "fix this stack trace bug", '
        '"policy": "BALANCED"}, "expected": {"task_type": "coding"}, "metadata": {}}\n',
        encoding="utf-8",
    )

    os.environ["NEXUS_EVALUATION__DATASETS_DIR"] = str(datasets_dir)
    get_settings(refresh=True)
    try:
        harness = EvalHarness(use_real_providers=False)
        run = await harness.run(["routing"])

        assert len(run.suites) == 1
        assert len(run.suites[0].outcomes) == 1
        assert run.suites[0].outcomes[0].case_id == "fixture-001"
        assert run.suites[0].outcomes[0].passed is True
    finally:
        os.environ.pop("NEXUS_EVALUATION__DATASETS_DIR", None)
        get_settings(refresh=True)
