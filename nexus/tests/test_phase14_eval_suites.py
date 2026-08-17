from __future__ import annotations

import json
from pathlib import Path

import pytest

from nexus.config.settings import get_settings
from nexus.evaluation.runner import EvalHarness
from nexus.evaluation.types import EvalCase

_NEW_SUITES = ("graph_rag", "orchestration", "forecast")
_DATASETS_DIR = Path("nexus/evaluation/datasets")


@pytest.fixture
def _memory_db_env(tmp_path, monkeypatch):
    db_path = str(tmp_path / "nexus.db")
    monkeypatch.setenv("NEXUS_MEMORY__DATABASE_PATH", db_path)
    get_settings(refresh=True)
    yield db_path
    monkeypatch.delenv("NEXUS_MEMORY__DATABASE_PATH", raising=False)
    get_settings(refresh=True)


def _cases(suite: str) -> list[EvalCase]:
    path = _DATASETS_DIR / f"{suite}.jsonl"
    cases = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                data = json.loads(line)
                cases.append(
                    EvalCase(
                        id=data["id"], suite=data["suite"], input=data["input"],
                        expected=data["expected"], metadata=data.get("metadata", {}),
                    )
                )
    return cases


@pytest.mark.parametrize("suite", _NEW_SUITES)
def test_dataset_exists_and_is_non_empty(suite: str) -> None:
    assert _cases(suite), f"{suite}.jsonl has no cases"


@pytest.mark.parametrize("suite", _NEW_SUITES)
def test_every_case_declares_its_own_suite(suite: str) -> None:
    assert all(case.suite == suite for case in _cases(suite))


@pytest.mark.parametrize("suite", _NEW_SUITES)
def test_case_ids_are_unique(suite: str) -> None:
    ids = [case.id for case in _cases(suite)]

    assert len(ids) == len(set(ids))


@pytest.mark.asyncio
@pytest.mark.parametrize("suite", _NEW_SUITES)
async def test_suite_scores_its_known_good_cases_correctly(suite: str, _memory_db_env) -> None:
    """Every committed case is a known-good expectation, so a correct
    implementation must score the suite at 1.0."""
    harness = EvalHarness(use_real_providers=False)

    run = await harness.run([suite])

    [result] = run.suites
    failures = [f"{o.case_id}: {o.detail}" for o in result.outcomes if not o.passed]
    assert result.pass_rate == 1.0, "\n".join(failures)


@pytest.mark.asyncio
async def test_graph_rag_scores_a_known_bad_expectation_as_failing(_memory_db_env) -> None:
    """The other half of a real gate: a wrong expectation must FAIL, or the
    suite would pass no matter what the code did."""
    from nexus.evaluation.suites.graph_rag_eval import GraphRagEvaluator

    harness = EvalHarness(use_real_providers=False)
    await harness.setup()

    case = _cases("graph_rag")[0]
    case.expected = {"contains_source": "a-document-that-does-not-exist.txt"}

    outcome = await GraphRagEvaluator().run_case(case, harness)

    assert outcome.passed is False


@pytest.mark.asyncio
async def test_orchestration_scores_a_known_bad_expectation_as_failing(_memory_db_env) -> None:
    from nexus.evaluation.suites.orchestration_eval import OrchestrationEvaluator

    harness = EvalHarness(use_real_providers=False)
    await harness.setup()

    case = next(c for c in _cases("orchestration") if c.input.get("mode") == "guard")
    case.expected = {"accepted": [False] * len(case.input["attempts"])}

    outcome = await OrchestrationEvaluator().run_case(case, harness)

    assert outcome.passed is False


@pytest.mark.asyncio
async def test_forecast_scores_a_known_bad_expectation_as_failing(_memory_db_env) -> None:
    from nexus.evaluation.suites.forecast_eval import ForecastEvaluator

    harness = EvalHarness(use_real_providers=False)
    await harness.setup()

    case = _cases("forecast")[0]
    case.expected = {"method": "trend_extrapolation", "projected_value": 0.01, "tolerance": 0.001}

    outcome = await ForecastEvaluator().run_case(case, harness)

    assert outcome.passed is False


@pytest.mark.asyncio
async def test_forecast_suite_rejects_a_number_where_insufficient_data_is_expected(
    _memory_db_env,
) -> None:
    """Specifically guards the thin-data contract: if the forecaster started
    emitting a projection from two points, this case must go red."""
    from nexus.evaluation.suites.forecast_eval import ForecastEvaluator

    harness = EvalHarness(use_real_providers=False)
    await harness.setup()

    case = next(c for c in _cases("forecast") if c.expected["method"] == "insufficient_data")
    case.expected = {"method": "trend_extrapolation", "projected_value": 0.5, "tolerance": 0.5}

    outcome = await ForecastEvaluator().run_case(case, harness)

    assert outcome.passed is False


@pytest.mark.asyncio
async def test_new_suites_are_in_the_default_set(_memory_db_env) -> None:
    default_suites = set(get_settings().evaluation.default_suites)

    assert set(_NEW_SUITES) <= default_suites


@pytest.mark.asyncio
async def test_all_nine_suites_run_together_cleanly(_memory_db_env) -> None:
    harness = EvalHarness(use_real_providers=False)

    run = await harness.run()

    assert len(run.suites) == 9
    failures = {
        s.suite: [o.case_id for o in s.outcomes if not o.passed]
        for s in run.suites
        if s.pass_rate < 1.0
    }
    assert not failures, failures
