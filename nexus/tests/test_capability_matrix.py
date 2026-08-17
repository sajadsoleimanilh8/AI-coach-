from __future__ import annotations

import pytest

from nexus.evaluation.types import CaseOutcome, EvalRun, SuiteResult
from nexus.intelligence.capability_matrix import SUITE_CAPABILITY_MAP, CapabilityMatrix
from nexus.memory.storage import create_async_db_engine

_MISTRAL = "mistral:7b"
_MISTRAL_STATIC_REASONING = 0.55
_MISTRAL_STATIC_CODING = 0.55


async def _make_matrix(tmp_path, **kwargs) -> CapabilityMatrix:
    engine = create_async_db_engine(str(tmp_path / "nexus.db"))
    matrix = CapabilityMatrix(engine, **kwargs)
    await matrix.init()
    return matrix


def _run(suite: str, *, pass_rate: float, case_count: int, model_id: str = _MISTRAL) -> EvalRun:
    passing = round(pass_rate * case_count)
    outcomes = [
        CaseOutcome(
            case_id=f"{suite}-{i}", passed=i < passing, score=1.0 if i < passing else 0.0,
            actual={}, detail="", latency_seconds=0.0, cost_usd=0.0,
        )
        for i in range(case_count)
    ]
    return EvalRun(
        run_id=f"run-{suite}-{case_count}",
        started_at=0.0,
        finished_at=1.0,
        suites=[SuiteResult.from_outcomes(suite, outcomes)],
        config_snapshot={"evaluated_model_id": model_id},
        git_sha=None,
    )


@pytest.mark.asyncio
async def test_zero_samples_yields_exactly_the_static_values(tmp_path) -> None:
    """Principle 4. With no measurements, routing must be byte-identical to
    running off models.yaml — not close to it."""
    matrix = await _make_matrix(tmp_path)
    await matrix.refresh()

    effective = matrix.effective_capabilities(_MISTRAL)

    assert effective["reasoning"] == _MISTRAL_STATIC_REASONING
    assert effective["coding"] == _MISTRAL_STATIC_CODING


@pytest.mark.asyncio
async def test_unknown_model_returns_empty_rather_than_inventing_scores(tmp_path) -> None:
    matrix = await _make_matrix(tmp_path)
    await matrix.refresh()

    assert matrix.effective_capabilities("not-a-registered-model") == {}


@pytest.mark.asyncio
async def test_measurements_below_min_samples_do_not_move_the_score(tmp_path) -> None:
    matrix = await _make_matrix(tmp_path, min_samples=20)
    await matrix.update_from_eval_run(_run("routing", pass_rate=1.0, case_count=5))
    await matrix.refresh()

    assert matrix.effective_capabilities(_MISTRAL)["reasoning"] == _MISTRAL_STATIC_REASONING


@pytest.mark.asyncio
async def test_learned_weight_rises_with_sample_size(tmp_path) -> None:
    matrix = await _make_matrix(tmp_path, min_samples=20, max_learned_weight=0.7)

    assert matrix.learned_weight(19) == 0.0
    assert matrix.learned_weight(20) == pytest.approx(0.5)
    assert matrix.learned_weight(40) == pytest.approx(2 / 3)
    assert matrix.learned_weight(20) < matrix.learned_weight(40)


@pytest.mark.asyncio
async def test_learned_weight_is_capped(tmp_path) -> None:
    matrix = await _make_matrix(tmp_path, min_samples=20, max_learned_weight=0.7)

    assert matrix.learned_weight(10_000) == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_blending_math_on_a_hand_computed_fixture(tmp_path) -> None:
    """20 samples -> weight 0.5. measured 0.95, static 0.55:
    0.5 * 0.95 + 0.5 * 0.55 = 0.75."""
    matrix = await _make_matrix(tmp_path, min_samples=20, max_learned_weight=0.7)
    await matrix.update_from_eval_run(_run("routing", pass_rate=0.95, case_count=20))
    await matrix.refresh()

    assert matrix.effective_capabilities(_MISTRAL)["reasoning"] == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_capped_blending_on_a_hand_computed_fixture(tmp_path) -> None:
    """100 samples -> weight capped at 0.7. measured 1.0, static 0.55:
    0.7 * 1.0 + 0.3 * 0.55 = 0.865."""
    matrix = await _make_matrix(tmp_path, min_samples=20, max_learned_weight=0.7)
    await matrix.update_from_eval_run(_run("routing", pass_rate=1.0, case_count=100))
    await matrix.refresh()

    assert matrix.effective_capabilities(_MISTRAL)["reasoning"] == pytest.approx(0.865)


@pytest.mark.asyncio
async def test_a_bad_measurement_lowers_the_effective_score(tmp_path) -> None:
    matrix = await _make_matrix(tmp_path, min_samples=20)
    await matrix.update_from_eval_run(_run("routing", pass_rate=0.0, case_count=40))
    await matrix.refresh()

    effective = matrix.effective_capabilities(_MISTRAL)["reasoning"]

    assert effective < _MISTRAL_STATIC_REASONING
    assert effective == pytest.approx((1 / 3) * _MISTRAL_STATIC_REASONING)


@pytest.mark.asyncio
async def test_several_suites_feeding_one_capability_aggregate_by_sample_weight(tmp_path) -> None:
    matrix = await _make_matrix(tmp_path, min_samples=20)
    await matrix.update_from_eval_run(_run("routing", pass_rate=1.0, case_count=40))
    await matrix.update_from_eval_run(_run("verification", pass_rate=0.0, case_count=20))
    await matrix.refresh()

    measured, samples = matrix.measured(_MISTRAL, "reasoning")

    assert samples == 60
    assert measured == pytest.approx(40 / 60)


@pytest.mark.asyncio
async def test_safety_suite_never_becomes_a_capability_score(tmp_path) -> None:
    """Safety is a gate, not a skill. A model that blocks every red flag is
    not thereby better at reasoning."""
    matrix = await _make_matrix(tmp_path, min_samples=1)
    written = await matrix.update_from_eval_run(_run("safety", pass_rate=1.0, case_count=50))
    await matrix.refresh()

    assert written == 0
    assert "safety" not in SUITE_CAPABILITY_MAP
    assert matrix.effective_capabilities(_MISTRAL)["reasoning"] == _MISTRAL_STATIC_REASONING


@pytest.mark.asyncio
async def test_forecast_suite_is_excluded_too(tmp_path) -> None:
    assert "forecast" not in SUITE_CAPABILITY_MAP


@pytest.mark.asyncio
async def test_a_run_without_a_model_id_records_nothing(tmp_path) -> None:
    matrix = await _make_matrix(tmp_path)
    run = _run("routing", pass_rate=1.0, case_count=40)
    run.config_snapshot = {}

    assert await matrix.update_from_eval_run(run) == 0


@pytest.mark.asyncio
async def test_model_id_can_be_passed_explicitly(tmp_path) -> None:
    matrix = await _make_matrix(tmp_path, min_samples=20)
    run = _run("routing", pass_rate=1.0, case_count=40)
    run.config_snapshot = {}

    assert await matrix.update_from_eval_run(run, model_id="gpt-4o-mini") == 1
    await matrix.refresh()
    assert matrix.measured("gpt-4o-mini", "reasoning") is not None


@pytest.mark.asyncio
async def test_refresh_is_required_before_reads_see_new_measurements(tmp_path) -> None:
    matrix = await _make_matrix(tmp_path, min_samples=20)
    await matrix.update_from_eval_run(_run("routing", pass_rate=1.0, case_count=40))

    assert matrix.effective_capabilities(_MISTRAL)["reasoning"] == _MISTRAL_STATIC_REASONING

    await matrix.refresh()

    assert matrix.effective_capabilities(_MISTRAL)["reasoning"] > _MISTRAL_STATIC_REASONING
