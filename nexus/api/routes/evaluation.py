from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from nexus.api.schemas import (
    EvalCaseOutcomeSchema,
    EvalCompareResponse,
    EvalRegressionFindingSchema,
    EvalRunRequest,
    EvalRunSchema,
    EvalSuiteResultSchema,
)
from nexus.api.services import get_services
from nexus.evaluation.regression import compare_runs
from nexus.evaluation.runner import EvalHarness
from nexus.evaluation.store import EvalStore
from nexus.evaluation.types import EvalRun

router = APIRouter()


def _to_schema(run: EvalRun) -> EvalRunSchema:
    return EvalRunSchema(
        run_id=run.run_id,
        started_at=run.started_at,
        finished_at=run.finished_at,
        suites=[
            EvalSuiteResultSchema(
                suite=s.suite,
                outcomes=[
                    EvalCaseOutcomeSchema(
                        case_id=o.case_id, passed=o.passed, score=o.score, actual=o.actual,
                        detail=o.detail, latency_seconds=o.latency_seconds, cost_usd=o.cost_usd,
                    )
                    for o in s.outcomes
                ],
                pass_rate=s.pass_rate, mean_score=s.mean_score,
                total_cost_usd=s.total_cost_usd, total_latency_seconds=s.total_latency_seconds,
            )
            for s in run.suites
        ],
        config_snapshot=run.config_snapshot,
        git_sha=run.git_sha,
    )


@router.post("/eval/run", response_model=EvalRunSchema)
async def run_eval(payload: EvalRunRequest, request: Request) -> EvalRunSchema:
    store: EvalStore = get_services(request).eval_store
    harness = EvalHarness(use_real_providers=payload.use_real_providers)
    run = await harness.run(payload.suites)
    await store.save_run(run)
    return _to_schema(run)


@router.get("/eval/runs", response_model=list[EvalRunSchema])
async def list_eval_runs(request: Request, limit: int = 20) -> list[EvalRunSchema]:
    store: EvalStore = get_services(request).eval_store
    runs = await store.list_runs(limit=limit)
    return [_to_schema(r) for r in runs]


@router.get("/eval/runs/{run_id}", response_model=EvalRunSchema)
async def get_eval_run(run_id: str, request: Request) -> EvalRunSchema:
    store: EvalStore = get_services(request).eval_store
    run = await store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"No eval run found for run_id={run_id!r}.")
    return _to_schema(run)


@router.get("/eval/compare", response_model=EvalCompareResponse)
async def compare_eval_runs(
    request: Request, baseline: str = Query(...), candidate: str = Query(...)
) -> EvalCompareResponse:
    store: EvalStore = get_services(request).eval_store
    baseline_run = await store.get_run(baseline)
    if baseline_run is None:
        raise HTTPException(status_code=404, detail=f"No eval run found for run_id={baseline!r}.")
    candidate_run = await store.get_run(candidate)
    if candidate_run is None:
        raise HTTPException(status_code=404, detail=f"No eval run found for run_id={candidate!r}.")

    settings = get_services(request).settings
    findings = compare_runs(
        baseline_run, candidate_run,
        pass_rate_tolerance=settings.evaluation.pass_rate_tolerance,
        cost_tolerance=settings.evaluation.cost_tolerance,
        latency_tolerance=settings.evaluation.latency_tolerance,
    )

    return EvalCompareResponse(
        baseline_run_id=baseline,
        candidate_run_id=candidate,
        findings=[
            EvalRegressionFindingSchema(
                suite=f.suite, case_id=f.case_id, kind=f.kind, before=f.before, after=f.after,
                detail=f.detail,
            )
            for f in findings
        ],
    )
