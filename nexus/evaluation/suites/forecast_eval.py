from __future__ import annotations

import time
import uuid

from nexus.evaluation.runner import EvalHarness, Evaluator
from nexus.evaluation.types import CaseOutcome, EvalCase
from nexus.personal.forecast import StateForecaster
from nexus.personal.state import PersonalStateEngine

_SECONDS_PER_DAY = 86400.0


class ForecastEvaluator(Evaluator):
    """Builds a synthetic signal series with a KNOWN slope, projects it,
    and checks the projection lands where arithmetic says it should.

    The second half of this suite matters as much as the first: a case
    with too few points must come back method="insufficient_data" and NO
    number. A forecaster that quietly emits a projection from two data
    points is the exact failure this gate exists to catch.
    """

    suite = "forecast"

    async def run_case(self, case: EvalCase, harness: EvalHarness) -> CaseOutcome:
        start = time.monotonic()
        user_id = f"eval-forecast-{uuid.uuid4().hex}"

        engine = PersonalStateEngine(harness.db_engine)
        await engine.init()

        dimension = case.input["dimension"]
        values = case.input["values"]
        day_spacing = case.input.get("day_spacing", 1.0)

        # Written oldest-first ending at "now" so the recency-weighted
        # current state reflects the last value, and the trend fit sees the
        # whole run-up.
        now = time.time()
        for index, value in enumerate(values):
            offset_days = (len(values) - 1 - index) * day_spacing
            await engine.record_signal_at(
                user_id=user_id,
                dimension=dimension,
                value=value,
                source="explicit",
                recorded_at=now - offset_days * _SECONDS_PER_DAY,
            )

        forecaster = StateForecaster(
            engine,
            max_horizon_days=case.input.get("max_horizon_days", 30),
            min_trend_confidence=case.input.get("min_trend_confidence", 0.5),
            trend_min_samples=case.input.get("trend_min_samples", 4),
        )
        result = await forecaster.forecast(user_id, horizon_days=case.input["horizon_days"])

        forecast = next((f for f in result.forecasts if f.dimension == dimension), None)
        if forecast is None:
            return CaseOutcome(
                case_id=case.id, passed=False, score=0.0, actual={},
                detail=f"no forecast produced for dimension={dimension!r}",
                latency_seconds=time.monotonic() - start, cost_usd=0.0,
            )

        actual = {
            "method": forecast.method,
            "projected_value": forecast.projected_value,
            "confidence": forecast.confidence,
            "horizon_days": result.horizon_days,
        }

        checks: list[tuple[str, bool]] = [
            ("method", forecast.method == case.expected["method"])
        ]
        if case.expected["method"] == "insufficient_data":
            # The whole point: no number at all, and a caveat that says why.
            checks.append(("no_projection", forecast.projected_value is None))
            checks.append(("has_caveat", bool(forecast.caveat.strip())))
        else:
            expected_value = case.expected["projected_value"]
            tolerance = case.expected.get("tolerance", 0.05)
            checks.append(
                (
                    "projected_value",
                    forecast.projected_value is not None
                    and abs(forecast.projected_value - expected_value) <= tolerance,
                )
            )
        if "max_horizon_days" in case.expected:
            checks.append(
                ("horizon_capped", result.horizon_days <= case.expected["max_horizon_days"])
            )

        passed = all(ok for _name, ok in checks)
        failed = [name for name, ok in checks if not ok]

        return CaseOutcome(
            case_id=case.id,
            passed=passed,
            score=1.0 if passed else 0.0,
            actual=actual,
            detail=(
                f"method={forecast.method!r} projected={forecast.projected_value} "
                f"horizon={result.horizon_days}"
                + (f"; failed checks: {', '.join(failed)}" if failed else "")
            ),
            latency_seconds=time.monotonic() - start,
            cost_usd=0.0,
        )
