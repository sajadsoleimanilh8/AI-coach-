from __future__ import annotations

import time
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from nexus.evaluation.types import EvalRun
from nexus.logging_setup.logger import get_logger
from nexus.memory.storage import ModelCapabilityRecord, init_db, make_session_factory
from nexus.models.registry import get_model

logger = get_logger("intelligence.capability_matrix")

# Which capability key a suite's pass rate is evidence FOR. Suites that
# measure something other than model capability are deliberately absent:
#
#   safety   — a gate, not a skill. A model that blocks every red flag is
#              not thereby better at reasoning, and letting safety inflate
#              a capability score would let a cautious model outrank a
#              more capable one on tasks safety says nothing about.
#   forecast — deterministic arithmetic over recorded signals. It exercises
#              no model at all, so it is evidence about the code, not about
#              whichever model happened to be configured.
SUITE_CAPABILITY_MAP: dict[str, str] = {
    "coding": "coding",
    "math": "math",
    "routing": "reasoning",
    "verification": "reasoning",
    "tools": "reasoning",
    "orchestration": "reasoning",
    "classification": "language",
    "rag": "language",
    "graph_rag": "language",
}

# Sample size at which learned evidence earns half its maximum weight.
# Chosen so a handful of cases cannot swing model selection: at 5 samples
# the learned score carries 0.2 weight, at 20 it carries 0.5.
_LEARNED_WEIGHT_HALF_POINT = 20


@dataclass
class CapabilityMeasurement:
    model_id: str
    capability_key: str
    measured_score: float
    sample_size: int
    source_run_id: str
    measured_at: float


class CapabilityMatrix:
    """Blends measured eval scores with the static models.yaml capability
    numbers, so routing improves as evidence accumulates without ever
    lurching on thin data.

    Reads are SYNCHRONOUS off an in-memory cache because
    ModelRouter._ranked_decisions() is not async — the same shape
    LatencyTracker already uses for p50(). refresh() repopulates the cache
    at startup and after each eval run.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        min_samples: int = 20,
        max_learned_weight: float = 0.7,
    ) -> None:
        self._engine = engine
        self._session_factory: async_sessionmaker = make_session_factory(engine)
        self._min_samples = min_samples
        self._max_learned_weight = max_learned_weight
        # model_id -> capability_key -> (weighted_score, total_samples)
        self._cache: dict[str, dict[str, tuple[float, int]]] = {}

    async def init(self) -> None:
        await init_db(self._engine)

    async def update_from_eval_run(self, run: EvalRun, *, model_id: str | None = None) -> int:
        """Records one measurement per (model, capability) the run produced
        evidence for. Returns how many rows were written."""
        target_model = model_id or run.config_snapshot.get("evaluated_model_id")
        if not target_model:
            logger.warning(
                "eval run %s carries no evaluated_model_id; no capability measurements "
                "recorded (a run has to say which model it measured)",
                run.run_id,
            )
            return 0

        now = time.time()
        written = 0
        async with self._session_factory() as db:
            for suite in run.suites:
                capability_key = SUITE_CAPABILITY_MAP.get(suite.suite)
                if capability_key is None or not suite.outcomes:
                    continue
                db.add(
                    ModelCapabilityRecord(
                        model_id=target_model,
                        capability_key=capability_key,
                        measured_score=suite.pass_rate,
                        sample_size=len(suite.outcomes),
                        source_run_id=run.run_id,
                        measured_at=now,
                    )
                )
                written += 1
            await db.commit()
        return written

    async def refresh(self) -> None:
        async with self._session_factory() as db:
            result = await db.execute(select(ModelCapabilityRecord))
            rows = list(result.scalars().all())

        # Several suites can map to the same capability key (routing,
        # verification and tools all feed "reasoning"), and a capability is
        # re-measured on every run. Aggregating as a sample-weighted mean
        # means a 50-case suite counts for more than a 5-case one, and a
        # capability measured repeatedly accumulates confidence rather than
        # being overwritten by whichever run finished last.
        accumulated: dict[str, dict[str, tuple[float, int]]] = {}
        for row in rows:
            per_model = accumulated.setdefault(row.model_id, {})
            weighted_sum, samples = per_model.get(row.capability_key, (0.0, 0))
            per_model[row.capability_key] = (
                weighted_sum + row.measured_score * row.sample_size,
                samples + row.sample_size,
            )

        self._cache = accumulated

    def learned_weight(self, sample_size: int) -> float:
        if sample_size < self._min_samples:
            return 0.0
        return min(
            self._max_learned_weight,
            sample_size / (sample_size + _LEARNED_WEIGHT_HALF_POINT),
        )

    def measured(self, model_id: str, capability_key: str) -> tuple[float, int] | None:
        entry = self._cache.get(model_id, {}).get(capability_key)
        if entry is None:
            return None
        weighted_sum, samples = entry
        if samples <= 0:
            return None
        return weighted_sum / samples, samples

    def effective_capabilities(self, model_id: str) -> dict[str, float]:
        """effective = w * measured + (1 - w) * static, with
        w = learned_weight(sample_size).

        Zero samples (or fewer than min_samples) yields EXACTLY the static
        value, so a NEXUS install that has never run an eval routes
        identically to one running off models.yaml alone — degrading to
        today's behavior rather than to some near-miss of it.
        """
        model_info = get_model(model_id)
        static = dict(model_info.capabilities) if model_info is not None else {}

        learned = self._cache.get(model_id)
        if not learned:
            return static

        effective = dict(static)
        for capability_key, (weighted_sum, samples) in learned.items():
            if samples <= 0:
                continue
            weight = self.learned_weight(samples)
            if weight == 0.0:
                continue
            measured_score = weighted_sum / samples
            static_score = static.get(capability_key, 0.0)
            effective[capability_key] = weight * measured_score + (1 - weight) * static_score
        return effective

    def all_measurements(self) -> list[CapabilityMeasurement]:
        """Flattened cache for the /api/models/capabilities view — what the
        router is actually scoring on, not what models.yaml claims."""
        return [
            CapabilityMeasurement(
                model_id=model_id,
                capability_key=capability_key,
                measured_score=weighted_sum / samples,
                sample_size=samples,
                source_run_id="",
                measured_at=0.0,
            )
            for model_id, per_model in self._cache.items()
            for capability_key, (weighted_sum, samples) in per_model.items()
            if samples > 0
        ]
