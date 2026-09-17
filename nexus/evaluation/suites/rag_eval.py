from __future__ import annotations

import time
import uuid

from nexus.evaluation.runner import EvalHarness, Evaluator
from nexus.evaluation.types import CaseOutcome, EvalCase

_TOP_K = 3


class RagEvaluator(Evaluator):
    """Seeds the documents a case declares, then checks whether the
    expected source_name shows up in top-k retrieval for the query —
    measures retrieval quality, not generation (no LLM call at all).

    Datasets can't hardcode a doc_id (RagService.ingest() generates a
    fresh uuid per run), so cases identify the expected match by
    source_name instead — the one stable identifier under the caller's
    control."""

    suite = "rag"

    async def run_case(self, case: EvalCase, harness: EvalHarness) -> CaseOutcome:
        start = time.monotonic()
        user_id = f"eval-{uuid.uuid4().hex}"  # isolate each case's seeded docs from every other

        for doc in case.input["documents"]:
            await harness.rag_service.ingest(
                user_id=user_id, source_name=doc["source_name"], source_type="txt",
                raw_bytes=doc["text"].encode("utf-8"),
            )

        retrieved = await harness.rag_service.retrieve(
            user_id=user_id, query=case.input["query"], top_k=_TOP_K
        )
        retrieved_names = [chunk.source_name for chunk in retrieved]
        expected_name = case.expected["source_name"]

        # recall@k: did the expected document appear anywhere in top-k —
        # binary per case, but reported as a score (not just pass/fail) so
        # a suite average reads as a real recall@k rate across cases.
        hit = expected_name in retrieved_names
        top1 = bool(retrieved_names) and retrieved_names[0] == expected_name

        return CaseOutcome(
            case_id=case.id, passed=hit, score=1.0 if hit else 0.0,
            actual={"retrieved": retrieved_names, "top1": top1},
            detail=f"expected {expected_name!r} in top-{_TOP_K}, got {retrieved_names!r}",
            latency_seconds=time.monotonic() - start, cost_usd=0.0,
        )
