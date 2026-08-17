from __future__ import annotations

import time
import uuid

from nexus.evaluation.runner import EvalHarness, Evaluator
from nexus.evaluation.types import CaseOutcome, EvalCase
from nexus.rag.graph import Triple

_SEED_TOP_K = 1


class GraphRagEvaluator(Evaluator):
    """Seeds documents plus an explicit entity graph, retrieves seeds by
    vector search, then expands through the graph.
    """

    suite = "graph_rag"

    async def run_case(self, case: EvalCase, harness: EvalHarness) -> CaseOutcome:
        start = time.monotonic()
        user_id = f"eval-graph-{uuid.uuid4().hex}"

        doc_ids: dict[str, str] = {}
        for doc in case.input["documents"]:
            summary = await harness.rag_service.ingest(
                user_id=user_id,
                source_name=doc["source_name"],
                source_type="txt",
                raw_bytes=doc["text"].encode("utf-8"),
            )
            doc_ids[doc["source_name"]] = summary.doc_id

        for entry in case.input.get("triples", []):
            await harness.graph_store.add_triples(
                user_id=user_id,
                doc_id=doc_ids[entry["source_name"]],
                triples=[
                    Triple(
                        source=entry["source"],
                        relation=entry["relation"],
                        target=entry["target"],
                    )
                ],
            )

        seeds = await harness.rag_service.retrieve(
            user_id=user_id, query=case.input["query"], top_k=_SEED_TOP_K
        )
        expansion = await harness.graph_retriever.expand(
            seeds,
            user_id=user_id,
            max_hops=case.input.get("max_hops", 2),
            max_nodes=case.input.get("max_nodes", 20),
        )

        expanded_sources = sorted({chunk.source_name for chunk in expansion.chunks})
        actual = {
            "expanded_sources": expanded_sources,
            "nodes_visited": expansion.nodes_visited,
            "hops_used": expansion.hops_used,
        }

        checks: list[tuple[str, bool]] = []
        if "contains_source" in case.expected:
            checks.append(
                ("contains_source", case.expected["contains_source"] in expanded_sources)
            )
        if "excludes_source" in case.expected:
            checks.append(
                ("excludes_source", case.expected["excludes_source"] not in expanded_sources)
            )
        if "max_nodes_visited" in case.expected:
            checks.append(
                (
                    "max_nodes_visited",
                    expansion.nodes_visited <= case.expected["max_nodes_visited"],
                )
            )
        if "max_hops_used" in case.expected:
            checks.append(
                ("max_hops_used", expansion.hops_used <= case.expected["max_hops_used"])
            )
        if "expanded_count" in case.expected:
            checks.append(
                ("expanded_count", len(expansion.chunks) == case.expected["expanded_count"])
            )

        passed = all(ok for _name, ok in checks) if checks else False
        failed_names = [name for name, ok in checks if not ok]
        detail = (
            f"expanded_sources={expanded_sources}, nodes_visited={expansion.nodes_visited}, "
            f"hops_used={expansion.hops_used}"
            + (f"; failed checks: {', '.join(failed_names)}" if failed_names else "")
        )

        return CaseOutcome(
            case_id=case.id,
            passed=passed,
            score=1.0 if passed else 0.0,
            actual=actual,
            detail=detail,
            latency_seconds=time.monotonic() - start,
            cost_usd=0.0,
        )
