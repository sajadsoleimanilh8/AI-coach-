from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from nexus.core.router import ModelRouter
from nexus.core.types import Message, TaskType, Usage
from nexus.core.vector_store import RetrievedChunk
from nexus.logging_setup.logger import get_logger
from nexus.memory.storage import (
    DocumentChunkRecord,
    DocumentRecord,
    GraphEdgeRecord,
    GraphNodeRecord,
    init_db,
    make_session_factory,
)

logger = get_logger("rag.graph")

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_NAME_MAX_CHARS = 200

# A graph-expanded chunk has, by construction, little or no lexical overlap
# with the query — that is the point of expanding. Scoring it purely on
# cosine/keyword would therefore rank it last and the reranker would drop
# it, defeating the feature. Instead it enters the rerank carrying a
# provenance score: "reachable from something the query DID match", decayed
# per hop so a two-hop connection is weaker evidence than a one-hop one.
_HOP_ONE_AFFINITY = 0.5
_HOP_DECAY = 0.7

_EXTRACT_SYSTEM_PROMPT = (
    "Extract entities and the relations between them from the text. Entities are concrete "
    "named things: people, organizations, systems, components, places, products. Relations "
    "are short snake_case verbs describing how one entity relates to another. Use only what "
    "the text states — never infer a relation the text does not support. Respond with ONLY a "
    "JSON object, no prose, in this exact shape: "
    '{"triples": [{"source": "<entity>", "relation": "<snake_case>", "target": "<entity>"}]}'
)


@dataclass
class Triple:
    source: str
    relation: str
    target: str


@dataclass
class GraphExpansion:
    chunks: list[RetrievedChunk]
    nodes_visited: int
    hops_used: int


def _parse_triples(text: str) -> list[Triple]:
    payload: Any = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(text)
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = None

    if not isinstance(payload, dict):
        return []

    triples: list[Triple] = []
    for item in payload.get("triples", []):
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()[:_NAME_MAX_CHARS]
        relation = str(item.get("relation", "")).strip()[:_NAME_MAX_CHARS]
        target = str(item.get("target", "")).strip()[:_NAME_MAX_CHARS]
        if source and relation and target and source.lower() != target.lower():
            triples.append(Triple(source=source, relation=relation, target=target))
    return triples


class EntityExtractor:
    """The one place in Phase 14 where an LLM is genuinely required —
    pulling (entity, relation, entity) triples out of prose is a language
    task with no deterministic equivalent. Everything downstream of it
    (traversal, capping, scoring) is arithmetic.

    Skipped entirely when disabled: extract() returns nothing and makes no
    provider call at all, so a NEXUS with graph RAG off never pays for it.
    """

    def __init__(self, router: ModelRouter, *, enabled: bool = True, batch_size: int = 4) -> None:
        self._router = router
        self._enabled = enabled
        self._batch_size = batch_size

    async def extract(self, chunks: list[str]) -> tuple[list[Triple], Usage]:
        if not self._enabled or not chunks:
            return [], Usage()

        all_triples: list[Triple] = []
        total = Usage()
        # Batched rather than one call per chunk: extraction quality barely
        # changes but the call count drops by batch_size, and this runs over
        # every chunk of every ingested document.
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            triples, usage = await self._extract_batch(batch)
            all_triples.extend(triples)
            total = Usage(
                prompt_tokens=total.prompt_tokens + usage.prompt_tokens,
                completion_tokens=total.completion_tokens + usage.completion_tokens,
            )
        return all_triples, total

    async def _extract_batch(self, batch: list[str]) -> tuple[list[Triple], Usage]:
        decision, provider = await self._router.route_with_failover(task_type=TaskType.RESEARCH)
        joined = "\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(batch))
        result = await provider.generate(
            [
                Message(role="system", content=_EXTRACT_SYSTEM_PROMPT),
                Message(role="user", content=joined),
            ],
            model_id=decision.model_id,
            temperature=0.0,
        )
        return _parse_triples(result.content), result.usage


class GraphStore:
    """Persistence for the entity graph. Node identity is (user_id, name)
    across documents — the same entity named in two documents is what makes
    a cross-document hop possible at all — while each row also records the
    doc_id it was observed in so expansion can get back to source chunks."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._session_factory: async_sessionmaker = make_session_factory(engine)

    async def init(self) -> None:
        await init_db(self._engine)

    async def add_triples(self, *, user_id: str, doc_id: str, triples: list[Triple]) -> int:
        if not triples:
            return 0
        now = time.time()
        async with self._session_factory() as db:
            node_ids: dict[str, int] = {}

            async def node_id_for(name: str) -> int:
                key = name.lower()
                if key in node_ids:
                    return node_ids[key]
                existing = await db.execute(
                    select(GraphNodeRecord).where(
                        GraphNodeRecord.user_id == user_id,
                        GraphNodeRecord.doc_id == doc_id,
                        GraphNodeRecord.name == name,
                    )
                )
                row = existing.scalar_one_or_none()
                if row is None:
                    row = GraphNodeRecord(
                        user_id=user_id,
                        doc_id=doc_id,
                        name=name,
                        node_type="entity",
                        created_at=now,
                    )
                    db.add(row)
                    await db.flush()
                node_ids[key] = row.id
                return row.id

            for triple in triples:
                source_id = await node_id_for(triple.source)
                target_id = await node_id_for(triple.target)
                db.add(
                    GraphEdgeRecord(
                        user_id=user_id,
                        source_node_id=source_id,
                        target_node_id=target_id,
                        relation=triple.relation,
                        weight=1.0,
                        doc_id=doc_id,
                        created_at=now,
                    )
                )
            await db.commit()
        return len(triples)

    async def all_nodes(self, user_id: str) -> list[GraphNodeRecord]:
        async with self._session_factory() as db:
            result = await db.execute(
                select(GraphNodeRecord).where(GraphNodeRecord.user_id == user_id)
            )
            return list(result.scalars().all())

    async def edges_for_nodes(self, user_id: str, node_ids: set[int]) -> list[GraphEdgeRecord]:
        if not node_ids:
            return []
        async with self._session_factory() as db:
            result = await db.execute(
                select(GraphEdgeRecord).where(GraphEdgeRecord.user_id == user_id)
            )
            edges = list(result.scalars().all())
        return [
            edge
            for edge in edges
            if edge.source_node_id in node_ids or edge.target_node_id in node_ids
        ]

    async def chunks_for_docs(
        self, user_id: str, doc_ids: set[str]
    ) -> list[tuple[str, str, str]]:
        """(doc_id, source_name, chunk_text) for every chunk in the given
        documents, scoped to the user."""
        if not doc_ids:
            return []
        async with self._session_factory() as db:
            result = await db.execute(
                select(DocumentChunkRecord, DocumentRecord.source_name)
                .join(DocumentRecord, DocumentChunkRecord.doc_id == DocumentRecord.id)
                .where(
                    DocumentRecord.user_id == user_id,
                    DocumentChunkRecord.doc_id.in_(doc_ids),
                )
            )
            rows = result.all()
        return [(chunk.doc_id, source_name, chunk.chunk_text) for chunk, source_name in rows]

    async def delete_document(self, doc_id: str) -> None:
        async with self._session_factory() as db:
            await db.execute(delete(GraphEdgeRecord).where(GraphEdgeRecord.doc_id == doc_id))
            await db.execute(delete(GraphNodeRecord).where(GraphNodeRecord.doc_id == doc_id))
            await db.commit()


class GraphRetriever:
    """Walks the entity graph outward from vector-retrieved seeds and
    returns ADDITIONAL chunks connected through shared entities — material
    that is genuinely relevant but shares no vocabulary with the query, and
    so is invisible to pure vector search.

    Hop and node caps are enforced here, in the traversal loop, not
    requested of a model.
    """

    def __init__(self, graph_store: GraphStore) -> None:
        self._graph_store = graph_store

    async def expand(
        self,
        seed_chunks: list[RetrievedChunk],
        *,
        user_id: str,
        max_hops: int = 2,
        max_nodes: int = 20,
    ) -> GraphExpansion:
        if not seed_chunks or max_hops <= 0 or max_nodes <= 0:
            return GraphExpansion(chunks=[], nodes_visited=0, hops_used=0)

        nodes = await self._graph_store.all_nodes(user_id)
        if not nodes:
            return GraphExpansion(chunks=[], nodes_visited=0, hops_used=0)

        by_name: dict[str, list[GraphNodeRecord]] = {}
        for node in nodes:
            by_name.setdefault(node.name.lower(), []).append(node)

        seed_text = " ".join(chunk.chunk_text for chunk in seed_chunks).lower()
        seed_doc_ids = {chunk.doc_id for chunk in seed_chunks}

        frontier_names = {name for name in by_name if name in seed_text}
        if not frontier_names:
            return GraphExpansion(chunks=[], nodes_visited=0, hops_used=0)

        visited_names: set[str] = set(frontier_names)
        # name -> the hop at which it was first reached, which becomes the
        # provenance score of any chunk it pulls in.
        reached_at_hop: dict[str, int] = {}
        hops_used = 0

        for hop in range(1, max_hops + 1):
            frontier_ids = {
                node.id for name in frontier_names for node in by_name.get(name, [])
            }
            edges = await self._graph_store.edges_for_nodes(user_id, frontier_ids)
            if not edges:
                break

            id_to_name = {node.id: node.name.lower() for node in nodes}
            next_names: set[str] = set()
            for edge in edges:
                for candidate_id in (edge.source_node_id, edge.target_node_id):
                    name = id_to_name.get(candidate_id)
                    if name is None or name in visited_names:
                        continue
                    next_names.add(name)

            if not next_names:
                break

            hops_used = hop
            for name in sorted(next_names):
                if len(visited_names) >= max_nodes:
                    break
                visited_names.add(name)
                reached_at_hop[name] = hop

            if len(visited_names) >= max_nodes:
                break
            frontier_names = next_names & visited_names

        if not reached_at_hop:
            return GraphExpansion(chunks=[], nodes_visited=len(visited_names), hops_used=hops_used)

        doc_hop: dict[str, int] = {}
        for name, hop in reached_at_hop.items():
            for node in by_name.get(name, []):
                if node.doc_id in seed_doc_ids:
                    continue
                doc_hop[node.doc_id] = min(doc_hop.get(node.doc_id, hop), hop)

        rows = await self._graph_store.chunks_for_docs(user_id, set(doc_hop))
        seed_texts = {chunk.chunk_text for chunk in seed_chunks}

        expanded = [
            RetrievedChunk(
                doc_id=doc_id,
                source_name=source_name,
                chunk_text=chunk_text,
                score=self._hop_affinity(doc_hop[doc_id]),
                cosine_score=self._hop_affinity(doc_hop[doc_id]),
            )
            for doc_id, source_name, chunk_text in rows
            if chunk_text not in seed_texts
        ]

        return GraphExpansion(
            chunks=expanded, nodes_visited=len(visited_names), hops_used=hops_used
        )

    @staticmethod
    def _hop_affinity(hop: int) -> float:
        return _HOP_ONE_AFFINITY * (_HOP_DECAY ** (hop - 1))
