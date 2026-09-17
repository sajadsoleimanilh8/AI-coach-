from __future__ import annotations

import pytest

from nexus.core.providers import AIProvider
from nexus.core.router import RoutingDecision
from nexus.core.types import (
    GenerationChunk,
    GenerationResult,
    ModelInfo,
    RoutingPolicy,
    TaskType,
    Usage,
)
from nexus.core.vector_store import RetrievedChunk
from nexus.memory.storage import create_async_db_engine
from nexus.rag.graph import EntityExtractor, GraphRetriever, GraphStore, Triple


class _ScriptedProvider(AIProvider):
    name = "fake"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.call_count = 0

    async def generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        self.call_count += 1
        content = self._responses.pop(0) if self._responses else ""
        return GenerationResult(
            content=content, model_used=model_id, provider_name=self.name,
            usage=Usage(prompt_tokens=5, completion_tokens=5),
        )

    async def stream_generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        yield GenerationChunk(delta="", done=True, usage=Usage())

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> bool:
        return True

    def count_tokens(self, text: str, *, model_id: str) -> int:
        return len(text)


class _FakeRouter:
    def __init__(self, provider: AIProvider) -> None:
        self._provider = provider

    async def route_with_failover(self, **kwargs):
        decision = RoutingDecision(
            provider_name=self._provider.name, model_id="fake-model",
            task_type=kwargs.get("task_type", TaskType.GENERAL),
            policy=RoutingPolicy.BALANCED, reason="fake",
        )
        return decision, self._provider


async def _store(tmp_path) -> GraphStore:
    store = GraphStore(create_async_db_engine(str(tmp_path / "nexus.db")))
    await store.init()
    return store


def _chunk(doc_id: str, text: str, source_name: str = "seed.txt") -> RetrievedChunk:
    return RetrievedChunk(
        doc_id=doc_id, source_name=source_name, chunk_text=text, score=0.9, cosine_score=0.9
    )


@pytest.mark.asyncio
async def test_extraction_produces_triples() -> None:
    provider = _ScriptedProvider(
        ['{"triples": [{"source": "Router", "relation": "reads", "target": "Matrix"}]}']
    )
    extractor = EntityExtractor(_FakeRouter(provider), enabled=True)

    triples, usage = await extractor.extract(["The router reads the matrix."])

    assert len(triples) == 1
    assert triples[0].source == "Router"
    assert triples[0].relation == "reads"
    assert triples[0].target == "Matrix"
    assert usage.total_tokens > 0


@pytest.mark.asyncio
async def test_extraction_makes_no_provider_call_when_disabled() -> None:
    provider = _ScriptedProvider(['{"triples": []}'])
    extractor = EntityExtractor(_FakeRouter(provider), enabled=False)

    triples, usage = await extractor.extract(["Anything at all."])

    assert triples == []
    assert provider.call_count == 0
    assert usage.total_tokens == 0


@pytest.mark.asyncio
async def test_extraction_survives_unparseable_output() -> None:
    extractor = EntityExtractor(_FakeRouter(_ScriptedProvider(["not json at all"])), enabled=True)

    triples, _usage = await extractor.extract(["text"])

    assert triples == []


@pytest.mark.asyncio
async def test_extraction_tolerates_prose_around_the_json() -> None:
    provider = _ScriptedProvider(
        ['Sure! {"triples": [{"source": "A", "relation": "r", "target": "B"}]} Hope that helps.']
    )
    extractor = EntityExtractor(_FakeRouter(provider), enabled=True)

    triples, _usage = await extractor.extract(["text"])

    assert len(triples) == 1


@pytest.mark.asyncio
async def test_extraction_drops_self_referential_and_incomplete_triples() -> None:
    provider = _ScriptedProvider(
        [
            '{"triples": ['
            '{"source": "A", "relation": "r", "target": "A"},'
            '{"source": "", "relation": "r", "target": "B"},'
            '{"source": "C", "relation": "r", "target": "D"}]}'
        ]
    )
    extractor = EntityExtractor(_FakeRouter(provider), enabled=True)

    triples, _usage = await extractor.extract(["text"])

    assert [(t.source, t.target) for t in triples] == [("C", "D")]


@pytest.mark.asyncio
async def test_extraction_batches_chunks() -> None:
    provider = _ScriptedProvider(['{"triples": []}'] * 10)
    extractor = EntityExtractor(_FakeRouter(provider), enabled=True, batch_size=4)

    await extractor.extract(["chunk"] * 9)

    assert provider.call_count == 3  # ceil(9 / 4)


@pytest.mark.asyncio
async def test_add_triples_creates_nodes_and_edges(tmp_path) -> None:
    store = await _store(tmp_path)

    written = await store.add_triples(
        user_id="u1", doc_id="doc1",
        triples=[Triple(source="alpha", relation="calls", target="beta")],
    )

    assert written == 1
    nodes = await store.all_nodes("u1")
    assert sorted(n.name for n in nodes) == ["alpha", "beta"]
    edges = await store.edges_for_nodes("u1", {n.id for n in nodes})
    assert len(edges) == 1
    assert edges[0].relation == "calls"


@pytest.mark.asyncio
async def test_repeated_entity_in_one_document_reuses_its_node(tmp_path) -> None:
    store = await _store(tmp_path)

    await store.add_triples(
        user_id="u1", doc_id="doc1",
        triples=[
            Triple(source="alpha", relation="calls", target="beta"),
            Triple(source="alpha", relation="calls", target="gamma"),
        ],
    )

    assert len([n for n in await store.all_nodes("u1") if n.name == "alpha"]) == 1


@pytest.mark.asyncio
async def test_graph_is_scoped_per_user(tmp_path) -> None:
    store = await _store(tmp_path)
    await store.add_triples(
        user_id="u1", doc_id="doc1", triples=[Triple(source="a", relation="r", target="b")]
    )

    assert await store.all_nodes("u2") == []


@pytest.mark.asyncio
async def test_expansion_finds_a_chunk_sharing_no_query_vocabulary(tmp_path) -> None:
    """The whole point of graph RAG: material that vector search cannot
    reach because it shares no words with the query."""
    from nexus.memory.sqlite_vector_store import SqliteVectorStore
    from nexus.memory.storage import DocumentRecord, make_session_factory

    engine = create_async_db_engine(str(tmp_path / "nexus.db"))
    store = GraphStore(engine)
    await store.init()
    vector_store = SqliteVectorStore(engine)
    await vector_store.init()

    async with make_session_factory(engine)() as db:
        db.add(DocumentRecord(id="doc1", user_id="u1", source_name="seed.txt", source_type="txt", chunk_count=1, ingested_at=0.0))
        db.add(DocumentRecord(id="doc2", user_id="u1", source_name="far.txt", source_type="txt", chunk_count=1, ingested_at=0.0))
        await db.commit()
    await vector_store.add_chunks(doc_id="doc2", chunks=["Entirely different wording here."], embeddings=[[1.0]])

    await store.add_triples(user_id="u1", doc_id="doc1", triples=[Triple(source="bitsandbytes", relation="targets", target="sm_120")])
    await store.add_triples(user_id="u1", doc_id="doc2", triples=[Triple(source="sm_120", relation="describes", target="blackwell")])

    retriever = GraphRetriever(store)
    expansion = await retriever.expand(
        [_chunk("doc1", "The bitsandbytes library ships kernels.")], user_id="u1"
    )

    assert [c.source_name for c in expansion.chunks] == ["far.txt"]
    assert expansion.chunks[0].chunk_text == "Entirely different wording here."


@pytest.mark.asyncio
async def test_expansion_returns_nothing_without_a_graph(tmp_path) -> None:
    retriever = GraphRetriever(await _store(tmp_path))

    expansion = await retriever.expand([_chunk("doc1", "anything")], user_id="u1")

    assert expansion.chunks == []
    assert expansion.nodes_visited == 0


@pytest.mark.asyncio
async def test_expansion_returns_nothing_when_no_entity_appears_in_the_seed(tmp_path) -> None:
    store = await _store(tmp_path)
    await store.add_triples(
        user_id="u1", doc_id="doc1", triples=[Triple(source="gardening", relation="in", target="spring")]
    )
    retriever = GraphRetriever(store)

    expansion = await retriever.expand(
        [_chunk("doc1", "completely unrelated seed text")], user_id="u1"
    )

    assert expansion.chunks == []


@pytest.mark.asyncio
async def test_hop_cap_is_enforced(tmp_path) -> None:
    store = await _store(tmp_path)
    await store.add_triples(user_id="u1", doc_id="d1", triples=[Triple(source="alpha", relation="c", target="beta")])
    await store.add_triples(user_id="u1", doc_id="d2", triples=[Triple(source="beta", relation="c", target="gamma")])
    await store.add_triples(user_id="u1", doc_id="d3", triples=[Triple(source="gamma", relation="c", target="delta")])
    retriever = GraphRetriever(store)

    one_hop = await retriever.expand([_chunk("d1", "alpha starts here")], user_id="u1", max_hops=1)
    two_hop = await retriever.expand([_chunk("d1", "alpha starts here")], user_id="u1", max_hops=2)

    assert one_hop.hops_used <= 1
    assert two_hop.hops_used <= 2
    assert two_hop.nodes_visited > one_hop.nodes_visited


@pytest.mark.asyncio
async def test_node_cap_is_enforced(tmp_path) -> None:
    store = await _store(tmp_path)
    for index in range(10):
        await store.add_triples(
            user_id="u1", doc_id=f"d{index}",
            triples=[Triple(source="hub", relation="c", target=f"spoke{index}")],
        )
    retriever = GraphRetriever(store)

    expansion = await retriever.expand(
        [_chunk("d0", "hub is mentioned here")], user_id="u1", max_hops=3, max_nodes=4
    )

    assert expansion.nodes_visited <= 4


@pytest.mark.asyncio
async def test_zero_caps_disable_expansion_entirely(tmp_path) -> None:
    store = await _store(tmp_path)
    await store.add_triples(user_id="u1", doc_id="d1", triples=[Triple(source="alpha", relation="c", target="beta")])
    retriever = GraphRetriever(store)

    assert (await retriever.expand([_chunk("d1", "alpha")], user_id="u1", max_hops=0)).chunks == []
    assert (await retriever.expand([_chunk("d1", "alpha")], user_id="u1", max_nodes=0)).chunks == []


@pytest.mark.asyncio
async def test_deeper_hops_score_lower_than_nearer_ones(tmp_path) -> None:
    retriever = GraphRetriever(await _store(tmp_path))

    assert retriever._hop_affinity(1) > retriever._hop_affinity(2)  # noqa: SLF001


@pytest.mark.asyncio
async def test_disabled_graph_means_zero_behavior_change_in_rag_service(tmp_path) -> None:
    """RagService with graph_enabled False must take exactly the pre-Phase-14
    path: no extractor call, no graph query, identical results."""
    from nexus.evaluation.runner import _HashingBagOfWordsEmbeddingProvider
    from nexus.memory.sqlite_vector_store import SqliteVectorStore
    from nexus.rag.service import RagService

    engine = create_async_db_engine(str(tmp_path / "nexus.db"))
    vector_store = SqliteVectorStore(engine)
    await vector_store.init()
    store = GraphStore(engine)
    await store.init()

    provider = _ScriptedProvider(['{"triples": [{"source": "A", "relation": "r", "target": "B"}]}'])
    service = RagService(
        engine,
        _HashingBagOfWordsEmbeddingProvider(),
        vector_store,
        entity_extractor=EntityExtractor(_FakeRouter(provider), enabled=False),
        graph_store=store,
        graph_retriever=GraphRetriever(store),
        graph_enabled=False,
    )
    await service.init()

    await service.ingest(
        user_id="u1", source_name="a.txt", source_type="txt", raw_bytes=b"alpha beta gamma"
    )
    results = await service.retrieve(user_id="u1", query="alpha", top_k=5)

    assert provider.call_count == 0
    assert await store.all_nodes("u1") == []
    assert len(results) == 1
