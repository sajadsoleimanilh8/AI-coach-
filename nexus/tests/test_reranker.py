from __future__ import annotations

from nexus.core.vector_store import RetrievedChunk
from nexus.rag.reranker import rerank


def _chunk(doc_id: str, text: str, cosine_score: float) -> RetrievedChunk:
    return RetrievedChunk(
        doc_id=doc_id, source_name="s.txt", chunk_text=text, score=cosine_score, cosine_score=cosine_score
    )


def test_keyword_overlap_can_reorder_close_cosine_scores() -> None:
    chunks = [
        _chunk("a", "Arsenal play in London and won the match.", cosine_score=0.60),
        _chunk("b", "Some unrelated text about cooking recipes.", cosine_score=0.62),
    ]

    reranked = rerank("Arsenal match London", chunks)

    # "b" edges out "a" on raw cosine, but shares zero keywords with the
    # query while "a" shares three — the blended score should flip the order.
    assert [c.doc_id for c in reranked] == ["a", "b"]


def test_final_score_matches_the_documented_formula() -> None:
    chunk = _chunk("a", "arsenal london match", cosine_score=0.4)
    [result] = rerank("arsenal london match", [chunk])

    # query tokens (stopword-light) == chunk tokens exactly -> overlap 1.0
    expected = 0.75 * 0.4 + 0.25 * 1.0
    assert result.score == expected


def test_cosine_score_is_preserved_for_explainability() -> None:
    chunk = _chunk("a", "totally different content", cosine_score=0.9)
    [result] = rerank("nothing matches here", [chunk])

    assert result.cosine_score == 0.9
    assert result.score != result.cosine_score


def test_empty_query_falls_back_to_pure_cosine_ordering() -> None:
    chunks = [_chunk("a", "text one", cosine_score=0.3), _chunk("b", "text two", cosine_score=0.7)]

    reranked = rerank("", chunks)

    assert [c.doc_id for c in reranked] == ["b", "a"]


def test_stopwords_do_not_count_toward_overlap() -> None:
    chunk = _chunk("a", "the a an is are was", cosine_score=0.5)
    [result] = rerank("the a an is are was", [chunk])

    # Every token on both sides is a stopword -> tokenize() yields an empty
    # set for both -> keyword_overlap_score is defined as 0.0 in that case.
    assert result.score == 0.75 * 0.5
