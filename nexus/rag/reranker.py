from __future__ import annotations

import re

from nexus.core.vector_store import RetrievedChunk

_COSINE_WEIGHT = 0.75
_KEYWORD_WEIGHT = 0.25

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were",
    "with",
}


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower())) - _STOPWORDS


def _keyword_overlap_score(query_tokens: set[str], chunk_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & chunk_tokens) / len(query_tokens)


def rerank(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """final_score = 0.75 * cosine_score + 0.25 * keyword_overlap_score."""
    query_tokens = _tokenize(query)

    reranked = [
        RetrievedChunk(
            doc_id=chunk.doc_id,
            source_name=chunk.source_name,
            chunk_text=chunk.chunk_text,
            score=_COSINE_WEIGHT * chunk.cosine_score
            + _KEYWORD_WEIGHT * _keyword_overlap_score(query_tokens, _tokenize(chunk.chunk_text)),
            cosine_score=chunk.cosine_score,
        )
        for chunk in chunks
    ]
    reranked.sort(key=lambda chunk: chunk.score, reverse=True)
    return reranked
