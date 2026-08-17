from __future__ import annotations

import pytest

from nexus.rag.chunking import chunk_text


def test_empty_text_returns_no_chunks() -> None:
    assert chunk_text("") == []


def test_text_shorter_than_chunk_size_returns_single_chunk() -> None:
    chunks = chunk_text("hello world", chunk_size=100, overlap=10)
    assert chunks == ["hello world"]


def test_chunk_size_is_respected() -> None:
    text = "a" * 250
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_consecutive_chunks_overlap() -> None:
    text = "0123456789" * 10
    chunks = chunk_text(text, chunk_size=40, overlap=10)

    for first, second in zip(chunks, chunks[1:]):
        assert first[-10:] == second[:10]


def test_full_text_is_covered_by_chunks() -> None:
    text = "x" * 97
    chunks = chunk_text(text, chunk_size=30, overlap=5)
    assert "".join(chunks)[-1] == text[-1]
    reconstructed_end = chunks[-1]
    assert text.endswith(reconstructed_end)


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=50, overlap=50)


def test_whitespace_only_chunks_are_dropped() -> None:
    text = "word1" + " " * 20 + "word2"
    chunks = chunk_text(text, chunk_size=5, overlap=1)
    assert all(chunk.strip() for chunk in chunks)
