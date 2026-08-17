from __future__ import annotations


def chunk_text(text: str, *, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Character-based sliding window chunker."""
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    text_len = len(text)
    chunks: list[str] = []
    start = 0
    while start < text_len:
        end = min(start + chunk_size, text_len)
        piece = text[start:end]
        if piece.strip():
            chunks.append(piece)
        if end == text_len:
            break
        start += step
    return chunks
