from __future__ import annotations

import csv
import io

from nexus.core.exceptions import UnsupportedDocumentType

_TEXT_LIKE_TYPES = {"txt", "md", "code"}


async def parse_document(raw_bytes: bytes, source_type: str) -> str:
    """Dispatches by source_type. Declared async (even though every
    branch here is CPU-bound, synchronous parsing) so a future parser that
    genuinely needs I/O — e.g. shelling out to a converter — can be added
    without changing this function's signature or its callers.
    """
    normalized = source_type.lower()
    if normalized in _TEXT_LIKE_TYPES:
        return raw_bytes.decode("utf-8", errors="replace")
    if normalized == "csv":
        return _parse_csv(raw_bytes)
    if normalized == "pdf":
        return _parse_pdf(raw_bytes)
    if normalized == "docx":
        return _parse_docx(raw_bytes)
    raise UnsupportedDocumentType(f"No parser registered for source_type={source_type!r}.")


def _parse_csv(raw_bytes: bytes) -> str:
    text = raw_bytes.decode("utf-8", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return ""
    header, *data_rows = rows
    lines = [", ".join(f"{col}: {val}" for col, val in zip(header, row)) for row in data_rows]
    return "\n".join(lines)


def _parse_pdf(raw_bytes: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _parse_docx(raw_bytes: bytes) -> str:
    from docx import Document

    document = Document(io.BytesIO(raw_bytes))
    paragraphs = [p.text for p in document.paragraphs]
    return "\n".join(paragraphs)
