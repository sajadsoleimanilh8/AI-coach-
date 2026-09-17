from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException, Request

from nexus.api.schemas import DocumentIngestRequest, DocumentSummarySchema
from nexus.api.services import get_services
from nexus.core.exceptions import ProviderUnavailableError, UnsupportedDocumentType
from nexus.rag.service import DocumentSummary, RagService

router = APIRouter()


def _to_schema(summary: DocumentSummary) -> DocumentSummarySchema:
    return DocumentSummarySchema(
        doc_id=summary.doc_id,
        source_name=summary.source_name,
        source_type=summary.source_type,
        chunk_count=summary.chunk_count,
        ingested_at=summary.ingested_at,
    )


@router.post("/documents", response_model=DocumentSummarySchema)
async def ingest_document(
    payload: DocumentIngestRequest, request: Request
) -> DocumentSummarySchema:
    rag_service: RagService = get_services(request).rag_service
    try:
        raw_bytes = base64.b64decode(payload.content_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid content_base64: {exc}") from exc

    try:
        summary = await rag_service.ingest(
            user_id=payload.user_id,
            source_name=payload.source_name,
            source_type=payload.source_type,
            raw_bytes=raw_bytes,
        )
    except UnsupportedDocumentType as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ProviderUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return _to_schema(summary)


@router.get("/documents", response_model=list[DocumentSummarySchema])
async def list_documents(
    request: Request, user_id: str = "default"
) -> list[DocumentSummarySchema]:
    rag_service: RagService = get_services(request).rag_service
    summaries = await rag_service.list_documents(user_id)
    return [_to_schema(summary) for summary in summaries]


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: str, request: Request, user_id: str = "default") -> None:
    rag_service: RagService = get_services(request).rag_service
    await rag_service.delete_document(user_id, doc_id)
