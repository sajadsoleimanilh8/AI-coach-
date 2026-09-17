from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from nexus.api.schemas import FeedbackRequest, FeedbackResponse
from nexus.api.services import get_services
from nexus.training.logger import InteractionLogger

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def record_feedback(payload: FeedbackRequest, request: Request) -> FeedbackResponse:
    logger: InteractionLogger | None = get_services(request).interaction_logger
    if logger is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Interaction logging is not configured, so there are no interactions to rate. "
                "Set training.log_interactions=true to enable it."
            ),
        )

    recorded = await logger.set_feedback(payload.interaction_id, payload.feedback)
    if not recorded:
        raise HTTPException(
            status_code=404, detail=f"No interaction with id={payload.interaction_id}."
        )

    return FeedbackResponse(
        interaction_id=payload.interaction_id, feedback=payload.feedback, recorded=True
    )
