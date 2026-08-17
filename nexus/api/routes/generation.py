from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from nexus.api.schemas import GenerationRequest, PersonalizedPlanResponse, UsageSchema
from nexus.core.exceptions import (
    ContextLengthExceededError,
    ModelNotFoundError,
    ProviderUnavailableError,
)
from nexus.generation.service import PersonalizedGenerator

router = APIRouter()

_VALID_REQUEST_TYPES = {"workout", "recovery", "study", "daily_plan", "nutrition"}


@router.post("/generate/{request_type}", response_model=PersonalizedPlanResponse)
async def generate_plan(
    request_type: str, payload: GenerationRequest, request: Request
) -> PersonalizedPlanResponse:
    if request_type not in _VALID_REQUEST_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown request_type={request_type!r}; expected one of {sorted(_VALID_REQUEST_TYPES)}.",
        )

    generator: PersonalizedGenerator = request.app.state.personalized_generator
    try:
        plan = await generator.generate(
            user_id=payload.user_id, request_type=request_type, constraints=payload.constraints
        )
    except ProviderUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ContextLengthExceededError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    return PersonalizedPlanResponse(
        request_type=plan.request_type,
        content=plan.content,
        brief_rationale=plan.brief_rationale,
        addressed_weaknesses=plan.addressed_weaknesses,
        target_intensity=plan.target_intensity,
        personalized=plan.personalized,
        model_used=plan.model_used,
        usage=UsageSchema(
            prompt_tokens=plan.usage.prompt_tokens,
            completion_tokens=plan.usage.completion_tokens,
            total_tokens=plan.usage.total_tokens,
        ),
    )
