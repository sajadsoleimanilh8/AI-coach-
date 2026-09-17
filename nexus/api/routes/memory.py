from __future__ import annotations

from fastapi import APIRouter, Request

from nexus.api.schemas import MemoryFactsResponse, MemorySetFactRequest
from nexus.api.services import get_services
from nexus.memory.long_term import LongTermMemoryStore

router = APIRouter()


@router.get("/memory/{user_id}", response_model=MemoryFactsResponse)
async def get_memory(user_id: str, request: Request) -> MemoryFactsResponse:
    long_term_memory: LongTermMemoryStore = get_services(request).long_term_memory
    facts = await long_term_memory.get_facts(user_id)
    return MemoryFactsResponse(user_id=user_id, facts=facts)


@router.post("/memory/{user_id}", response_model=MemoryFactsResponse)
async def set_memory_fact(
    user_id: str, payload: MemorySetFactRequest, request: Request
) -> MemoryFactsResponse:
    long_term_memory: LongTermMemoryStore = get_services(request).long_term_memory
    await long_term_memory.set_fact(user_id, payload.key, payload.value)
    facts = await long_term_memory.get_facts(user_id)
    return MemoryFactsResponse(user_id=user_id, facts=facts)


@router.delete("/memory/{user_id}/item/{key}", response_model=MemoryFactsResponse)
async def delete_memory_fact(user_id: str, key: str, request: Request) -> MemoryFactsResponse:
    long_term_memory: LongTermMemoryStore = get_services(request).long_term_memory
    await long_term_memory.delete_fact(user_id, key)
    facts = await long_term_memory.get_facts(user_id)
    return MemoryFactsResponse(user_id=user_id, facts=facts)


@router.post("/memory/{user_id}/clear", response_model=MemoryFactsResponse)
async def clear_memory(user_id: str, request: Request) -> MemoryFactsResponse:
    long_term_memory: LongTermMemoryStore = get_services(request).long_term_memory
    await long_term_memory.clear(user_id)
    return MemoryFactsResponse(user_id=user_id, facts={})
