from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request

from nexus.agents.base import AgentContext
from nexus.agents.registry import AGENT_REGISTRY, get_agent
from nexus.agents.runtime import AgentRuntime
from nexus.api.schemas import (
    AgentInfoSchema,
    AgentRunRequest,
    AgentRunResponse,
    AgentStepSchema,
    DelegationStepSchema,
    UsageSchema,
)
from nexus.api.services import get_services
from nexus.core.exceptions import (
    AgentNotFoundError,
    ContextLengthExceededError,
    ModelNotFoundError,
    ProviderUnavailableError,
)
from nexus.health.analyzer import HealthAnalyzer
from nexus.memory.long_term import LongTermMemoryStore
from nexus.personal.profile import ProfileStore
from nexus.personal.state import PersonalStateEngine
from nexus.personal.weakness import WeaknessEngine
from nexus.rag.service import RagService
from nexus.sports.adapter import SportsDataAdapter

router = APIRouter()


@router.get("/agents", response_model=list[AgentInfoSchema])
async def list_agents(request: Request) -> list[AgentInfoSchema]:
    enabled = set(get_services(request).settings.agents.enabled)
    return [
        AgentInfoSchema(name=cls.name, description=cls.description, allowed_tools=cls.allowed_tools)
        for cls in AGENT_REGISTRY.values()
        if cls.name in enabled
    ]


@router.post("/agents/{name}/run", response_model=AgentRunResponse)
async def run_agent(name: str, payload: AgentRunRequest, request: Request) -> AgentRunResponse:
    settings = get_services(request).settings
    try:
        agent = get_agent(name, enabled=settings.agents.enabled)
    except AgentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    runtime: AgentRuntime = get_services(request).agent_runtime
    rag_service: RagService = get_services(request).rag_service
    long_term_memory: LongTermMemoryStore = get_services(request).long_term_memory
    personal_state: PersonalStateEngine = get_services(request).personal_state_engine
    weakness_engine: WeaknessEngine = get_services(request).weakness_engine
    profile_store: ProfileStore = get_services(request).profile_store
    health_analyzer: HealthAnalyzer = get_services(request).health_analyzer
    sports_adapter: SportsDataAdapter = get_services(request).sports_adapter

    extra: dict[str, object] = {
        "weakness_engine": weakness_engine,
        "profile_store": profile_store,
        "health_analyzer": health_analyzer,
        "sports_adapter": sports_adapter,
    }
    if payload.max_iterations is not None:
        extra["max_iterations"] = payload.max_iterations
    if payload.match_id is not None:
        extra["match_id"] = payload.match_id
    if payload.player_id is not None:
        extra["player_id"] = payload.player_id

    context = AgentContext(
        goal=payload.goal,
        user_id=payload.user_id,
        session_id=payload.session_id,
        rag_service=rag_service,
        long_term_memory=long_term_memory,
        personal_state=personal_state,
        extra=extra,
    )

    try:
        result = await runtime.run(agent, context)
    except ProviderUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ContextLengthExceededError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    return AgentRunResponse(
        goal=result.goal,
        final_answer=result.final_answer,
        steps=[AgentStepSchema(**asdict(step)) for step in result.steps],
        completed=result.completed,
        iterations_used=result.iterations_used,
        usage=UsageSchema(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
        ),
        cost_usd=result.cost_usd,
        model_used=result.model_used,
        provider_name=result.provider_name,
        delegation_steps=[
            DelegationStepSchema(
                agent_name=step.agent_name,
                sub_goal=step.sub_goal,
                result_summary=step.result_summary,
                depth=step.depth,
            )
            for step in result.delegation_steps
        ],
        rounds_used=result.rounds_used,
    )
