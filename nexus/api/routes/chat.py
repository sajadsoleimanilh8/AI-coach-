from __future__ import annotations

import json
import time
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from nexus.api.schemas import (
    ChatRequest,
    ChatResponse,
    CheckResultSchema,
    CitationSchema,
    ToolCallSummary,
    UsageSchema,
    VerificationReportSchema,
)
from nexus.config.settings import PersonalSettings, VerificationSettings
from nexus.core.cost_tracker import CostTracker
from nexus.core.exceptions import (
    ContextLengthExceededError,
    ModelNotFoundError,
    ProviderUnavailableError,
    SessionNotFoundError,
)
from nexus.core.latency_tracker import LatencyTracker
from nexus.core.memory import MemoryStore
from nexus.core.providers import AIProvider
from nexus.core.router import ModelRouter, RoutingDecision
from nexus.core.tool_loop import run_tool_loop
from nexus.core.types import Message, RoutingPolicy, TaskType, Usage
from nexus.core.vector_store import RetrievedChunk
from nexus.intelligence.privacy_classifier import PrivacyClassification, PrivacyClassifier
from nexus.intelligence.task_classifier import TaskClassification, TaskClassifier
from nexus.logging_setup.logger import get_logger
from nexus.models.registry import get_model
from nexus.personal.context import build_personal_context_message
from nexus.personal.profile import ProfileStore
from nexus.personal.state import PersonalStateEngine
from nexus.personal.weakness import WeaknessEngine
from nexus.rag.service import RagService
from nexus.tools.registry import ToolRegistry
from nexus.training.logger import InteractionLogger
from nexus.verification.engine import VerificationEngine
from nexus.verification.presenter import apply_verification
from nexus.verification.types import VerificationReport

router = APIRouter()
logger = get_logger("api.chat")

_TRUNCATE_CHAR_LIMIT = 300


def _truncate(text: str, limit: int = _TRUNCATE_CHAR_LIMIT) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def _incoming_messages(payload: ChatRequest) -> list[Message]:
    return [Message(role=m.role, content=m.content) for m in payload.messages]


async def _resolve_session(
    memory: MemoryStore, session_id: str | None
) -> tuple[str, list[Message]]:
    if session_id is None:
        new_id = await memory.create_session()
        return new_id, []
    try:
        history = await memory.get_history(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return session_id, history


async def _record_cost(
    cost_tracker: CostTracker, session_id: str, decision: RoutingDecision, usage: Usage
) -> float:
    model_info = get_model(decision.model_id)
    return await cost_tracker.record(
        session_id=session_id,
        provider_name=decision.provider_name,
        model_id=decision.model_id,
        usage=usage,
        model_info=model_info,
    )


def _classify_task(
    task_classifier: TaskClassifier, query_text: str, history_char_count: int
) -> TaskClassification:
    return task_classifier.classify(query_text, extra_char_count=history_char_count)


def _resolve_effective_policy(
    payload: ChatRequest,
    privacy_classification: PrivacyClassification | None,
    force_local_on_private: bool,
    session_id: str,
) -> RoutingPolicy | None:
    user_made_explicit_choice = payload.policy is not None or payload.model_id is not None
    if (
        not user_made_explicit_choice
        and force_local_on_private
        and privacy_classification is not None
        and privacy_classification.level == "private"
    ):
        logger.warning(
            "session=%s: forcing LOCAL_ONLY, privacy classifier matched private-tier "
            "signals (%s)",
            session_id,
            privacy_classification.reason,
        )
        return RoutingPolicy.LOCAL_ONLY
    return payload.policy


def _classification_reason(
    task_classification: TaskClassification | None,
    privacy_classification: PrivacyClassification | None,
) -> str | None:
    parts = []
    if task_classification is not None:
        parts.append(f"task: {task_classification.reason}")
    if privacy_classification is not None:
        parts.append(f"privacy: {privacy_classification.reason}")
    return " | ".join(parts) if parts else None


async def _apply_rag(
    rag_service: RagService,
    payload: ChatRequest,
    query_text: str,
    full_context: list[Message],
) -> tuple[list[Message], list[CitationSchema], list[RetrievedChunk] | None]:
    if not payload.use_rag:
        return full_context, [], None

    retrieved = await rag_service.retrieve(
        user_id=payload.user_id, query=query_text, top_k=payload.rag_top_k
    )
    if not retrieved:
        return full_context, [], None

    context_lines = "\n".join(f"[{i + 1}] {chunk.chunk_text}" for i, chunk in enumerate(retrieved))
    synthetic_message = Message(
        role="system",
        content=f"Relevant context from your documents:\n\n{context_lines}",
    )
    citations = [
        CitationSchema(
            doc_id=chunk.doc_id,
            source_name=chunk.source_name,
            chunk_text=_truncate(chunk.chunk_text),
            score=chunk.score,
        )
        for chunk in retrieved
    ]
    return [synthetic_message] + full_context, citations, retrieved


async def _apply_personal_context(
    payload: ChatRequest,
    decision: RoutingDecision,
    full_context: list[Message],
    personal_state_engine: PersonalStateEngine,
    weakness_engine: WeaknessEngine,
    profile_store: ProfileStore,
    personal_settings: PersonalSettings,
) -> tuple[list[Message], bool | None]:
    """Returns (possibly-updated full_context, personal_context_used)."""
    if not payload.use_personal_context:
        return full_context, None
    if not personal_settings.enabled:
        return full_context, False

    if personal_settings.local_only_context and decision.provider_name != "local":
        logger.warning(
            "personal context withheld: route resolved to non-local provider=%s while "
            "personal.local_only_context=True",
            decision.provider_name,
        )
        return full_context, False

    state = await personal_state_engine.get_state(payload.user_id)
    weaknesses = await weakness_engine.detect(payload.user_id)
    profile = await profile_store.get_profile(payload.user_id)

    message = build_personal_context_message(state, weaknesses, profile)
    if message is None:
        return full_context, False
    return [message] + full_context, True


def _should_verify(
    payload: ChatRequest, task_type: TaskType, verification_settings: VerificationSettings
) -> bool:
    if not verification_settings.enabled:
        return False
    return payload.verify or task_type.value in verification_settings.always_verify_task_types


def _to_verification_schema(report: VerificationReport) -> VerificationReportSchema:
    return VerificationReportSchema(
        checks=[
            CheckResultSchema(
                name=c.name, status=c.status.value, weight=c.weight, detail=c.detail,
                evidence=c.evidence,
            )
            for c in report.checks
        ],
        score=report.score,
        band=report.band.value,
        summary=report.summary,
        uncertainty_notes=report.uncertainty_notes,
        escalated=report.escalated,
        extra_usage=UsageSchema(
            prompt_tokens=report.extra_usage.prompt_tokens,
            completion_tokens=report.extra_usage.completion_tokens,
            total_tokens=report.extra_usage.total_tokens,
        ),
    )


async def _run_verification(
    verification_engine: VerificationEngine,
    cost_tracker: CostTracker,
    session_id: str,
    decision: RoutingDecision,
    question: str,
    answer: str,
    task_type: TaskType,
    evidence: list[RetrievedChunk] | None,
) -> tuple[str, VerificationReportSchema]:
    """Runs verification and records its token cost as a SEPARATE
    CostTracker ledger entry tagged with the answering model (principle
    4) — never folded invisibly into the main generation's cost entry, so
    "how much did verifying this session cost" stays a real, queryable
    """
    report = await verification_engine.verify(
        question=question, answer=answer, model_id=decision.model_id, task_type=task_type,
        evidence=evidence,
    )
    if report.extra_usage.total_tokens > 0:
        await cost_tracker.record(
            session_id=session_id, provider_name=decision.provider_name,
            model_id=decision.model_id, usage=report.extra_usage,
            model_info=get_model(decision.model_id),
        )
    return apply_verification(answer, report), _to_verification_schema(report)


async def _log_interaction(
    interaction_logger: InteractionLogger | None,
    *,
    session_id: str,
    payload: ChatRequest,
    prompt_messages: list[Message],
    response_text: str,
    decision: RoutingDecision,
    task_type: TaskType,
    privacy_classification: PrivacyClassification | None,
    verification: VerificationReportSchema | None,
    tools_used: list[str] | None = None,
) -> int | None:
    """Writes one training-mining record per answered request. Returns None
    whenever logging is off, which is the default — the caller then reports
    interaction_id=None and there is nothing for /api/feedback to rate.
    """
    if interaction_logger is None or not interaction_logger.enabled:
        return None
    try:
        return await interaction_logger.record(
            session_id=session_id,
            user_id=payload.user_id,
            task_type=task_type.value,
            privacy_level=privacy_classification.level if privacy_classification else "public",
            prompt_messages=prompt_messages,
            response_text=response_text,
            model_id=decision.model_id,
            provider_name=decision.provider_name,
            tools_used=tools_used or [],
            verification_band=verification.band if verification else "",
            verification_score=verification.score if verification else None,
        )
    except Exception as exc:  # noqa: BLE001 - never fail a served answer over bookkeeping
        logger.warning("interaction logging failed for session=%s: %s", session_id, exc)
        return None


@router.post("/chat", response_model=None)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse | StreamingResponse:
    memory: MemoryStore = request.app.state.memory
    model_router: ModelRouter = request.app.state.router
    cost_tracker: CostTracker = request.app.state.cost_tracker
    latency_tracker: LatencyTracker = request.app.state.latency_tracker
    task_classifier: TaskClassifier = request.app.state.task_classifier
    privacy_classifier: PrivacyClassifier = request.app.state.privacy_classifier
    rag_service: RagService = request.app.state.rag_service
    tool_registry: ToolRegistry = request.app.state.tool_registry
    settings = request.app.state.settings

    session_id, history = await _resolve_session(memory, payload.session_id)
    new_messages = _incoming_messages(payload)
    full_context = history + new_messages

    query_text = "\n".join(m.content for m in new_messages)
    history_char_count = sum(len(m.content) for m in history)

    task_type = TaskType.GENERAL
    task_classification: TaskClassification | None = None
    if settings.routing.classification.enabled:
        task_classification = _classify_task(task_classifier, query_text, history_char_count)
        task_type = task_classification.task_type

    privacy_classification: PrivacyClassification | None = None
    effective_policy = payload.policy
    if settings.routing.privacy.enabled:
        privacy_classification = privacy_classifier.classify(query_text)
        effective_policy = _resolve_effective_policy(
            payload,
            privacy_classification,
            settings.routing.privacy.force_local_on_private,
            session_id,
        )

    full_context, citations, retrieved_chunks = await _apply_rag(
        rag_service, payload, query_text, full_context
    )

    try:
        decision, provider = await model_router.route_with_failover(
            task_type=task_type,
            policy=effective_policy,
            requested_model_id=payload.model_id,
            require_tool_calling=payload.use_tools,
        )
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ProviderUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    full_context, personal_context_used = await _apply_personal_context(
        payload,
        decision,
        full_context,
        request.app.state.personal_state_engine,
        request.app.state.weakness_engine,
        request.app.state.profile_store,
        settings.personal,
    )

    for message in new_messages:
        await memory.add_message(session_id, message)

    classification_reason = _classification_reason(task_classification, privacy_classification)
    verification_engine: VerificationEngine = request.app.state.verification_engine
    should_verify = _should_verify(payload, task_type, settings.verification)

    interaction_logger: InteractionLogger | None = getattr(
        request.app.state, "interaction_logger", None
    )

    if payload.use_tools:
        return await _handle_tools_request(
            provider,
            memory,
            cost_tracker,
            latency_tracker,
            tool_registry,
            session_id,
            full_context,
            decision,
            payload,
            citations,
            classification_reason,
            task_classification,
            privacy_classification,
            settings.tools.max_iterations,
            personal_context_used,
            should_verify,
            verification_engine,
            task_type,
            query_text,
            retrieved_chunks,
            interaction_logger,
        )

    if payload.stream:
        return StreamingResponse(
            _stream_response(
                provider,
                memory,
                cost_tracker,
                latency_tracker,
                session_id,
                full_context,
                decision,
                payload.temperature,
                payload.max_tokens,
                task_classification,
                privacy_classification,
                classification_reason,
                citations if payload.use_rag else None,
                personal_context_used,
                should_verify,
                verification_engine,
                task_type,
                query_text,
                retrieved_chunks,
            ),
            media_type="text/event-stream",
        )

    start = time.monotonic()
    try:
        result = await provider.generate(
            full_context,
            model_id=decision.model_id,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
        )
    except ProviderUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ContextLengthExceededError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    latency_seconds = time.monotonic() - start

    await memory.add_message(session_id, Message(role="assistant", content=result.content))

    cost_usd = await _record_cost(cost_tracker, session_id, decision, result.usage)
    await latency_tracker.record(
        session_id=session_id,
        provider_name=decision.provider_name,
        model_id=decision.model_id,
        latency_seconds=latency_seconds,
    )

    response_content = result.content
    verification_schema = None
    if should_verify:
        response_content, verification_schema = await _run_verification(
            verification_engine, cost_tracker, session_id, decision, query_text, result.content,
            task_type, retrieved_chunks,
        )

    interaction_id = await _log_interaction(
        interaction_logger,
        session_id=session_id,
        payload=payload,
        prompt_messages=full_context,
        response_text=result.content,
        decision=decision,
        task_type=task_type,
        privacy_classification=privacy_classification,
        verification=verification_schema,
    )

    return ChatResponse(
        content=response_content,
        model_used=result.model_used,
        usage=UsageSchema(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
        ),
        session_id=session_id,
        provider_name=decision.provider_name,
        routing_reason=decision.reason,
        cost_usd=cost_usd,
        task_type=task_classification.task_type.value if task_classification else None,
        privacy_level=privacy_classification.level if privacy_classification else None,
        classification_reason=classification_reason,
        citations=citations if payload.use_rag else None,
        personal_context_used=personal_context_used,
        verification=verification_schema,
        interaction_id=interaction_id,
    )


async def _handle_tools_request(
    provider: AIProvider,
    memory: MemoryStore,
    cost_tracker: CostTracker,
    latency_tracker: LatencyTracker,
    tool_registry: ToolRegistry,
    session_id: str,
    full_context: list[Message],
    decision: RoutingDecision,
    payload: ChatRequest,
    citations: list[CitationSchema],
    classification_reason: str | None,
    task_classification: TaskClassification | None,
    privacy_classification: PrivacyClassification | None,
    max_iterations: int,
    personal_context_used: bool | None,
    should_verify: bool,
    verification_engine: VerificationEngine,
    task_type: TaskType,
    query_text: str,
    retrieved_chunks: list[RetrievedChunk] | None,
    interaction_logger: InteractionLogger | None = None,
) -> ChatResponse | StreamingResponse:
    start = time.monotonic()
    try:
        outcome = await run_tool_loop(
            provider,
            full_context,
            model_id=decision.model_id,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
            tool_registry=tool_registry,
            max_iterations=max_iterations,
        )
    except ProviderUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ContextLengthExceededError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    latency_seconds = time.monotonic() - start

    result = outcome.result
    hit_iteration_cap = outcome.hit_iteration_cap
    tool_calls_made = [
        ToolCallSummary(name=t.name, arguments=t.arguments, result_summary=t.result_summary)
        for t in outcome.tool_calls_made
    ]

    await memory.add_message(session_id, Message(role="assistant", content=result.content))

    cost_usd = await _record_cost(cost_tracker, session_id, decision, result.usage)
    await latency_tracker.record(
        session_id=session_id,
        provider_name=decision.provider_name,
        model_id=decision.model_id,
        latency_seconds=latency_seconds,
    )

    if hit_iteration_cap:
        cap_note = f"tool_loop: max_iterations={max_iterations} reached without a final answer."
        classification_reason = (
            f"{classification_reason} | {cap_note}" if classification_reason else cap_note
        )

    response_content = result.content
    verification_schema = None
    if should_verify:
        response_content, verification_schema = await _run_verification(
            verification_engine, cost_tracker, session_id, decision, query_text, result.content,
            task_type, retrieved_chunks,
        )

    interaction_id = await _log_interaction(
        interaction_logger,
        session_id=session_id,
        payload=payload,
        prompt_messages=full_context,
        response_text=result.content,
        decision=decision,
        task_type=task_type,
        privacy_classification=privacy_classification,
        verification=verification_schema,
        tools_used=[t.name for t in tool_calls_made],
    )

    response = ChatResponse(
        content=response_content,
        model_used=result.model_used,
        usage=UsageSchema(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
        ),
        session_id=session_id,
        provider_name=decision.provider_name,
        routing_reason=decision.reason,
        cost_usd=cost_usd,
        task_type=task_classification.task_type.value if task_classification else None,
        privacy_level=privacy_classification.level if privacy_classification else None,
        classification_reason=classification_reason,
        citations=citations if payload.use_rag else None,
        tool_calls_made=tool_calls_made,
        personal_context_used=personal_context_used,
        verification=verification_schema,
        interaction_id=interaction_id,
    )

    if not payload.stream:
        return response

    async def _single_chunk_stream() -> AsyncIterator[str]:
        if result.content:
            yield f"data: {json.dumps({'delta': result.content, 'done': False})}\n\n"
        yield (
            "data: "
            + json.dumps(
                {
                    "delta": "",
                    "done": True,
                    "session_id": session_id,
                    "provider_name": decision.provider_name,
                    "routing_reason": decision.reason,
                    "cost_usd": cost_usd,
                    "task_type": response.task_type,
                    "privacy_level": response.privacy_level,
                    "classification_reason": response.classification_reason,
                    "citations": [c.model_dump() for c in citations] if payload.use_rag else None,
                    "tool_calls_made": [t.model_dump() for t in tool_calls_made],
                    "personal_context_used": personal_context_used,
                    "verification": verification_schema.model_dump() if verification_schema else None,
                }
            )
            + "\n\n"
        )

    return StreamingResponse(_single_chunk_stream(), media_type="text/event-stream")


async def _stream_response(
    provider: AIProvider,
    memory: MemoryStore,
    cost_tracker: CostTracker,
    latency_tracker: LatencyTracker,
    session_id: str,
    messages: list[Message],
    decision: RoutingDecision,
    temperature: float,
    max_tokens: int | None,
    task_classification: TaskClassification | None,
    privacy_classification: PrivacyClassification | None,
    classification_reason: str | None,
    citations: list[CitationSchema] | None,
    personal_context_used: bool | None,
    should_verify: bool,
    verification_engine: VerificationEngine,
    task_type: TaskType,
    query_text: str,
    retrieved_chunks: list[RetrievedChunk] | None,
) -> AsyncIterator[str]:
    accumulated: list[str] = []
    final_usage = Usage()
    start = time.monotonic()
    try:
        async for chunk in provider.stream_generate(
            messages, model_id=decision.model_id, temperature=temperature, max_tokens=max_tokens
        ):
            if chunk.delta:
                accumulated.append(chunk.delta)
                yield f"data: {json.dumps({'delta': chunk.delta, 'done': False})}\n\n"
            if chunk.done:
                if chunk.usage:
                    final_usage = chunk.usage
                latency_seconds = time.monotonic() - start
                cost_usd = await _record_cost(cost_tracker, session_id, decision, final_usage)
                await latency_tracker.record(
                    session_id=session_id,
                    provider_name=decision.provider_name,
                    model_id=decision.model_id,
                    latency_seconds=latency_seconds,
                )

                verification_schema = None
                if should_verify:
                    full_answer = "".join(accumulated)
                    _display, verification_schema = await _run_verification(
                        verification_engine, cost_tracker, session_id, decision, query_text,
                        full_answer, task_type, retrieved_chunks,
                    )

                yield (
                    "data: "
                    + json.dumps(
                        {
                            "delta": "",
                            "done": True,
                            "session_id": session_id,
                            "provider_name": decision.provider_name,
                            "routing_reason": decision.reason,
                            "cost_usd": cost_usd,
                            "task_type": (
                                task_classification.task_type.value
                                if task_classification
                                else None
                            ),
                            "privacy_level": (
                                privacy_classification.level if privacy_classification else None
                            ),
                            "classification_reason": classification_reason,
                            "citations": (
                                [c.model_dump() for c in citations]
                                if citations is not None
                                else None
                            ),
                            "personal_context_used": personal_context_used,
                            "verification": (
                                verification_schema.model_dump() if verification_schema else None
                            ),
                        }
                    )
                    + "\n\n"
                )
    except (ProviderUnavailableError, ModelNotFoundError, ContextLengthExceededError) as exc:
        logger.warning("stream_generate failed: %s", exc)
        yield f"data: {json.dumps({'error': str(exc), 'done': True})}\n\n"
        return
    finally:
        content = "".join(accumulated)
        if content:
            await memory.add_message(session_id, Message(role="assistant", content=content))
