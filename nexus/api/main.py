from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from nexus.agents.registry import get_agent
from nexus.agents.runtime import AgentRuntime
from nexus.api.routes import agents as agents_routes
from nexus.api.routes import chat, documents, health
from nexus.api.routes import evaluation as evaluation_routes
from nexus.api.routes import feedback as feedback_routes
from nexus.api.routes import generation as generation_routes
from nexus.api.routes import health_intel as health_intel_routes
from nexus.api.routes import memory as memory_routes
from nexus.api.routes import personal as personal_routes
from nexus.api.routes import psychology as psychology_routes
from nexus.api.routes import sports as sports_routes
from nexus.config.settings import get_settings
from nexus.core.cost_tracker import CostTracker
from nexus.core.embeddings import EmbeddingProvider
from nexus.core.latency_tracker import LatencyTracker
from nexus.core.provider_manager import ProviderManager
from nexus.core.providers import AIProvider
from nexus.core.router import ModelRouter
from nexus.core.types import RoutingPolicy
from nexus.core.vector_store import VectorStore
from nexus.evaluation.store import EvalStore
from nexus.generation.planner import BriefBuilder
from nexus.generation.service import PersonalizedGenerator
from nexus.health.analyzer import HealthAnalyzer
from nexus.intelligence.capability_matrix import CapabilityMatrix
from nexus.intelligence.privacy_classifier import PrivacyClassifier, PrivacyClassifierConfig
from nexus.intelligence.self_eval import SelfEvaluator
from nexus.intelligence.task_classifier import TaskClassifier, build_signal_map_from_names
from nexus.logging_setup.logger import configure_logging, get_logger
from nexus.memory.long_term import LongTermMemoryStore
from nexus.memory.short_term import ShortTermMemoryStore
from nexus.memory.sqlite_vector_store import SqliteVectorStore
from nexus.memory.storage import create_async_db_engine
from nexus.models.cloud.anthropic_provider import AnthropicProvider
from nexus.models.cloud.gemini_provider import GeminiProvider
from nexus.models.cloud.openai_embedding_provider import OpenAIEmbeddingProvider
from nexus.models.cloud.openai_provider import OpenAIProvider
from nexus.models.local.embedding_runtime import OllamaEmbeddingProvider
from nexus.models.local.runtime import OllamaRuntime
from nexus.models.registry import list_models as list_registered_models
from nexus.personal.baseline import BaselineCalculator
from nexus.personal.forecast import ScenarioSimulator, StateForecaster
from nexus.personal.profile import ProfileStore
from nexus.personal.state import PersonalStateEngine
from nexus.personal.weakness import WeaknessEngine
from nexus.rag.graph import EntityExtractor, GraphRetriever, GraphStore
from nexus.rag.service import RagService
from nexus.sports.adapter import HttpSportsDataAdapter
from nexus.sports.coach import CoachAssistant
from nexus.sports.prematch_health import PreMatchHealthClient
from nexus.sports.psychology_adapter import PsychologyClient
from nexus.sports.video import VideoAnalysisService
from nexus.tools.database_tool import DatabaseTool
from nexus.tools.files_tool import FileSystemTool
from nexus.tools.github_tool import GitHubTool
from nexus.tools.python_exec import PythonExecutionTool
from nexus.tools.registry import Tool, ToolRegistry
from nexus.tools.web_search import WebSearchTool
from nexus.training.logger import InteractionLogger
from nexus.verification.engine import VerificationEngine
from nexus.verification.fact_checker import FactChecker
from nexus.verification.judge import MultiModelJudge

logger = get_logger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(level=settings.server.log_level, json_output=settings.server.log_json)

    providers: dict[str, AIProvider] = {
        "local": OllamaRuntime(
            settings.local.base_url,
            timeout_seconds=settings.local.request_timeout_seconds,
        ),
    }
    if settings.cloud.openai.enabled and settings.cloud.openai.api_key:
        providers["openai"] = OpenAIProvider(
            settings.cloud.openai.api_key,
            base_url=settings.cloud.openai.base_url,
            timeout_seconds=settings.cloud.openai.timeout_seconds,
        )
    if settings.cloud.anthropic.enabled and settings.cloud.anthropic.api_key:
        providers["anthropic"] = AnthropicProvider(
            settings.cloud.anthropic.api_key,
            timeout_seconds=settings.cloud.anthropic.timeout_seconds,
        )
    if settings.cloud.gemini.enabled and settings.cloud.gemini.api_key:
        providers["gemini"] = GeminiProvider(
            settings.cloud.gemini.api_key,
            timeout_seconds=settings.cloud.gemini.timeout_seconds,
        )
    provider_manager = ProviderManager(providers)

    engine = create_async_db_engine(settings.memory.database_path)
    memory = ShortTermMemoryStore(
        engine,
        window_size=settings.memory.session_window_size,
        ttl_seconds=settings.memory.session_ttl_minutes * 60,
    )
    await memory.init()

    cost_tracker = CostTracker(engine)
    await cost_tracker.init()

    latency_tracker = LatencyTracker(
        engine,
        window_size=settings.routing.latency.window_size,
        min_samples=settings.routing.latency.min_samples,
    )
    await latency_tracker.init()

    capability_matrix = CapabilityMatrix(
        engine,
        min_samples=settings.capability_learning.min_samples,
        max_learned_weight=settings.capability_learning.max_learned_weight,
    )
    await capability_matrix.init()
    if settings.capability_learning.enabled:
        await capability_matrix.refresh()

    router = ModelRouter(
        provider_manager,
        list_registered_models(),
        default_policy=RoutingPolicy(settings.routing.default_policy),
        latency_tracker=latency_tracker,
        capability_matrix=capability_matrix if settings.capability_learning.enabled else None,
    )

    interaction_logger = InteractionLogger(
        engine,
        enabled=settings.training.log_interactions,
        retention_days=settings.training.interaction_retention_days,
    )
    await interaction_logger.init()

    task_classifier = TaskClassifier(
        build_signal_map_from_names(settings.routing.classification.extra_signals),
        min_confidence=settings.routing.classification.min_confidence,
        long_context_token_threshold=settings.routing.classification.long_context_token_threshold,
    )
    privacy_classifier = PrivacyClassifier(
        PrivacyClassifierConfig(
            extra_private_patterns=list(settings.routing.privacy.extra_private_patterns)
        )
    )

    embedding_provider: EmbeddingProvider
    if settings.rag.embedding_provider == "openai" and settings.cloud.openai.api_key:
        embedding_provider = OpenAIEmbeddingProvider(settings.cloud.openai.api_key)
    else:
        embedding_provider = OllamaEmbeddingProvider(
            settings.local.base_url,
            model_id=settings.local.models.embeddings,
            timeout_seconds=settings.local.request_timeout_seconds,
        )

    vector_store: VectorStore = SqliteVectorStore(engine)
    await vector_store.init()

    graph_store = GraphStore(engine)
    await graph_store.init()
    graph_retriever = GraphRetriever(graph_store)
    entity_extractor = EntityExtractor(router, enabled=settings.rag.graph.enabled)

    rag_service = RagService(
        engine,
        embedding_provider,
        vector_store,
        chunk_size=settings.rag.chunk_size,
        chunk_overlap=settings.rag.chunk_overlap,
        entity_extractor=entity_extractor,
        graph_store=graph_store,
        graph_retriever=graph_retriever,
        graph_enabled=settings.rag.graph.enabled,
        graph_max_hops=settings.rag.graph.max_hops,
        graph_max_nodes=settings.rag.graph.max_nodes,
    )
    await rag_service.init()

    long_term_memory = LongTermMemoryStore(engine)
    await long_term_memory.init()

    personal_state_engine = PersonalStateEngine(
        engine,
        half_life_days=settings.personal.state_half_life_days,
        recent_window_days=settings.personal.state_recent_window_days,
    )
    await personal_state_engine.init()

    baseline_calculator = BaselineCalculator(
        engine,
        window_days=settings.personal.baseline_window_days,
        recent_window_days=settings.personal.state_recent_window_days,
        min_samples=settings.personal.baseline_min_samples,
    )

    weakness_engine = WeaknessEngine(
        personal_state_engine,
        baseline_calculator,
        min_confidence=settings.personal.weakness_min_confidence,
        min_deviation=settings.personal.weakness_min_deviation,
        trend_window_days=settings.personal.baseline_window_days,
        trend_min_samples=settings.personal.trend_min_samples,
    )

    state_forecaster = StateForecaster(
        personal_state_engine,
        max_horizon_days=settings.personal.forecast.max_horizon_days,
        min_trend_confidence=settings.personal.forecast.min_trend_confidence,
        trend_window_days=settings.personal.baseline_window_days,
        trend_min_samples=settings.personal.trend_min_samples,
    )
    scenario_simulator = ScenarioSimulator(
        personal_state_engine,
        baseline_calculator,
        weakness_engine,
        min_deviation=settings.personal.weakness_min_deviation,
    )

    profile_store = ProfileStore(engine)
    await profile_store.init()

    health_analyzer = HealthAnalyzer(
        personal_state_engine,
        baseline_calculator,
        weakness_engine,
        min_sample_size=settings.health.min_sample_size,
    )

    brief_builder = BriefBuilder(
        personal_state_engine,
        weakness_engine,
        health_analyzer,
        profile_store,
        default_intensity=settings.generation.default_intensity,
        min_intensity=settings.generation.min_intensity,
        max_intensity=settings.generation.max_intensity,
    )

    sports_adapter = HttpSportsDataAdapter(
        settings.sports.backend_base_url,
        timeout_seconds=settings.sports.request_timeout_seconds,
    )

    enabled_tool_names = set(settings.tools.enabled)
    available_tools: dict[str, Tool] = {
        "web_search": WebSearchTool(max_results=settings.tools.web_search_max_results),
        "python": PythonExecutionTool(timeout_seconds=settings.tools.python_timeout_seconds),
        "files": FileSystemTool(allowed_root=settings.tools.files_allowed_root),
        "database": DatabaseTool(engine),
    }
    if settings.tools.github.token:
        available_tools["github"] = GitHubTool(settings.tools.github.token)
    tools = {name: tool for name, tool in available_tools.items() if name in enabled_tool_names}
    tool_registry = ToolRegistry(tools)

    fact_checker = FactChecker(router, max_claims=settings.verification.max_claims)
    judge = MultiModelJudge(router)
    verification_engine = VerificationEngine(
        router,
        fact_checker,
        judge,
        escalate_below=settings.verification.escalate_below,
        enable_fact_check=settings.verification.enable_fact_check,
        enable_judge=settings.verification.enable_judge,
    )

    agent_runtime = AgentRuntime(
        router,
        tool_registry,
        max_iterations=settings.agents.max_iterations,
        cost_tracker=cost_tracker,
        verification_engine=verification_engine,
        agent_factory=lambda name: get_agent(name, enabled=settings.agents.enabled),
        enabled_agents=list(settings.agents.enabled),
        orchestration_max_depth=settings.agents.orchestration.max_depth,
        orchestration_max_total_delegations=settings.agents.orchestration.max_total_delegations,
    )

    self_evaluator = SelfEvaluator(router, enabled=settings.self_eval.enabled)

    personalized_generator = PersonalizedGenerator(router, brief_builder)
    prematch_health_client = PreMatchHealthClient(
        settings.sports.backend_base_url,
        timeout_seconds=settings.sports.request_timeout_seconds,
    )
    psychology_client = PsychologyClient(
        settings.sports.backend_base_url,
        timeout_seconds=settings.sports.request_timeout_seconds,
    )
    coach_assistant = CoachAssistant(
        router, sports_adapter, prematch_health_client, psychology_client
    )
    video_analysis_service = VideoAnalysisService(
        settings.sports.backend_base_url,
        coach_assistant,
        timeout_seconds=settings.sports.request_timeout_seconds,
    )

    eval_store = EvalStore(engine)
    await eval_store.init()

    app.state.settings = settings
    app.state.provider_manager = provider_manager
    app.state.provider = providers["local"]
    app.state.memory = memory
    app.state.cost_tracker = cost_tracker
    app.state.latency_tracker = latency_tracker
    app.state.task_classifier = task_classifier
    app.state.privacy_classifier = privacy_classifier
    app.state.embedding_provider = embedding_provider
    app.state.vector_store = vector_store
    app.state.rag_service = rag_service
    app.state.long_term_memory = long_term_memory
    app.state.tool_registry = tool_registry
    app.state.router = router
    app.state.personal_state_engine = personal_state_engine
    app.state.baseline_calculator = baseline_calculator
    app.state.weakness_engine = weakness_engine
    app.state.profile_store = profile_store
    app.state.agent_runtime = agent_runtime
    app.state.health_analyzer = health_analyzer
    app.state.brief_builder = brief_builder
    app.state.sports_adapter = sports_adapter
    app.state.prematch_health_client = prematch_health_client
    app.state.psychology_client = psychology_client
    app.state.personalized_generator = personalized_generator
    app.state.coach_assistant = coach_assistant
    app.state.fact_checker = fact_checker
    app.state.judge = judge
    app.state.verification_engine = verification_engine
    app.state.eval_store = eval_store
    app.state.capability_matrix = capability_matrix
    app.state.interaction_logger = interaction_logger
    app.state.graph_store = graph_store
    app.state.graph_retriever = graph_retriever
    app.state.entity_extractor = entity_extractor
    app.state.state_forecaster = state_forecaster
    app.state.scenario_simulator = scenario_simulator
    app.state.self_evaluator = self_evaluator
    app.state.video_analysis_service = video_analysis_service

    logger.info(
        "NEXUS started (providers=%s, tools=%s)",
        ",".join(sorted(providers.keys())),
        ",".join(sorted(tools.keys())) or "none",
    )
    yield

    await engine.dispose()


_default_cors_origins = "http://localhost:3000,http://127.0.0.1:3000"
NEXUS_CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("NEXUS_CORS_ALLOWED_ORIGINS", _default_cors_origins).split(",")
    if origin.strip()
]


def create_app() -> FastAPI:
    app = FastAPI(title="NEXUS", version="0.7.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=NEXUS_CORS_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(chat.router, prefix="/api")
    app.include_router(health.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")
    app.include_router(memory_routes.router, prefix="/api")
    app.include_router(agents_routes.router, prefix="/api")
    app.include_router(personal_routes.router, prefix="/api")
    app.include_router(health_intel_routes.router, prefix="/api")
    app.include_router(generation_routes.router, prefix="/api")
    app.include_router(sports_routes.router, prefix="/api")
    app.include_router(psychology_routes.router, prefix="/api")
    app.include_router(evaluation_routes.router, prefix="/api")
    app.include_router(feedback_routes.router, prefix="/api")
    return app


app = create_app()
