"""Construction and typed access for every long-lived NEXUS component.

Previously ~40 objects were built inline in main.py's lifespan() and then
stored as individual untyped attributes on `app.state`, so every route did
`request.app.state.<anything>` with no static checking and adding a component
meant editing three places in one 300-line function. Now:

  * build_services() constructs everything, in the same order as before.
  * NexusServices is the one typed container, stored as app.state.services.
  * get_services(request) is how routes reach it.

Tests replace a component by assigning to it after startup, e.g.
`app.state.services.router = fake_router`, which is why this dataclass is
deliberately NOT frozen.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncEngine

from nexus.agents.registry import get_agent
from nexus.agents.runtime import AgentRuntime
from nexus.config.settings import NexusSettings
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
from nexus.logging_setup.logger import get_logger
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

logger = get_logger("api.services")


@dataclass
class NexusServices:
    engine: AsyncEngine
    settings: NexusSettings
    provider_manager: ProviderManager
    provider: AIProvider
    memory: ShortTermMemoryStore
    cost_tracker: CostTracker
    latency_tracker: LatencyTracker
    task_classifier: TaskClassifier
    privacy_classifier: PrivacyClassifier
    embedding_provider: EmbeddingProvider
    vector_store: VectorStore
    rag_service: RagService
    long_term_memory: LongTermMemoryStore
    tool_registry: ToolRegistry
    router: ModelRouter
    personal_state_engine: PersonalStateEngine
    baseline_calculator: BaselineCalculator
    weakness_engine: WeaknessEngine
    profile_store: ProfileStore
    agent_runtime: AgentRuntime
    health_analyzer: HealthAnalyzer
    brief_builder: BriefBuilder
    sports_adapter: HttpSportsDataAdapter
    prematch_health_client: PreMatchHealthClient
    psychology_client: PsychologyClient
    personalized_generator: PersonalizedGenerator
    coach_assistant: CoachAssistant
    fact_checker: FactChecker
    judge: MultiModelJudge
    verification_engine: VerificationEngine
    eval_store: EvalStore
    capability_matrix: CapabilityMatrix
    interaction_logger: InteractionLogger
    graph_store: GraphStore
    graph_retriever: GraphRetriever
    entity_extractor: EntityExtractor
    state_forecaster: StateForecaster
    scenario_simulator: ScenarioSimulator
    self_evaluator: SelfEvaluator
    video_analysis_service: VideoAnalysisService


async def build_services(settings: NexusSettings) -> NexusServices:
    # "local" is always registered so NEXUS boots and works local-only with
    # zero cloud API keys set, exactly as it did in Phase 1. Cloud providers
    # are only added when their key is actually configured.
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

    # Built before the router because the router scores on it, and before
    # RagService because RagService's EntityExtractor needs the router.
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
        # None when learning is disabled, which is exactly the pre-Phase-14
        # behavior: score straight off models.yaml.
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

    # "local" (Ollama) is the default embedding backend so RAG works fully
    # with zero cloud keys configured; "openai" is only used if explicitly
    # selected AND a key is actually present, mirroring the chat-provider
    # local-first fallback above.
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

    # Every tool is individually opt-in via routing.tools.enabled — an
    # unlisted tool is simply never constructed, so the model can neither
    # see nor call it.
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
    # Same backend, same timeout as the tactical adapter -- a separate client
    # only because it reads a different endpoint family
    # (/api/prematch_health/...) with its own response contract.
    prematch_health_client = PreMatchHealthClient(
        settings.sports.backend_base_url,
        timeout_seconds=settings.sports.request_timeout_seconds,
    )
    # Same backend and the same timeout again, reusing sports.backend_base_url
    # rather than introducing a second base-url setting -- a separate client
    # only because it reads the /api/psychology/... endpoint family with its
    # own response contract.
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

    logger.info(
        "NEXUS started (providers=%s, tools=%s)",
        ",".join(sorted(providers.keys())),
        ",".join(sorted(tools.keys())) or "none",
    )
    return NexusServices(
        engine=engine,
        settings=settings,
        provider_manager=provider_manager,
        provider=providers["local"],
        memory=memory,
        cost_tracker=cost_tracker,
        latency_tracker=latency_tracker,
        task_classifier=task_classifier,
        privacy_classifier=privacy_classifier,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        rag_service=rag_service,
        long_term_memory=long_term_memory,
        tool_registry=tool_registry,
        router=router,
        personal_state_engine=personal_state_engine,
        baseline_calculator=baseline_calculator,
        weakness_engine=weakness_engine,
        profile_store=profile_store,
        agent_runtime=agent_runtime,
        health_analyzer=health_analyzer,
        brief_builder=brief_builder,
        sports_adapter=sports_adapter,
        prematch_health_client=prematch_health_client,
        psychology_client=psychology_client,
        personalized_generator=personalized_generator,
        coach_assistant=coach_assistant,
        fact_checker=fact_checker,
        judge=judge,
        verification_engine=verification_engine,
        eval_store=eval_store,
        capability_matrix=capability_matrix,
        interaction_logger=interaction_logger,
        graph_store=graph_store,
        graph_retriever=graph_retriever,
        entity_extractor=entity_extractor,
        state_forecaster=state_forecaster,
        scenario_simulator=scenario_simulator,
        self_evaluator=self_evaluator,
        video_analysis_service=video_analysis_service,
    )


def get_services(request: Request) -> NexusServices:
    """The typed services container for the running app."""
    return request.app.state.services
