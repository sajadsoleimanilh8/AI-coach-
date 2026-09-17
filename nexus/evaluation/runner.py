from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from hashlib import md5
from pathlib import Path
from typing import Any

from nexus.agents.registry import get_agent
from nexus.agents.runtime import AgentRuntime
from nexus.config.settings import get_settings
from nexus.core.embeddings import EmbeddingProvider
from nexus.core.provider_manager import ProviderManager
from nexus.core.providers import AIProvider
from nexus.core.router import ModelRouter
from nexus.core.types import GenerationChunk, GenerationResult, ModelInfo, RoutingPolicy, Usage
from nexus.evaluation.types import CaseOutcome, EvalCase, EvalRun, SkippedSuite, SuiteResult
from nexus.intelligence.privacy_classifier import PrivacyClassifier, PrivacyClassifierConfig
from nexus.intelligence.task_classifier import TaskClassifier, build_signal_map_from_names
from nexus.memory.sqlite_vector_store import SqliteVectorStore
from nexus.memory.storage import create_async_db_engine
from nexus.models.registry import list_models as list_registered_models
from nexus.rag.graph import GraphRetriever, GraphStore
from nexus.rag.service import RagService
from nexus.tools.registry import Tool, ToolRegistry, ToolResult
from nexus.verification.engine import VerificationEngine
from nexus.verification.fact_checker import FactChecker
from nexus.verification.judge import MultiModelJudge

_WORD_RE = re.compile(r"[a-z0-9]+")


def _load_dataset(path: Path) -> list[EvalCase]:
    if not path.exists():
        return []
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.append(
                EvalCase(
                    id=data["id"], suite=data["suite"], input=data["input"],
                    expected=data["expected"], metadata=data.get("metadata", {}),
                )
            )
    return cases


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _config_snapshot(settings: Any) -> dict[str, Any]:
    data = settings.model_dump()
    # Never persist secrets into a reproducibility snapshot that gets
    # written to disk/DB and potentially compared/diffed later.
    data.get("cloud", {}).get("openai", {}).pop("api_key", None)
    data.get("cloud", {}).get("anthropic", {}).pop("api_key", None)
    data.get("tools", {}).get("github", {}).pop("token", None)
    return data


class _HashingBagOfWordsEmbeddingProvider(EmbeddingProvider):
    """Deterministic bag-of-words embedding via a hashing trick — crude,
    but genuinely differentiates semantically-different text (unlike a
    pure length-based fake), which is what the RAG suite actually needs
    to test retrieval rather than just plumbing. No network, no real
    embedding model (principle 6)."""

    _DIM = 128

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    async def health_check(self) -> bool:
        return True

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._DIM
        for word in _WORD_RE.findall(text.lower()):
            index = int(md5(word.encode("utf-8")).hexdigest(), 16) % self._DIM
            vector[index] += 1.0
        return vector


class FakeChatProvider(AIProvider):
    """Scriptable fake — returns queued results in order, or a bland
    default when the queue is empty. Suites that need a specific model
    response (e.g. tools_eval forcing a tool_call) enqueue it directly.

    `queue` lets several providers SHARE one script. The harness wires all
    three fakes to a single queue because a multi-agent case routes
    different sub-agents to different providers (an orchestrator on one,
    the specialist it delegates to on another): with per-provider queues
    the scripted order would silently split in two and the case would read
    a default response instead of its script.
    """

    def __init__(self, name: str, queue: list[GenerationResult] | None = None) -> None:
        self.name = name
        self._queue: list[GenerationResult] = queue if queue is not None else []

    def enqueue(self, result: GenerationResult) -> None:
        self._queue.append(result)

    async def generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        if self._queue:
            return self._queue.pop(0)
        return GenerationResult(
            content="", model_used=model_id, provider_name=self.name, usage=Usage()
        )

    async def stream_generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        yield GenerationChunk(delta="", done=True, usage=Usage())

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> bool:
        return True

    def count_tokens(self, text: str, *, model_id: str) -> int:
        return len(text)


class _FakeTool(Tool):
    """Records whether it was called, without any real side effect
    (subprocess/network/filesystem) — the tools suite tests SELECTION and
    PERMISSION filtering, not real tool execution correctness."""

    def __init__(self, name: str, description: str, parameters_schema: dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.parameters_schema = parameters_schema
        self.executed_with: list[dict[str, Any]] = []

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        self.executed_with.append(arguments)
        return ToolResult(success=True, output="ok")


def _build_fake_tools() -> dict[str, Tool]:
    schema = {"type": "object", "properties": {}}
    names = ("python", "files", "database", "web_search", "github")
    return {name: _FakeTool(name, f"Fake {name} tool.", schema) for name in names}


class Evaluator(ABC):
    suite: str

    # True ONLY if this suite's outcomes actually depend on which chat
    # model serves the request, so that pinning a model_id changes what is
    # measured. It is False by default because most Group D suites are
    # deliberately model-independent — they exercise deterministic
    # machinery (classifiers, retrieval, safety rules, the tool loop) or
    # script their own fake provider, precisely so they are reproducible
    # enough to gate on.
    #
    # A False suite under a pinned run is SKIPPED rather than included:
    # its score would be identical for every model, so reporting it under
    # a custom model's name would credit that model with a result it had
    # no part in.
    supports_model_pinning: bool = False

    @abstractmethod
    async def run_case(self, case: EvalCase, harness: EvalHarness) -> CaseOutcome: ...

    def cases_under_pin(self, cases: list[EvalCase]) -> list[EvalCase]:
        """The subset of `cases` whose outcome genuinely depends on which
        model serves the request. Consulted ONLY on a pinned run.

        Defaults to all of them, which is right for a suite that is
        model-dependent end to end. A suite can be MIXED, though — the
        safety suite checks pure rule functions in most of its cases and
        generates with the model in the rest — and reporting the rule
        cases under a pinned model's name would credit that model with
        scores it had no part in, exactly as including a wholly
        model-independent suite would. Filtering here keeps a mixed suite
        usable under a pin instead of forcing it to skip entirely.
        """
        return cases


class EvalHarness:
    """Owns the components under test and runs suites against them.
    Defaults to FAKE providers (principle 6 — no real external services in
    CI); pass use_real_providers=True to hit configured real ones."""

    def __init__(
        self, *, use_real_providers: bool = False, pinned_model_id: str | None = None
    ) -> None:
        self._use_real_providers = use_real_providers
        # When set, every model-dispatching component the harness builds is
        # routed to this model id (ModelRouter honours requested_model_id
        # directly), and suites that cannot honour the pin are skipped.
        self.pinned_model_id = pinned_model_id
        self._settings = get_settings()
        self._datasets_dir = Path(self._settings.evaluation.datasets_dir)
        self._tmp_dir = tempfile.mkdtemp(prefix="nexus-eval-")

        self.router: ModelRouter | None = None
        self.task_classifier: TaskClassifier | None = None
        self.privacy_classifier: PrivacyClassifier | None = None
        self.rag_service: RagService | None = None
        self.tool_registry: ToolRegistry | None = None
        self.verification_engine: VerificationEngine | None = None
        self.chat_provider: FakeChatProvider | None = None
        # Phase 14 suites need components the Group D suites did not: a
        # graph store to seed triples into, an agent runtime to drive a
        # real orchestrated run, and the engine itself so forecast can
        # stand up a PersonalStateEngine on the same throwaway database.
        self.db_engine = None
        self.graph_store = None
        self.graph_retriever = None
        self.agent_runtime = None

        from nexus.evaluation.suites.classification_eval import ClassificationEvaluator
        from nexus.evaluation.suites.forecast_eval import ForecastEvaluator
        from nexus.evaluation.suites.graph_rag_eval import GraphRagEvaluator
        from nexus.evaluation.suites.orchestration_eval import OrchestrationEvaluator
        from nexus.evaluation.suites.rag_eval import RagEvaluator
        from nexus.evaluation.suites.routing_eval import RoutingEvaluator
        from nexus.evaluation.suites.safety_eval import SafetyEvaluator
        from nexus.evaluation.suites.tools_eval import ToolsEvaluator
        from nexus.evaluation.suites.verification_eval import VerificationEvaluator

        self._evaluators: dict[str, Evaluator] = {
            e.suite: e
            for e in (
                RoutingEvaluator(), ClassificationEvaluator(), RagEvaluator(),
                ToolsEvaluator(), SafetyEvaluator(), VerificationEvaluator(),
                GraphRagEvaluator(), OrchestrationEvaluator(), ForecastEvaluator(),
            )
        }

    async def setup(self) -> None:
        engine = create_async_db_engine(str(Path(self._tmp_dir) / "eval.db"))
        self.db_engine = engine

        if self._use_real_providers:
            providers, embedding_provider = _build_real_providers(self._settings)
        else:
            shared_queue: list[GenerationResult] = []
            self.chat_provider = FakeChatProvider("local", queue=shared_queue)
            providers = {
                "local": self.chat_provider,
                "openai": FakeChatProvider("openai", queue=shared_queue),
                "anthropic": FakeChatProvider("anthropic", queue=shared_queue),
            }
            embedding_provider = _HashingBagOfWordsEmbeddingProvider()

        provider_manager = ProviderManager(providers)
        self.router = ModelRouter(
            provider_manager, list_registered_models(),
            default_policy=RoutingPolicy(self._settings.routing.default_policy),
        )

        self.task_classifier = TaskClassifier(
            build_signal_map_from_names(self._settings.routing.classification.extra_signals),
            min_confidence=self._settings.routing.classification.min_confidence,
            long_context_token_threshold=self._settings.routing.classification.long_context_token_threshold,
        )
        self.privacy_classifier = PrivacyClassifier(
            PrivacyClassifierConfig(
                extra_private_patterns=list(self._settings.routing.privacy.extra_private_patterns)
            )
        )

        vector_store = SqliteVectorStore(engine)
        await vector_store.init()

        # Graph components are constructed but graph_enabled stays False:
        # the graph_rag suite seeds triples directly and calls the retriever
        # itself, so no suite ever triggers LLM-based entity extraction
        # during a fakes-only run.
        self.graph_store = GraphStore(engine)
        await self.graph_store.init()
        self.graph_retriever = GraphRetriever(self.graph_store)

        self.rag_service = RagService(
            engine,
            embedding_provider,
            vector_store,
            graph_store=self.graph_store,
            graph_retriever=self.graph_retriever,
            graph_enabled=False,
        )
        await self.rag_service.init()

        self.tool_registry = ToolRegistry(_build_fake_tools())

        fact_checker = FactChecker(
            self.router,
            max_claims=self._settings.verification.max_claims,
            pinned_model_id=self.pinned_model_id,
        )
        # The judge is deliberately NOT pinned. Its whole purpose is to
        # consult a DIFFERENT model than the one under test; pinning it to
        # the custom model would have that model grade its own answer,
        # which is the single thing MultiModelJudge exists to prevent.
        judge = MultiModelJudge(self.router)
        self.verification_engine = VerificationEngine(
            self.router, fact_checker, judge,
            escalate_below=self._settings.verification.escalate_below,
            # Real fact-checking/judging needs real LLM calls — only turn
            # these on when the harness itself is allowed to use real
            # providers, otherwise the verification suite would silently
            # make network calls a "fakes-only" eval run must never make.
            enable_fact_check=self._use_real_providers,
            enable_judge=self._use_real_providers,
        )

        self.agent_runtime = AgentRuntime(
            self.router,
            self.tool_registry,
            max_iterations=self._settings.agents.max_iterations,
            agent_factory=lambda name: get_agent(name, enabled=self._settings.agents.enabled),
            enabled_agents=list(self._settings.agents.enabled),
            orchestration_max_depth=self._settings.agents.orchestration.max_depth,
            orchestration_max_total_delegations=(
                self._settings.agents.orchestration.max_total_delegations
            ),
        )

    async def run(self, suites: list[str] | None = None) -> EvalRun:
        await self.setup()
        run_id = uuid.uuid4().hex
        started_at = time.time()

        target_suites = suites or list(self._settings.evaluation.default_suites)
        suite_results: list[SuiteResult] = []
        skipped: list[SkippedSuite] = []
        for suite_name in target_suites:
            evaluator = self._evaluators.get(suite_name)
            if evaluator is None:
                continue
            if self.pinned_model_id and not evaluator.supports_model_pinning:
                skipped.append(
                    SkippedSuite(
                        suite=suite_name,
                        reason=(
                            f"cannot honour pinned model_id={self.pinned_model_id!r} — this "
                            f"suite is model-independent (it exercises deterministic "
                            f"machinery or scripts its own provider), so its score would be "
                            f"identical for every model"
                        ),
                    )
                )
                continue
            cases = _load_dataset(self._datasets_dir / f"{suite_name}.jsonl")
            if self.pinned_model_id:
                cases = evaluator.cases_under_pin(cases)
                if not cases:
                    # A suite that declares it supports pinning but has no
                    # model-dependent cases in its dataset measures nothing
                    # about the pinned model. Reporting an empty suite would
                    # score 0.0 and read as a failure; reporting it as a pass
                    # would be worse. It is a skip.
                    skipped.append(
                        SkippedSuite(
                            suite=suite_name,
                            reason=(
                                f"no case in this suite's dataset depends on which model "
                                f"serves the request, so pinning model_id="
                                f"{self.pinned_model_id!r} would measure nothing"
                            ),
                        )
                    )
                    continue
            outcomes = [await evaluator.run_case(case, self) for case in cases]
            suite_results.append(SuiteResult.from_outcomes(suite_name, outcomes))

        finished_at = time.time()
        return EvalRun(
            run_id=run_id, started_at=started_at, finished_at=finished_at,
            suites=suite_results, config_snapshot=_config_snapshot(self._settings),
            git_sha=_git_sha(),
            provider_mode="real" if self._use_real_providers else "fake",
            pinned_model_id=self.pinned_model_id,
            skipped_suites=skipped,
        )


def _build_real_providers(settings: Any) -> tuple[dict[str, AIProvider], EmbeddingProvider]:
    from nexus.models.cloud.anthropic_provider import AnthropicProvider
    from nexus.models.cloud.openai_embedding_provider import OpenAIEmbeddingProvider
    from nexus.models.cloud.openai_provider import OpenAIProvider
    from nexus.models.local.embedding_runtime import OllamaEmbeddingProvider
    from nexus.models.local.runtime import OllamaRuntime

    providers: dict[str, AIProvider] = {
        "local": OllamaRuntime(
            settings.local.base_url, timeout_seconds=settings.local.request_timeout_seconds
        ),
    }
    if settings.cloud.openai.enabled and settings.cloud.openai.api_key:
        providers["openai"] = OpenAIProvider(
            settings.cloud.openai.api_key, timeout_seconds=settings.cloud.openai.timeout_seconds
        )
    if settings.cloud.anthropic.enabled and settings.cloud.anthropic.api_key:
        providers["anthropic"] = AnthropicProvider(
            settings.cloud.anthropic.api_key, timeout_seconds=settings.cloud.anthropic.timeout_seconds
        )

    embedding_provider: EmbeddingProvider
    if settings.rag.embedding_provider == "openai" and settings.cloud.openai.api_key:
        embedding_provider = OpenAIEmbeddingProvider(settings.cloud.openai.api_key)
    else:
        embedding_provider = OllamaEmbeddingProvider(
            settings.local.base_url, model_id=settings.local.models.embeddings,
            timeout_seconds=settings.local.request_timeout_seconds,
        )
    return providers, embedding_provider
