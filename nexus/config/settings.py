from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

CONFIG_DIR = Path(__file__).parent
DEFAULT_CONFIG_PATH = CONFIG_DIR / "nexus.yaml"
DEFAULT_MODELS_PATH = CONFIG_DIR / "models.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class ServerSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8100
    log_level: str = "info"
    log_json: bool = False


class LocalModelSettings(BaseModel):
    general: str = "mistral:7b"
    embeddings: str = "nomic-embed-text"


class LocalSettings(BaseModel):
    backend: str = "ollama"
    base_url: str = "http://localhost:11434"
    request_timeout_seconds: int = 120
    models: LocalModelSettings = LocalModelSettings()


class MemorySettings(BaseModel):
    backend: str = "sqlite"
    database_path: str = "nexus/data/nexus.db"
    session_window_size: int = 20
    session_ttl_minutes: int = 1440


class ClassificationSettings(BaseModel):
    enabled: bool = True
    long_context_token_threshold: int = 6000
    min_confidence: float = 0.3
    extra_signals: dict[str, list[str]] = {}


class PrivacySettings(BaseModel):
    enabled: bool = True
    force_local_on_private: bool = True
    extra_private_patterns: list[str] = []


class LatencySettings(BaseModel):
    window_size: int = 50
    min_samples: int = 5


class RoutingSettings(BaseModel):
    default_policy: str = "BALANCED"
    classification: ClassificationSettings = ClassificationSettings()
    privacy: PrivacySettings = PrivacySettings()
    latency: LatencySettings = LatencySettings()


class OpenAISettings(BaseModel):
    api_key: str | None = None
    enabled: bool = True
    default_model: str = "gpt-4o-mini"
    timeout_seconds: int = 60
    # Any OpenAI-compatible endpoint (vLLM, LM Studio, llama.cpp's server,
    # LiteLLM). OpenAIProvider already accepted this; exposing it in config
    # is what lets a fully local, tool-calling-capable model back the
    # "openai" provider slot — OllamaRuntime deliberately drops tools, so
    # it cannot serve agents that need tool calling.
    base_url: str = "https://api.openai.com/v1"


class AnthropicSettings(BaseModel):
    api_key: str | None = None
    enabled: bool = True
    default_model: str = "claude-sonnet-4-5"
    timeout_seconds: int = 60


class GeminiSettings(BaseModel):
    api_key: str | None = None
    enabled: bool = True
    default_model: str = "gemini-2.5-flash"
    timeout_seconds: int = 60


class CloudSettings(BaseModel):
    openai: OpenAISettings = OpenAISettings()
    anthropic: AnthropicSettings = AnthropicSettings()
    gemini: GeminiSettings = GeminiSettings()


class GraphRagSettings(BaseModel):
    # Off by default: entity extraction is the one genuinely LLM-dependent
    # step in Phase 14, so enabling it changes both cost and ingest latency.
    enabled: bool = False
    max_hops: int = 2
    max_nodes: int = 20


class RagSettings(BaseModel):
    # "local" keeps RAG fully usable with zero cloud keys configured
    # (embeds via Ollama) — the whole point of principle 2 (local-first).
    embedding_provider: Literal["local", "openai"] = "local"
    chunk_size: int = 800
    chunk_overlap: int = 150
    default_top_k: int = 5
    graph: GraphRagSettings = GraphRagSettings()


class GithubToolSettings(BaseModel):
    token: str | None = None


class ToolsSettings(BaseModel):
    enabled: list[str] = ["web_search", "files", "database"]
    max_iterations: int = 3
    python_timeout_seconds: int = 10
    files_allowed_root: str = "nexus/data/workspace"
    web_search_max_results: int = 5
    github: GithubToolSettings = GithubToolSettings()


class OrchestrationSettings(BaseModel):
    # Enforced by DelegationGuard inside AgentRuntime, not requested in a
    # prompt — a cap a model can talk its way past is not a cap.
    max_depth: int = 2
    max_total_delegations: int = 6


class ResearchLoopSettings(BaseModel):
    max_rounds: int = 3


class AgentsSettings(BaseModel):
    enabled: list[str] = [
        "research",
        "coding",
        "data",
        "planning",
        "health",
        "sports",
        "orchestrator",
        "autonomous_research",
    ]
    max_iterations: int = 8
    default_policy: str = "BALANCED"
    orchestration: OrchestrationSettings = OrchestrationSettings()
    research: ResearchLoopSettings = ResearchLoopSettings()


class ForecastSettings(BaseModel):
    max_horizon_days: int = 30
    min_trend_confidence: float = 0.5


class PersonalSettings(BaseModel):
    enabled: bool = True
    forecast: ForecastSettings = ForecastSettings()
    # Withhold personal context from cloud providers by default (principle 7)
    # — see chat.py's use_personal_context handling for where this gate is
    # enforced (must run AFTER routing resolves a provider).
    local_only_context: bool = True
    state_half_life_days: float = 7.0
    state_recent_window_days: float = 14.0
    baseline_window_days: float = 90.0
    baseline_min_samples: int = 5
    weakness_min_confidence: float = 0.5
    weakness_min_deviation: float = 0.08
    trend_min_samples: int = 4


class HealthSettings(BaseModel):
    enabled: bool = True
    min_sample_size: int = 3
    # NOTE: deliberately no safety-layer toggle here — nexus/health/safety.py
    # runs unconditionally on both the health endpoint and HealthAgent.


class GenerationSettings(BaseModel):
    enabled: bool = True
    default_intensity: float = 0.7
    min_intensity: float = 0.2
    max_intensity: float = 1.0


class SportsSettings(BaseModel):
    enabled: bool = True
    backend_base_url: str = "http://localhost:8000"
    request_timeout_seconds: float = 30.0
    # football metric_name -> NEXUS sports.* dimension — config-driven since
    # the pipeline's metric set is expected to grow independently of NEXUS.
    metric_dimension_map: dict[str, str] = {
        "decision_making_score": "sports.decision_making",
        "off_ball_movement_score": "sports.positioning",
        "press_resistance_score": "sports.agility",
        "first_touch_score": "sports.agility",
        "defensive_positioning_score": "sports.positioning",
    }


class VerificationSettings(BaseModel):
    enabled: bool = True
    # Off per-request by default (verification costs extra tokens) — these
    # task types always verify regardless of the request's `verify` flag.
    always_verify_task_types: list[str] = ["health", "research", "mathematics"]
    enable_fact_check: bool = True
    enable_judge: bool = True
    escalate_below: float = 0.65
    max_claims: int = 6


class TrainingSettings(BaseModel):
    # Off by default: interaction logging stores full prompts and responses,
    # which is a consent decision, not a convenience default.
    log_interactions: bool = False
    interaction_retention_days: int = 180
    dataset_output_dir: str = "nexus/training/data"
    exclude_privacy_levels: list[str] = ["private"]
    min_verification_score: float = 0.8
    available_vram_gb: float = 12.0  # RTX 5070 Ti Laptop


class CapabilityLearningSettings(BaseModel):
    enabled: bool = True
    min_samples: int = 20
    max_learned_weight: float = 0.7


class SelfEvalSettings(BaseModel):
    # Off by default: a self-critique is a second generation per answer.
    enabled: bool = False
    log_low_scores: bool = True


class EvaluationSettings(BaseModel):
    datasets_dir: str = "nexus/evaluation/datasets"
    default_suites: list[str] = [
        "routing",
        "classification",
        "rag",
        "tools",
        "safety",
        "verification",
        "graph_rag",
        "orchestration",
        "forecast",
    ]
    use_real_providers: bool = False
    pass_rate_tolerance: float = 0.02
    cost_tolerance: float = 0.25
    latency_tolerance: float = 0.5


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """Seeds defaults from nexus.yaml, below init args and env vars in priority."""

    def __init__(self, settings_cls: type[BaseSettings], yaml_path: Path) -> None:
        self._data = _load_yaml(yaml_path)
        super().__init__(settings_cls)

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return dict(self._data)


class NexusSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NEXUS_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    server: ServerSettings = ServerSettings()
    local: LocalSettings = LocalSettings()
    memory: MemorySettings = MemorySettings()
    routing: RoutingSettings = RoutingSettings()
    cloud: CloudSettings = CloudSettings()
    rag: RagSettings = RagSettings()
    tools: ToolsSettings = ToolsSettings()
    agents: AgentsSettings = AgentsSettings()
    personal: PersonalSettings = PersonalSettings()
    health: HealthSettings = HealthSettings()
    generation: GenerationSettings = GenerationSettings()
    sports: SportsSettings = SportsSettings()
    verification: VerificationSettings = VerificationSettings()
    evaluation: EvaluationSettings = EvaluationSettings()
    training: TrainingSettings = TrainingSettings()
    capability_learning: CapabilityLearningSettings = CapabilityLearningSettings()
    self_eval: SelfEvalSettings = SelfEvalSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls, DEFAULT_CONFIG_PATH),
            file_secret_settings,
        )

    @model_validator(mode="after")
        # Documented in Phase 1 spec as the standalone env var docker-compose uses
        # to wire NEXUS to the `ollama` service, independent of the NEXUS_ prefix.
    def _apply_ollama_base_url_override(self) -> NexusSettings:
        override = os.environ.get("OLLAMA_BASE_URL")
        if override:
            self.local.base_url = override
        return self

    @model_validator(mode="after")
        # Standalone env vars (not NEXUS_-prefixed) so keys can be shared with
        # other tools that already read OPENAI_API_KEY / ANTHROPIC_API_KEY, and
        # so they're never accidentally checked into nexus.yaml.
    def _apply_cloud_api_key_env_overrides(self) -> NexusSettings:
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            self.cloud.openai.api_key = openai_key
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            self.cloud.anthropic.api_key = anthropic_key
        # Google's own tooling is split between these two names, so accepting
        # only one guarantees someone loses an hour to a key that is set but
        # ignored. GEMINI_API_KEY wins when both are present.
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if gemini_key:
            self.cloud.gemini.api_key = gemini_key
        return self

    @model_validator(mode="after")
        # Same standalone-env-var pattern as OPENAI_API_KEY/ANTHROPIC_API_KEY.
    def _apply_python_tool_env_override(self) -> NexusSettings:
        """PythonExecutionTool executes model-supplied code with no sandbox, so it
        is off unless an operator explicitly opts in. This is additive on purpose:
        it can only ever turn the tool ON, never silently off."""
        flag = os.environ.get("NEXUS_ENABLE_PYTHON_TOOL", "").strip().lower()
        if flag in {"1", "true", "yes", "on"} and "python" not in self.tools.enabled:
            self.tools.enabled = [*self.tools.enabled, "python"]
        return self

    @model_validator(mode="after")
    def _apply_github_token_env_override(self) -> NexusSettings:
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            self.tools.github.token = token
        return self


_settings: NexusSettings | None = None


def get_settings(*, refresh: bool = False) -> NexusSettings:
    global _settings
    if _settings is None or refresh:
        _settings = NexusSettings()
    return _settings


def load_model_registry(path: Path = DEFAULT_MODELS_PATH) -> list[dict[str, Any]]:
    data = _load_yaml(path)
    return data.get("models", [])
