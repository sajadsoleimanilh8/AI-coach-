from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from nexus.core.types import RoutingPolicy


class MessageSchema(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatRequest(BaseModel):
    messages: list[MessageSchema]
    model_id: str | None = None
    session_id: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = None
    stream: bool = False
    policy: RoutingPolicy | None = None
    user_id: str = "default"
    use_rag: bool = False
    rag_top_k: int = 5
    use_tools: bool = False
    use_personal_context: bool = False
    verify: bool = False


class UsageSchema(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CitationSchema(BaseModel):
    doc_id: str
    source_name: str
    chunk_text: str
    score: float


class ToolCallSummary(BaseModel):
    name: str
    arguments: dict[str, Any]
    result_summary: str


class CheckResultSchema(BaseModel):
    name: str
    status: Literal["pass", "fail", "inconclusive"]
    weight: float
    detail: str
    evidence: list[str] = []


class VerificationReportSchema(BaseModel):
    checks: list[CheckResultSchema]
    score: float | None
    band: Literal["high", "medium", "low", "uncertain", "unverified"]
    summary: str
    uncertainty_notes: list[str] = []
    escalated: bool = False
    extra_usage: UsageSchema


class ChatResponse(BaseModel):
    content: str
    model_used: str
    usage: UsageSchema
    session_id: str
    provider_name: str | None = None
    routing_reason: str | None = None
    cost_usd: float | None = None
    task_type: str | None = None
    privacy_level: str | None = None
    classification_reason: str | None = None
    citations: list[CitationSchema] | None = None
    tool_calls_made: list[ToolCallSummary] | None = None
    personal_context_used: bool | None = None
    verification: VerificationReportSchema | None = None
    interaction_id: int | None = None


class FeedbackRequest(BaseModel):
    interaction_id: int
    feedback: Literal[-1, 0, 1]


class FeedbackResponse(BaseModel):
    interaction_id: int
    feedback: int
    recorded: bool


class ModelStatus(BaseModel):
    id: str
    available: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    providers: dict[str, bool]
    local_models: list[str]


class DocumentIngestRequest(BaseModel):
    user_id: str = "default"
    source_name: str
    source_type: Literal["pdf", "docx", "txt", "md", "csv", "code"]
    content_base64: str


class DocumentSummarySchema(BaseModel):
    doc_id: str
    source_name: str
    source_type: str
    chunk_count: int
    ingested_at: float


class MemoryFactsResponse(BaseModel):
    user_id: str
    facts: dict[str, Any]


class MemorySetFactRequest(BaseModel):
    key: str
    value: Any


class AgentInfoSchema(BaseModel):
    name: str
    description: str
    allowed_tools: list[str]


class AgentRunRequest(BaseModel):
    goal: str
    user_id: str = "default"
    session_id: str | None = None
    max_iterations: int | None = None
    match_id: str | None = None
    player_id: int | None = None


class AgentStepSchema(BaseModel):
    index: int
    thought: str
    tool_name: str | None
    tool_arguments: dict[str, Any] | None
    tool_output: str | None
    timestamp: float


class DelegationStepSchema(BaseModel):
    agent_name: str
    sub_goal: str
    result_summary: str
    depth: int


class AgentRunResponse(BaseModel):
    goal: str
    final_answer: str
    steps: list[AgentStepSchema]
    completed: bool
    iterations_used: int
    usage: UsageSchema
    cost_usd: float | None = None
    model_used: str | None = None
    provider_name: str | None = None
    delegation_steps: list[DelegationStepSchema] = []
    rounds_used: int = 1


class PersonalSignalRequest(BaseModel):
    dimension: str
    value: float = Field(ge=0.0, le=1.0)
    source: str
    note: str = ""


class DimensionStateSchema(BaseModel):
    dimension: str
    value: float
    confidence: float
    sample_count: int
    latest_at: float | None


class PersonalStateResponse(BaseModel):
    user_id: str
    dimensions: dict[str, DimensionStateSchema]
    computed_at: float


class BaselineSchema(BaseModel):
    dimension: str
    value: float
    sample_count: int
    window_days: float


class PersonalBaselinesResponse(BaseModel):
    user_id: str
    baselines: dict[str, BaselineSchema]


class TrendSchema(BaseModel):
    dimension: str
    direction: Literal["improving", "declining", "stable"]
    slope: float
    confidence: float
    sample_count: int


class WeaknessSchema(BaseModel):
    dimension: str
    current: float
    baseline: float
    deviation: float
    priority: Literal["HIGH", "MEDIUM", "LOW"]
    confidence: float
    trend: TrendSchema | None
    explanation: str


class PersonalWeaknessesResponse(BaseModel):
    user_id: str
    weaknesses: list[WeaknessSchema]


class PersonalProfileResponse(BaseModel):
    user_id: str
    profile: dict[str, Any]


class PersonalProfileSetRequest(BaseModel):
    profile: dict[str, Any]


class DimensionForecastSchema(BaseModel):
    dimension: str
    current_value: float
    projected_value: float | None
    horizon_days: int
    confidence: float
    method: Literal["trend_extrapolation", "insufficient_data"]
    caveat: str
    trend: TrendSchema | None = None


class PersonalForecastResponse(BaseModel):
    user_id: str
    horizon_days: int
    forecasts: list[DimensionForecastSchema]
    caveat: str


class ScenarioSimulateRequest(BaseModel):
    deltas: dict[str, float]


class ClearedWeaknessSchema(BaseModel):
    dimension: str
    deviation_before: float
    deviation_after: float
    cleared: bool


class ScenarioSimulateResponse(BaseModel):
    user_id: str
    deltas: dict[str, float]
    weaknesses_before: list[str]
    weaknesses_after: list[str]
    changes: list[ClearedWeaknessSchema]
    caveat: str


class VideoAnalyzeRequest(BaseModel):
    video_path_or_url: str
    match_id: str
    player_id: int | None = None


class HealthPatternSchema(BaseModel):
    name: str
    dimensions_involved: list[str]
    severity: Literal["watch", "notable", "significant"]
    confidence: float
    sample_size: int
    explanation: str


class HealthAnalysisResponse(BaseModel):
    user_id: str
    patterns: list[HealthPatternSchema]
    scorecard: dict[str, float]
    data_sufficiency: Literal["none", "sparse", "adequate"]
    computed_at: float
    disclaimer: str


class HealthScorecardResponse(BaseModel):
    user_id: str
    scorecard: dict[str, float]
    data_sufficiency: Literal["none", "sparse", "adequate"]
    disclaimer: str


class GenerationRequest(BaseModel):
    user_id: str = "default"
    constraints: dict[str, Any] = {}


class PersonalizedPlanResponse(BaseModel):
    request_type: str
    content: str
    brief_rationale: list[str]
    addressed_weaknesses: list[str]
    target_intensity: float
    personalized: bool
    model_used: str
    usage: UsageSchema


class SportsMetricSchema(BaseModel):
    metric_name: str
    value: float | str | None
    method: str
    confidence: str
    sample_size: int
    sub_scores: dict[str, Any]
    player_id: int | None = None


class TacticalFindingSchema(BaseModel):
    area: str
    assessment: Literal["strength", "neutral", "weakness"]
    supporting_metrics: list[str]
    confidence: str
    explanation: str


class OpponentWeaknessSchema(BaseModel):
    """One measured weakness. `supporting_metrics` is what makes the claim
    auditable — every number in `evidence` came from those metrics."""

    key: str
    label: str
    severity: Literal["minor", "moderate", "major"]
    value: float
    supporting_metrics: list[str]
    confidence: str
    sample_size: int
    evidence: str
    zone: str | None = None


class OpponentStrengthSchema(BaseModel):
    key: str
    label: str
    value: float
    supporting_metrics: list[str]
    confidence: str
    sample_size: int
    evidence: str


class PrincipleSchema(BaseModel):
    statement: str
    grounded_in: str
    supporting_metrics: list[str]


class AdjustmentSchema(BaseModel):
    """An in-game change. `targets_weakness` names the observed weakness it
    answers, so no adjustment can be surfaced without a measured reason."""

    instruction: str
    targets_weakness: str
    rationale: str
    supporting_metrics: list[str]
    priority: int


class ProposedShapeSchema(BaseModel):
    shape: str
    reason: str
    supporting_metrics: list[str]


class GamePlanSchema(BaseModel):
    """Derived deterministically from measured metrics — no LLM involved.
    The narrative on CoachReportResponse explains this; it does not
    produce it."""

    opponent_strengths: list[OpponentStrengthSchema]
    opponent_weaknesses: list[OpponentWeaknessSchema]
    proposed_shape: ProposedShapeSchema | None = None
    principles: list[PrincipleSchema]
    adjustments: list[AdjustmentSchema]
    uncovered_weaknesses: list[str] = []
    unmeasured_metrics: list[str] = []
    is_partial: bool = False
    partial_reason: str = ""


class TimelineSectionsSchema(BaseModel):
    """Which Phase 4 timeline sections backed this report."""

    available: list[str] = []
    missing: list[str] = []


class CoachReportResponse(BaseModel):
    match_id: str
    findings: list[TacticalFindingSchema]
    unavailable_metrics: list[str]
    coverage: float
    narrative: str
    model_used: str
    game_plan: GamePlanSchema | None = None
    timeline_sections: TimelineSectionsSchema | None = None
    is_partial: bool = False
    # Claims the narrative made that the measured data could not support,
    # surfaced rather than hidden when a corrective retry did not clear them.
    narrative_warnings: list[str] = []


class PreMatchAssessmentSchema(BaseModel):
    """The football backend's computed pre-match assessment, passed through
    unchanged. NEXUS never recomputes any of these values."""
    player_id: str
    match_id: str | None = None
    physical_readiness: float
    fatigue_score: float
    recovery_score: float
    performance_risk: str
    workload_risk: str
    key_positive_factors: list[str]
    key_negative_factors: list[str]
    method: str
    schema_version: str
    computed_at: str
    data_source: str
    notes: str | None = None
    disclaimer: str = ""


class PreMatchCoachReportResponse(BaseModel):
    """`narrative` is the only LLM-generated field; `assessment` is the
    deterministic input it was given, returned alongside so the two can be
    compared."""
    player_id: str
    match_id: str | None = None
    assessment: PreMatchAssessmentSchema
    narrative: str
    model_used: str


class PsychologyCoachReportResponse(BaseModel):
    """`narrative` is the only LLM-generated field."""
    player_id: str
    match_id: str | None = None
    mental_readiness: int
    focus: int
    confidence: int
    stress: int
    pressure_risk: str
    mental_performance_risk: str
    key_positive_factors: list[str]
    key_negative_factors: list[str]
    narrative: str
    neutral_factors: list[str] = []
    method: str = ""
    confidence_level: str = ""
    schema_version: str = ""
    data_source: str = ""
    historical_context: list[str] = []
    disclaimer: str = ""
    model_used: str = ""


class SportsIngestRequest(BaseModel):
    user_id: str = "default"
    player_id: int


class SportsIngestResponse(BaseModel):
    match_id: str
    user_id: str
    signals_recorded: int


class EvalCaseOutcomeSchema(BaseModel):
    case_id: str
    passed: bool
    score: float
    actual: dict[str, Any]
    detail: str
    latency_seconds: float
    cost_usd: float


class EvalSuiteResultSchema(BaseModel):
    suite: str
    outcomes: list[EvalCaseOutcomeSchema]
    pass_rate: float
    mean_score: float
    total_cost_usd: float
    total_latency_seconds: float


class EvalRunSchema(BaseModel):
    run_id: str
    started_at: float
    finished_at: float
    suites: list[EvalSuiteResultSchema]
    config_snapshot: dict[str, Any]
    git_sha: str | None


class EvalRunRequest(BaseModel):
    suites: list[str] | None = None
    use_real_providers: bool = False


class EvalRegressionFindingSchema(BaseModel):
    suite: str
    case_id: str | None
    kind: Literal["suite_pass_rate_drop", "case_regressed", "cost_spike", "latency_spike"]
    before: float
    after: float
    detail: str


class EvalCompareResponse(BaseModel):
    baseline_run_id: str
    candidate_run_id: str
    findings: list[EvalRegressionFindingSchema]
