from datetime import datetime, time
from typing import Any, Literal
from pydantic import BaseModel, Field

from ai.performance_ai.match_readiness_predictor.constants import (
    SCALE_MAX,
    SCALE_MIN,
    SLEEP_HOURS_MAX,
    SLEEP_HOURS_MIN_EXCLUSIVE,
)

from ai.psychology_ai.constants import (
    SCALE_MAX as PSYCH_SCALE_MAX,
    SCALE_MIN as PSYCH_SCALE_MIN,
)

class VideoUploadResponse(BaseModel):
    video_id: str
    job_id: str
    match_id: str
    filename: str
    status: str
    message: str

class ProcessingStatusResponse(BaseModel):
    job_id: str
    video_id: str
    match_id: str | None = None
    status: str
    progress: int
    message: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None

class JobStatusUpdate(BaseModel):
    status: Literal["queued", "processing", "completed", "failed"]
    progress: int = Field(default=0, ge=0, le=100)
    message: str | None = None
    error: str | None = None

class AnalysisResultCreate(BaseModel):
    result: dict[str, Any]
    summary: str | None = None

class AnalysisResultResponse(BaseModel):
    job_id: str
    video_id: str
    status: str
    summary: str | None = None
    result: dict[str, Any] | None = None

class PlayerMetricResponse(BaseModel):
    metric_id: str
    match_id: str
    player_id: int
    metric_name: str
    value: float | None = None
    method: str
    confidence: str
    sample_size: int
    sub_scores: dict[str, Any]
    computed_at: datetime
    schema_version: str

class TeamMetricResponse(BaseModel):
    metric_id: str
    match_id: str
    team_id: str
    metric_name: str
    value: Any = None
    method: str
    confidence: str
    confidence_score: float | None = None
    sample_size: int
    sub_scores: dict[str, Any]
    computed_at: datetime
    schema_version: str

class PlayerIntelligenceResponse(BaseModel):
    player_id: int
    player_name: str | None = None
    metrics: list[PlayerMetricResponse]

class TrackedPlayerPoint(BaseModel):
    player_id: int
    team_id: str | None = None
    pixel_x: float
    pixel_y: float
    pitch_x_m: float | None = None
    pitch_y_m: float | None = None

class TrackedBallPoint(BaseModel):
    pixel_x: float
    pixel_y: float

class TrackingFrame(BaseModel):
    frame_number: int
    timestamp: float
    players: list[TrackedPlayerPoint]
    ball: TrackedBallPoint | None = None

class TrackingCalibrationSummary(BaseModel):
    """Whether the pitch coordinates in this window can be trusted."""
    has_status_rows: bool
    valid_frame_ranges: list[tuple[int, int]]
    valid_in_window: bool
    tracking_rows_in_window: int
    rows_with_pitch_coordinates: int
    note: str

class TrackingWindowResponse(BaseModel):
    match_id: str
    start_frame: int
    end_frame: int
    fps: float
    frames: list[TrackingFrame]
    calibration: TrackingCalibrationSummary | None = None

    frame_width_px: int | None = None
    frame_height_px: int | None = None

    max_frame: int | None = None


class EventItem(BaseModel):
    """One detected match event."""
    event_id: str
    event_type: str
    timestamp: float
    player_id: int | None = None
    related_player_id: int | None = None
    team_id: str | None = None
    pitch_x_m: float | None = None
    pitch_y_m: float | None = None
    homography_confidence: float | None = None
    space: Literal["pitch", "image"] = "pitch"
    metadata: dict | None = None


class EventsResponse(BaseModel):
    """GET /api/matches/{match_id}/events."""
    match_id: str
    total: int
    returned: int
    by_type: dict[str, int]
    events: list[EventItem]

    space: Literal["pitch", "image", "none"] = "none"

class MatchSummaryResponse(BaseModel):
    """What a caller needs to work with a match that ALREADY exists."""
    match_id: str
    home_team: str | None = None
    away_team: str | None = None
    duration: float | None = None
    video_id: str | None = None
    video_filename: str | None = None
    video_file_exists: bool = False
    processed_video_exists: bool = False
    processed_video_error: str | None = None
    processed_video_codec: str | None = None
    frame_width_px: int | None = None
    frame_height_px: int | None = None
    job_id: str | None = None
    job_status: str | None = None

class MatchListItem(BaseModel):
    """One row of GET /api/matches -- enough to CHOOSE a match, no more."""
    match_id: str
    home_team: str | None = None
    away_team: str | None = None
    created_at: datetime | None = None
    video_filename: str | None = None
    video_file_exists: bool = False
    job_id: str | None = None
    job_status: str | None = None
    tracking_rows: int = 0

class HeatmapCell(BaseModel):
    grid_x: int
    grid_y: int
    count: int
    density: float

class HeatmapResponse(BaseModel):
    match_id: str
    player_id: int
    grid_cols: int
    grid_rows: int
    pitch_length_m: float
    pitch_width_m: float
    cells: list[HeatmapCell]
    confidence: str
    sample_size: int
    usable_sample_size: int

    space: Literal["pitch", "image"] = "pitch"

    frame_width_px: int | None = None
    frame_height_px: int | None = None


class SimulationInterventionRequest(BaseModel):
    kind: Literal["compactness", "transition_speed", "remove_player", "swap_player"]
    team_id: str | None = None
    player_id: int | None = None
    other_player_id: int | None = None
    pct: float | None = None


class SimulationRequest(BaseModel):
    interventions: list[SimulationInterventionRequest] = Field(min_length=1, max_length=8)


class SimulationResponse(BaseModel):
    match_id: str
    interventions: list[str]
    metrics: list[dict[str, Any]]
    caveats: list[str]
    unavailable: list[str]
    is_reinforcement_learning: Literal[False]
    method: Literal["heuristic_proxy"]
    real_input_metrics: list[dict[str, Any]]
    team_assignment_confidence: float | None = None


class CalibrationDebugResponse(BaseModel):
    match_id: str
    frame_number: int
    overlay_png_base64: str
    calibration_valid: bool
    calibration_confidence: float
    calibration_metric_method: Literal["deterministic"]
    calibration_metric_confidence: Literal["normal", "low_upstream_confidence"]
    invalid_reason: str | None = None
    keypoint_detection_method: Literal["ml_trained"]
    keypoint_detection_confidence: float | None = None
    n_keypoints: int
    field_detected: bool
    field_detection_confidence: float | None = None
    projected_players: list[dict[str, Any]]
    projection_suppressed: bool

class PipelineStageTiming(BaseModel):
    """One row of a real, measured pipeline run -- see backend/pipeline/latency.py.
    Never hand-typed: every instance of this model in the system is produced
    by an actual PipelineTimer.stage() context manager wrapping real work."""
    stage: str
    seconds: float
    detail: str | None = None

class PipelineLatencyReport(BaseModel):
    job_id: str
    match_id: str | None = None
    total_seconds: float
    stages: list[PipelineStageTiming]
    generated_at: datetime
    source: Literal["measured"] = "measured"


class PreMatchQuestionnaireRequest(BaseModel):
    """One player's pre-match self-report."""

    match_id: str | None = None

    sleep_duration_hours: float = Field(gt=SLEEP_HOURS_MIN_EXCLUSIVE, le=SLEEP_HOURS_MAX)
    sleep_quality: int = Field(ge=SCALE_MIN, le=SCALE_MAX)
    bedtime: time | None = None
    wake_time: time | None = None
    night_awakenings: int = Field(default=0, ge=0)

    trained_last_24h: bool = False
    trained_last_48h: bool = False
    training_duration_minutes: float = Field(default=0.0, ge=0)
    training_intensity: int = Field(default=SCALE_MIN, ge=SCALE_MIN, le=SCALE_MAX)
    high_intensity_activity: bool = False
    hours_since_last_training: float | None = Field(default=None, ge=0)

    fatigue: int = Field(ge=SCALE_MIN, le=SCALE_MAX)
    muscle_soreness: int = Field(ge=SCALE_MIN, le=SCALE_MAX)
    pain_level: int = Field(ge=SCALE_MIN, le=SCALE_MAX)
    perceived_readiness: int = Field(ge=SCALE_MIN, le=SCALE_MAX)

    hydration_liters: float = Field(default=0.0, ge=0)
    nutrition_quality: int = Field(ge=SCALE_MIN, le=SCALE_MAX)
    hours_since_last_meal: float = Field(default=0.0, ge=0)

    caffeine_or_supplement_notes: str | None = None

class PreMatchFactorResponse(BaseModel):
    """One input dimension, classified -- the explainability record behind
    the headline numbers."""
    dimension: str
    label: Literal["positive", "negative", "neutral"]
    score: float
    detail: str

class PreMatchAssessmentResponse(BaseModel):
    """The stable contract consumed by the frontend tab and by the nexus
    LLM Coach integration (nexus/sports/prematch_health.py).
    """
    player_id: str
    match_id: str | None = None
    physical_readiness: float
    fatigue_score: float
    recovery_score: float
    performance_risk: Literal["low", "moderate", "high"]
    workload_risk: Literal["low", "moderate", "high"]
    key_positive_factors: list[str]
    key_negative_factors: list[str]
    method: str
    schema_version: str
    computed_at: datetime

    assessment_id: str
    questionnaire_id: str
    submission_index: int
    factors: list[PreMatchFactorResponse]
    data_source: str
    notes: str | None = None
    disclaimer: str

class PreMatchFeaturesResponse(BaseModel):
    """The normalized feature vector on its own, for future ML/DL consumers."""
    assessment_id: str
    player_id: str
    match_id: str | None = None
    features: dict[str, Any]
    feature_directions: dict[str, str]
    method: str
    schema_version: str
    computed_at: datetime


class PsychologyQuestionnaireRequest(BaseModel):
    """One player's pre-match mental-readiness self-report: 13 items, ~1-2
    minutes.
    """

    match_id: str | None = None

    cv_player_id: int | None = None

    concentration_level: int = Field(ge=PSYCH_SCALE_MIN, le=PSYCH_SCALE_MAX)
    focus_maintenance: int = Field(ge=PSYCH_SCALE_MIN, le=PSYCH_SCALE_MAX)
    mental_clarity_raw: int = Field(ge=PSYCH_SCALE_MIN, le=PSYCH_SCALE_MAX)

    pre_match_stress: int = Field(ge=PSYCH_SCALE_MIN, le=PSYCH_SCALE_MAX)
    importance_pressure: int = Field(ge=PSYCH_SCALE_MIN, le=PSYCH_SCALE_MAX)
    nervousness: int = Field(ge=PSYCH_SCALE_MIN, le=PSYCH_SCALE_MAX)

    performance_confidence: int = Field(ge=PSYCH_SCALE_MIN, le=PSYCH_SCALE_MAX)
    tactical_confidence_raw: int = Field(ge=PSYCH_SCALE_MIN, le=PSYCH_SCALE_MAX)

    match_motivation: int = Field(ge=PSYCH_SCALE_MIN, le=PSYCH_SCALE_MAX)
    competitive_motivation_raw: int = Field(ge=PSYCH_SCALE_MIN, le=PSYCH_SCALE_MAX)

    mistake_recovery_speed: int = Field(ge=PSYCH_SCALE_MIN, le=PSYCH_SCALE_MAX)
    pressure_performance_effect: Literal["improves", "no_change", "reduces"]
    post_error_calm: int = Field(ge=PSYCH_SCALE_MIN, le=PSYCH_SCALE_MAX)

class PsychologyAssessmentResponse(BaseModel):
    """The stable contract consumed by the frontend tab and by the nexus LLM
    Coach integration (nexus/sports/psychology.py).
    """
    player_id: str
    match_id: str | None = None
    cv_player_id: int | None = None

    mental_readiness: int
    focus: int
    confidence: int
    stress: int

    pressure_risk: Literal["low", "moderate", "high"]
    mental_performance_risk: Literal["low", "moderate", "high"]

    factors: dict[str, str]
    key_positive_factors: list[str]
    key_negative_factors: list[str]

    method: str
    confidence_level: str
    sample_size: int
    schema_version: str
    submitted_at: datetime
    computed_at: datetime

    assessment_id: str
    submission_index: int
    data_source: str
    historical_context: list[str] = []
    disclaimer: str

class PsychologyFeaturesResponse(BaseModel):
    """The normalized feature vector on its own, for future ML/DL consumers."""
    assessment_id: str
    player_id: str
    match_id: str | None = None
    features: dict[str, Any]
    feature_directions: dict[str, str]
    method: str
    schema_version: str
    computed_at: datetime
