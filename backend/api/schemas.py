from datetime import datetime, time
from typing import Any, Literal

from pydantic import BaseModel, Field

# The questionnaire's valid ranges are defined once, in the scoring engine
# (ai/performance_ai/match_readiness_predictor/constants.py), and imported
# here so the Field(ge=/le=) constraints that produce the API's 422 cannot
# drift from the bounds the engine itself enforces. Importing ai/ from
# backend/ is the established direction of dependency in this repo -- see
# backend/pipeline/runner.py.
from ai.performance_ai.match_readiness_predictor.constants import (
    SCALE_MAX,
    SCALE_MIN,
    SLEEP_HOURS_MAX,
    SLEEP_HOURS_MIN_EXCLUSIVE,
)

# Same rule for the psychology questionnaire, from its own engine. Aliased
# because both engines define a 1-10 scale under the same constant names: they
# happen to agree today, but they are separate questionnaires and neither
# should silently inherit the other's bounds if one ever changes.
from ai.psychology_ai.constants import (
    SCALE_MAX as PSYCH_SCALE_MAX,
)
from ai.psychology_ai.constants import (
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
    """Whether the pitch coordinates in this window can be trusted.

    Added Phase 3. `TrackingFrame.players[].pitch_x_m` is None wherever
    calibration was invalid and is never imputed, but the payload used to
    carry no way to tell that apart from a genuinely empty window -- and on
    current broadcast footage every frame is in that state
    (calibration_valid_fraction = 0.0, docs/pipeline_architecture.md 6.3).
    """
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

    # The SOURCE frame geometry every pixel_x/pixel_y in this payload is
    # expressed in.
    #
    # Load-bearing for the video overlay, not informational. Tracking pixels
    # are in the ORIGINAL clip's coordinate space, while the annotated render
    # the dashboard now plays by default is downscaled (OVERLAY_MAX_WIDTH,
    # 960 by default). An overlay that sized its coordinate system from the
    # <video> element's own intrinsic width therefore drew 1280-space
    # coordinates into a 960-wide box and pushed every marker off to the
    # right. Sending the source extent lets the overlay establish one
    # coordinate system that is correct over EITHER source.
    frame_width_px: int | None = None
    frame_height_px: int | None = None

    # Highest frame number this match has tracking rows for, so a player can
    # page forward through windows on its own instead of making the user
    # type a start frame. None when the match has no tracking rows at all.
    max_frame: int | None = None


class EventItem(BaseModel):
    """One detected match event.

    `space` is the honest part. Possession -- which every pass, turnover and
    first-touch event is derived from -- is normally measured in pitch
    metres. When calibration does not validate there are no pitch metres, and
    this pipeline then falls back to measuring possession in IMAGE space
    using each player's own bounding-box height as the pixels-per-metre
    scale at their position in frame (see
    ai/computer_vision/tactical_analysis/possession.py). That fallback
    produces real events from real measurements, but it is a weaker
    instrument than a calibrated one, and pitch_x_m/pitch_y_m stay None
    rather than being back-filled with a plausible-looking number.

    So: space="pitch" means the coordinates below are metres and are
    populated. space="image" means possession was resolved in pixels, the
    metre fields are null, and no distance in this event is in metres.
    """
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
    """GET /api/matches/{match_id}/events.

    Counted as well as listed: `total` is the number of events for the match
    and `returned` the number in this payload, so a truncated list is
    distinguishable from a short one. `by_type` lets the UI show what was
    found without walking the list.
    """
    match_id: str
    total: int
    returned: int
    by_type: dict[str, int]
    events: list[EventItem]

    # How possession was resolved for this match, once, rather than per
    # event: "pitch" when calibration validated, "image" when it did not,
    # "none" when there were no events at all. The UI states this next to
    # the timeline instead of leaving the reading unqualified.
    space: Literal["pitch", "image", "none"] = "none"

class MatchSummaryResponse(BaseModel):
    """What a caller needs to work with a match that ALREADY exists.

    Lets the dashboard resolve a match's video_id/job_id after the fact. Only
    the upload response carries those ids, so without this lookup any match
    processed in an earlier session -- most of them -- would have no id for
    video playback, the processing poller, or the latency report, even though
    the video is still on disk and the job row still exists.

    video_id/job_id are None when there genuinely is no Video or ProcessingJob
    row, and video_file_exists reports the file's real presence on disk rather
    than assuming a stored path is still valid.
    """
    match_id: str
    home_team: str | None = None
    away_team: str | None = None
    duration: float | None = None
    video_id: str | None = None
    video_filename: str | None = None
    video_file_exists: bool = False
    # Whether the annotated render (source clip + burned-in tracking boxes)
    # exists on disk. Reported so the player can DEFAULT to the processed
    # video instead of probing for it and handling a 404 as a normal case.
    processed_video_exists: bool = False
    # Why there is no annotated render, when there is none and a run has
    # finished. Read from the job's AnalysisResult (`overlay_render`), which
    # the pipeline writes whether the stage succeeded or not -- so the UI can
    # say "the encoder was unavailable" instead of showing an empty player
    # and letting the user infer that processing silently did nothing.
    # None when the render exists, or when no run has completed yet.
    processed_video_error: str | None = None
    # Container/codec actually written, e.g. "avc1". Informational, and the
    # honest answer to "what am I watching" for a re-encoded artifact.
    processed_video_codec: str | None = None
    # SOURCE clip geometry -- see TrackingWindowResponse.frame_width_px for
    # why the overlay needs this rather than the played video's own size.
    frame_width_px: int | None = None
    frame_height_px: int | None = None
    job_id: str | None = None
    job_status: str | None = None

class MatchListItem(BaseModel):
    """One row of GET /api/matches -- enough to CHOOSE a match, no more.

    `tracking_rows` is a real count, not a boolean dressed up as one, because
    "this match exists" and "this match has data worth opening" are different
    facts and the UI has to be able to tell a user which it is looking at. A
    match whose job failed still appears here: hiding it would leave the user
    unable to see that their upload exists at all.
    """
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
    density: float  # count / max_count in this grid, 0..1

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

    # Which coordinate system the cells are binned in.
    #
    # "pitch" is the real thing: metres on a 105x68 model, derived through a
    # validated homography, comparable between matches and camera angles.
    #
    # "image" is where the player appeared in the VIDEO FRAME. It is a
    # genuine measurement of the tracked boxes and needs no calibration, but
    # it is not a pitch position: it moves when the camera pans or zooms,
    # two matches shot from different angles cannot be compared, and no
    # distance in it means metres. It exists because on footage this
    # project's calibration cannot solve -- which is currently all of it --
    # the pitch heatmap is correctly but permanently empty, and "where in
    # frame was this player" is a real question that the stored data can
    # actually answer. Consumers must label it as image space; presenting it
    # as a pitch heatmap would be exactly the fabrication the empty pitch
    # grid exists to avoid.
    space: Literal["pitch", "image"] = "pitch"

    # Populated only for space="image": the frame the pixels were binned
    # over. None in pitch space, where the extent is the pitch model above.
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

# ---------------------------------------------------------------------------
# Pre-Match Health Intelligence (backend/api/prematch_health.py)
#
# A performance-readiness estimate from a self-reported questionnaire. Not a
# medical assessment, not a diagnosis.
# ---------------------------------------------------------------------------

class PreMatchQuestionnaireRequest(BaseModel):
    """One player's pre-match self-report.

    Every range below is enforced here so an out-of-range answer is rejected
    with a 422 before it reaches the scoring engine -- never silently clamped,
    which would turn a data-entry mistake into a real-looking reading.

    player_id is deliberately NOT a field: it comes from the path
    (POST /api/prematch_health/{player_id}/submit), so there is exactly one
    place it can be specified and no way for a path/body mismatch to arise.
    """

    # Only set when a Match row already exists for the upcoming game. A
    # genuine pre-match submission legitimately has none yet.
    match_id: str | None = None

    # --- sleep ---
    sleep_duration_hours: float = Field(gt=SLEEP_HOURS_MIN_EXCLUSIVE, le=SLEEP_HOURS_MAX)
    sleep_quality: int = Field(ge=SCALE_MIN, le=SCALE_MAX)
    # Accepts a `time` or an ISO time string ("22:30") -- Pydantic parses both.
    bedtime: time | None = None
    wake_time: time | None = None
    night_awakenings: int = Field(default=0, ge=0)

    # --- physical activity ---
    trained_last_24h: bool = False
    trained_last_48h: bool = False
    training_duration_minutes: float = Field(default=0.0, ge=0)
    training_intensity: int = Field(default=SCALE_MIN, ge=SCALE_MIN, le=SCALE_MAX)
    high_intensity_activity: bool = False
    # None is a real answer here ("no recent training to date from"), which is
    # why this is nullable rather than defaulted to 0.
    hours_since_last_training: float | None = Field(default=None, ge=0)

    # --- recovery ---
    fatigue: int = Field(ge=SCALE_MIN, le=SCALE_MAX)
    muscle_soreness: int = Field(ge=SCALE_MIN, le=SCALE_MAX)
    pain_level: int = Field(ge=SCALE_MIN, le=SCALE_MAX)
    perceived_readiness: int = Field(ge=SCALE_MIN, le=SCALE_MAX)

    # --- hydration & nutrition ---
    hydration_liters: float = Field(default=0.0, ge=0)
    nutrition_quality: int = Field(ge=SCALE_MIN, le=SCALE_MAX)
    hours_since_last_meal: float = Field(default=0.0, ge=0)

    # Free text. Echoed back on the response for a coach to read, and never
    # fed into any formula -- see ai/.../score.py, which attaches it to the
    # assessment only after scoring has finished.
    caffeine_or_supplement_notes: str | None = None

class PreMatchFactorResponse(BaseModel):
    """One input dimension, classified -- the explainability record behind
    the headline numbers."""
    dimension: str
    label: Literal["positive", "negative", "neutral"]
    score: float  # 0-100, always higher-is-better after normalization
    detail: str

class PreMatchAssessmentResponse(BaseModel):
    """The stable contract consumed by the frontend tab and by the nexus
    LLM Coach integration (nexus/sports/prematch_health.py).

    The coach receives this as already-computed text context and narrates it.
    It never recomputes any field here.
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

    # Additive beyond the core contract above -- useful to the UI and to any
    # consumer that wants the full reasoning rather than the two summary
    # lists.
    assessment_id: str
    questionnaire_id: str
    # 1-based per player. Submission order is tracked explicitly rather than
    # inferred from computed_at, whose resolution is the OS clock's -- see
    # PreMatchHealthAssessment.submission_index.
    submission_index: int
    factors: list[PreMatchFactorResponse]
    data_source: str
    notes: str | None = None
    disclaimer: str

class PreMatchFeaturesResponse(BaseModel):
    """The normalized feature vector on its own, for future ML/DL consumers.

    feature_directions ships alongside it because a bare vector of 0-100
    numbers is ambiguous without knowing which way each one points
    (hydration_score higher is better; muscle_soreness higher is worse).
    """
    assessment_id: str
    player_id: str
    match_id: str | None = None
    features: dict[str, Any]
    feature_directions: dict[str, str]
    method: str
    schema_version: str
    computed_at: datetime

# ---------------------------------------------------------------------------
# Pre-Match Psychology Intelligence (backend/api/psychology.py)
#
# A mental-READINESS estimate from a self-reported 13-item questionnaire. Not
# emotion detection, not a psychological or clinical assessment, not a
# diagnosis.
# ---------------------------------------------------------------------------

class PsychologyQuestionnaireRequest(BaseModel):
    """One player's pre-match mental-readiness self-report: 13 items, ~1-2
    minutes.

    Every range is enforced here so an out-of-range answer is rejected with a
    422 before it reaches the scoring engine -- never silently clamped, which
    would turn a data-entry mistake into a real-looking answer. The engine
    validates the identical bounds independently (both read the same
    constants), so a caller using ai/ directly gets the same rule.

    player_id is deliberately NOT a field: it comes from the path
    (POST /api/psychology/{player_id}/submit), so there is exactly one place it
    can be specified and no way for a path/body mismatch to arise.
    """

    # Only set when a Match row already exists for the upcoming game. A genuine
    # pre-match submission legitimately has none yet.
    match_id: str | None = None

    # The explicit opt-in to linking this self-report to CV-derived history.
    # Never inferred from player_id -- they are different ID spaces (a
    # ByteTrack tracking ID vs. an external identifier), and guessing a mapping
    # would attribute one player's on-pitch data to another's answers.
    cv_player_id: int | None = None

    # --- focus ---
    concentration_level: int = Field(ge=PSYCH_SCALE_MIN, le=PSYCH_SCALE_MAX)
    focus_maintenance: int = Field(ge=PSYCH_SCALE_MIN, le=PSYCH_SCALE_MAX)
    mental_clarity_raw: int = Field(ge=PSYCH_SCALE_MIN, le=PSYCH_SCALE_MAX)

    # --- stress ---
    pre_match_stress: int = Field(ge=PSYCH_SCALE_MIN, le=PSYCH_SCALE_MAX)
    importance_pressure: int = Field(ge=PSYCH_SCALE_MIN, le=PSYCH_SCALE_MAX)
    nervousness: int = Field(ge=PSYCH_SCALE_MIN, le=PSYCH_SCALE_MAX)

    # --- confidence ---
    performance_confidence: int = Field(ge=PSYCH_SCALE_MIN, le=PSYCH_SCALE_MAX)
    tactical_confidence_raw: int = Field(ge=PSYCH_SCALE_MIN, le=PSYCH_SCALE_MAX)

    # --- motivation ---
    match_motivation: int = Field(ge=PSYCH_SCALE_MIN, le=PSYCH_SCALE_MAX)
    competitive_motivation_raw: int = Field(ge=PSYCH_SCALE_MIN, le=PSYCH_SCALE_MAX)

    # --- pressure response ---
    mistake_recovery_speed: int = Field(ge=PSYCH_SCALE_MIN, le=PSYCH_SCALE_MAX)
    # The one categorical item. A Literal rather than a free string, so an
    # unrecognised value is a 422 rather than something the engine has to
    # decide how to interpret.
    pressure_performance_effect: Literal["improves", "no_change", "reduces"]
    post_error_calm: int = Field(ge=PSYCH_SCALE_MIN, le=PSYCH_SCALE_MAX)

class PsychologyAssessmentResponse(BaseModel):
    """The stable contract consumed by the frontend tab and by the nexus LLM
    Coach integration (nexus/sports/psychology.py).

    The coach receives this as already-computed context and narrates it. It
    never recomputes any field here.
    """
    player_id: str
    match_id: str | None = None
    cv_player_id: int | None = None

    # 0-100 integers. Not floats: these are estimates from 1-10 self-ratings,
    # and decimal places would imply a precision the input does not have.
    mental_readiness: int
    focus: int
    confidence: int
    stress: int  # higher = more reported stress, unlike the three above

    pressure_risk: Literal["low", "moderate", "high"]
    mental_performance_risk: Literal["low", "moderate", "high"]

    # dimension -> "positive" | "neutral" | "negative"
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
    # 1-based per player. Submission order is tracked explicitly rather than
    # inferred from computed_at, whose resolution is the OS clock's -- see
    # PsychologyAssessment.submission_index.
    submission_index: int
    data_source: str
    # Explicitly-labelled CV-derived corroboration, when a cv_player_id was
    # supplied and matching PlayerMetric rows existed. Empty in the common case.
    historical_context: list[str] = []
    disclaimer: str

class PsychologyFeaturesResponse(BaseModel):
    """The normalized feature vector on its own, for future ML/DL consumers.

    feature_directions ships alongside it because a bare vector of 0-100
    numbers is ambiguous without knowing which way each one points
    (focus_score higher is better; stress_score higher is worse).
    """
    assessment_id: str
    player_id: str
    match_id: str | None = None
    features: dict[str, Any]
    feature_directions: dict[str, str]
    method: str
    schema_version: str
    computed_at: datetime
