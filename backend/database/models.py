import enum
import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from backend.database.session import Base


def new_id() -> str:
    return str(uuid.uuid4())

class ProcessingStatus(str, enum.Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"

class MetricMethod(str, enum.Enum):
    ml_trained = "ml_trained"
    deterministic = "deterministic"
    heuristic_proxy = "heuristic_proxy"

class MetricConfidence(str, enum.Enum):
    normal = "normal"
    low_sample = "low_sample"
    low_upstream_confidence = "low_upstream_confidence"

class RiskLevel(str, enum.Enum):
    """Banding shared by both risk fields on PreMatchHealthAssessment.

    Not a duplicate of MetricConfidence: that enum says how much to trust a
    number, this one is the number's own verdict. A pre-match assessment can
    be perfectly trustworthy (complete, validated self-report) and still say
    "high" risk -- collapsing the two would make those indistinguishable.
    """
    low = "low"
    moderate = "moderate"
    high = "high"

class Video(Base):
    __tablename__ = "videos"
    id = Column(String(36), primary_key=True, default=new_id)
    original_filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False, unique=True)
    content_type = Column(String(120), nullable=True)
    file_size = Column(Integer, nullable=False)
    storage_path = Column(Text, nullable=False)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Links an upload to the Match row its PlayerMetric/TeamMetric/Event rows
    # are written under.
    # Nullable because the Match row is created immediately after the Video
    # row in the same upload request (see backend/api/main.py::upload_video)
    # -- there's a brief instant where the Video exists and the Match
    # doesn't yet, not because this link is meant to stay unset long-term.
    match_id = Column(String(36), ForeignKey("matches.match_id"), nullable=True, index=True)

    jobs = relationship("ProcessingJob", back_populates="video", cascade="all, delete-orphan")
    match = relationship("Match", back_populates="videos")

class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    id = Column(String(36), primary_key=True, default=new_id)
    video_id = Column(String(36), ForeignKey("videos.id"), nullable=False, index=True)
    status = Column(Enum(ProcessingStatus), default=ProcessingStatus.queued, nullable=False)
    progress = Column(Integer, default=0, nullable=False)
    message = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    video = relationship("Video", back_populates="jobs")
    result = relationship("AnalysisResult", back_populates="job", uselist=False, cascade="all, delete-orphan")

class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    id = Column(String(36), primary_key=True, default=new_id)
    job_id = Column(String(36), ForeignKey("processing_jobs.id"), nullable=False, unique=True, index=True)
    result_json = Column(JSON, nullable=False)
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    job = relationship("ProcessingJob", back_populates="result")

class Match(Base):
    """General match information. Root of the football analysis schema."""
    __tablename__ = "matches"
    match_id = Column(String(36), primary_key=True, default=new_id)
    home_team = Column(String(255), nullable=False)
    away_team = Column(String(255), nullable=False)
    video_path = Column(Text, nullable=False)
    duration = Column(Float, nullable=False)  # seconds
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    frames = relationship("Frame", back_populates="match", cascade="all, delete-orphan")
    player_trackings = relationship("PlayerTracking", back_populates="match", cascade="all, delete-orphan")
    events = relationship("Event", back_populates="match", cascade="all, delete-orphan")
    player_metrics = relationship("PlayerMetric", back_populates="match", cascade="all, delete-orphan")
    team_metrics = relationship("TeamMetric", back_populates="match", cascade="all, delete-orphan")
    videos = relationship("Video", back_populates="match")
    # Same cascade reasoning as frames/player_trackings above: a calibration
    # episode describes THIS match's camera geometry and is meaningless
    # without it.
    calibration_status = relationship("CalibrationStatus", back_populates="match",
                                      cascade="all, delete-orphan")

    # Deliberately NOT cascade="all, delete-orphan", unlike every relationship
    # above it. Those rows are derived from this match's video and are
    # meaningless without it, so deleting the match should delete them. A
    # pre-match questionnaire is the opposite: it is a player's own
    # self-report, submitted before any video exists, and it is valid on its
    # own (match_id is nullable precisely because a questionnaire routinely
    # has no match yet). Deleting a match must not destroy it -- SQLAlchemy's
    # default here nullifies the FK instead, leaving the questionnaire and its
    # assessment intact but unlinked, which is the honest outcome.
    prematch_questionnaires = relationship("PreMatchQuestionnaire", back_populates="match")
    prematch_assessments = relationship("PreMatchHealthAssessment", back_populates="match")
    # Same reasoning as the two above: a psychology self-report stands on its
    # own and must survive the deletion of a match it happened to be linked to.
    psychology_assessments = relationship("PsychologyAssessment", back_populates="match")

    def __repr__(self) -> str:
        return f"<Match id={self.match_id} {self.home_team} vs {self.away_team}>"

class Frame(Base):
    """A processed video frame belonging to a match."""
    __tablename__ = "frames"
    frame_id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(36), ForeignKey("matches.match_id"), nullable=False, index=True)
    frame_number = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)
    fps = Column(Float, nullable=False)
    match = relationship("Match", back_populates="frames")

class PlayerDetection(Base):
    __tablename__ = "player_detections"
    detection_id = Column(String(36), primary_key=True, default=new_id)
    frame_id = Column(Integer, ForeignKey("frames.frame_id"), nullable=False, index=True)
    player_id = Column(Integer, nullable=False)
    team_id = Column(String(64), nullable=True)
    team_assignment_confidence = Column(Float, nullable=True)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    width = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)

class BallDetection(Base):
    __tablename__ = "ball_detections"
    detection_id = Column(String(36), primary_key=True, default=new_id)
    frame_id = Column(Integer, ForeignKey("frames.frame_id"), nullable=False, index=True)
    ball_x = Column(Float, nullable=False)
    ball_y = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)

class PlayerTracking(Base):
    __tablename__ = "player_tracking"
    tracking_id = Column(String(36), primary_key=True, default=new_id)
    match_id = Column(String(36), ForeignKey("matches.match_id"), nullable=False, index=True)
    player_id = Column(Integer, nullable=False)

    # Deliberately NOT a ForeignKey to frames.frame_id. `frames` rows are
    # persisted only for a FRAME_PERSIST_STRIDE-sampled subset (see
    # backend/pipeline/runner.py), with an autoincrement PK unrelated to the
    # video frame number, while every TrackingPoint carries the real frame
    # number (0, 1, 2, ...). The two ID spaces do not correspond, so this
    # stores the real frame number. Nothing joins on this column as an FK.
    frame_id = Column(Integer, nullable=False, index=True)  # actual video frame number, NOT a frames.frame_id FK
    team_id = Column(String(64), nullable=True)
    pixel_x = Column(Float, nullable=False)
    pixel_y = Column(Float, nullable=False)
    pitch_x_m = Column(Float, nullable=True)
    pitch_y_m = Column(Float, nullable=True)
    homography_confidence = Column(Float, nullable=True)
    speed = Column(Float, nullable=True)
    distance = Column(Float, nullable=True)
    acceleration = Column(Float, nullable=True)

    # Pose / body orientation. Nullable for the
    # same reason pitch_x_m/homography_confidence are nullable: a frame
    # can legitimately have no reading (pose not sampled this frame --
    # see POSE_SAMPLE_STRIDE in backend/pipeline/runner.py -- or MediaPipe
    # found no visible shoulder landmarks). None means "not measured",
    # never silently defaulted to 0.0 -- a 0deg orientation is a real,
    # different fact from "we don't know". Downstream consumers must
    # check body_orientation_confidence before trusting the angle, same
    # pattern as homography_confidence gating pitch_x_m/pitch_y_m.
    body_orientation_deg = Column(Float, nullable=True)         # shoulder-line angle, 0-360
    body_orientation_confidence = Column(Float, nullable=True)  # min visibility of the two shoulder landmarks, 0-1

    match = relationship("Match", back_populates="player_trackings")

class Event(Base):
    __tablename__ = "events"
    event_id = Column(String(36), primary_key=True, default=new_id)
    match_id = Column(String(36), ForeignKey("matches.match_id"), nullable=False, index=True)
    frame_id = Column(Integer, ForeignKey("frames.frame_id"), nullable=True, index=True)
    event_type = Column(String(64), nullable=False)
    player_id = Column(Integer, nullable=True)
    related_player_id = Column(Integer, nullable=True)
    team_id = Column(String(64), nullable=True)
    pitch_x_m = Column(Float, nullable=True)
    pitch_y_m = Column(Float, nullable=True)
    homography_confidence = Column(Float, nullable=True)
    timestamp = Column(Float, nullable=False)
    metadata_json = Column("metadata", JSON, nullable=True)
    match = relationship("Match", back_populates="events")

class PlayerMetric(Base):
    __tablename__ = "player_metrics"
    metric_id = Column(String(36), primary_key=True, default=new_id)
    match_id = Column(String(36), ForeignKey("matches.match_id"), nullable=False, index=True)
    player_id = Column(Integer, nullable=False)
    metric_name = Column(String(120), nullable=False)
    value = Column(Float, nullable=True)
    method = Column(Enum(MetricMethod), nullable=False)
    confidence = Column(Enum(MetricConfidence), nullable=False)
    sample_size = Column(Integer, nullable=False)
    sub_scores = Column(JSON, nullable=False)
    computed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    schema_version = Column(String(16), nullable=False)
    match = relationship("Match", back_populates="player_metrics")

class TeamMetric(Base):
    __tablename__ = "team_metrics"
    metric_id = Column(String(36), primary_key=True, default=new_id)
    match_id = Column(String(36), ForeignKey("matches.match_id"), nullable=False, index=True)
    team_id = Column(String(64), nullable=False)
    metric_name = Column(String(120), nullable=False)

    # Two value columns, not one. Formation Detection stores a template name
    # like "4-3-3"
    # (ai/computer_vision/tactical_analysis/formation_detection.py), while
    # numeric metrics -- team_rating, xG, compactness_score -- need real Float
    # semantics for sorting, aggregation and range queries. SQLite silently
    # tolerates a string in a Float column; Postgres, the docker-compose.yml
    # target, raises. Exactly one of the two is set per row;
    # backend/api/tactical.py picks whichever is non-null for the response
    # `value`, so the split is invisible to the API contract.
    value_numeric = Column(Float, nullable=True)
    value_label = Column(String(64), nullable=True)

    method = Column(Enum(MetricMethod), nullable=False)
    confidence = Column(Enum(MetricConfidence), nullable=False)
    confidence_score = Column(Float, nullable=True)
    sample_size = Column(Integer, nullable=False)
    sub_scores = Column(JSON, nullable=False)
    computed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    schema_version = Column(String(16), nullable=False)
    match = relationship("Match", back_populates="team_metrics")

    @property
    def value(self):
        """Convenience accessor mirroring the pre-split single-`value`
        shape for any code (or future migration) that just wants
        whichever value is set, without caring which column it lives in."""
        return self.value_label if self.value_label is not None else self.value_numeric

    def __repr__(self) -> str:
        return f"<TeamMetric id={self.metric_id} team_id={self.team_id} metric={self.metric_name} value={self.value}>"

# ---------------------------------------------------------------------------
# Calibration status / history
#
# Before this table, calibration existed ONLY as flat JSON files under
# calibrations/{match_id}.json -- one record per match, overwritten in place,
# with no history and nothing queryable. That was tenable while calibration
# was a single manual click-session per match. It is not tenable now that
# auto_calibration.py solves per frame, carries a calibration forward across
# static-camera stretches, and rejects implausible recalculations: those are
# decisions taken thousands of times per video, and "was this match's
# geometry trustworthy, and over which frames" became a question the JSON
# file could not answer.
#
# One row per CALIBRATION EPISODE, not per frame. An episode is a contiguous
# frame range over which one homography was in force -- solved once, then
# reused while the camera stayed static. Writing a row per frame would be
# ~90k rows for a 50-minute clip and would say the same thing 3,000 times in
# a row; frame_start/frame_end make the same information range-queryable at
# a fraction of the size.
# ---------------------------------------------------------------------------

class CalibrationSourceKind(str, enum.Enum):
    """Where a homography came from. Mirrors
    ai/computer_vision/frame_data.py::CalibrationSource exactly -- the CV
    layer's enum is the source of truth and this one persists it.

    `carried` is a first-class value, not a variant of `model`: a frame
    whose geometry was solved 400 frames earlier and reused is a materially
    different claim from one solved on the frame itself, and collapsing the
    two would hide exactly the drift this table exists to make visible.
    """
    manual = "manual"
    model = "model"
    carried = "carried"
    none = "none"


class CalibrationStatus(Base):
    """One calibration episode for one match, over a frame range."""
    __tablename__ = "calibration_status"

    calibration_id = Column(String(36), primary_key=True, default=new_id)
    match_id = Column(String(36), ForeignKey("matches.match_id"), nullable=False, index=True)

    # Inclusive frame range this calibration was in force for. Indexed
    # together with match_id because "what was the calibration at frame N"
    # is the query this table exists to serve.
    frame_start = Column(Integer, nullable=False, index=True)
    frame_end = Column(Integer, nullable=False)

    # The single source of truth downstream reads. Deliberately stored as
    # its own column rather than re-derived from `confidence` at query
    # time: validity is confidence AND geometric consistency against the
    # field model, and a consumer recomputing "confidence >= 0.6" would
    # silently disagree with the pipeline on frames the geometry check
    # rejected.
    valid = Column(Boolean, nullable=False, index=True)
    invalid_reason = Column(Text, nullable=True)

    confidence = Column(Float, nullable=False)
    reprojection_error_m = Column(Float, nullable=True)
    n_points = Column(Integer, nullable=False, default=0)
    source = Column(Enum(CalibrationSourceKind), nullable=False)

    # The frame the homography was actually solved on -- equal to
    # frame_start for a fresh solve, EARLIER than it for a carried one.
    solved_on_frame = Column(Integer, nullable=True)

    # Camera motion classification at frame_start ("static"/"panning"/
    # "cut"/"unknown"), and the measured median feature shift in pixels.
    # Nullable because motion is genuinely unmeasured on the first frame
    # and on featureless frames -- None means "not measured", never 0.0,
    # same convention as body_orientation_deg on PlayerTracking.
    camera_motion = Column(String(16), nullable=True)
    camera_shift_px = Column(Float, nullable=True)

    # The 3x3 matrix itself, as a nested list. Stored so a run can be
    # re-projected and audited later without re-running the model -- the
    # flat JSON files kept it and losing it here would make this table
    # strictly less useful than what it replaces.
    homography_matrix = Column(JSON, nullable=True)

    # ml_trained when the pose model solved it, deterministic when the
    # operator clicked it: a manual homography is exact DLT/RANSAC over
    # human-supplied correspondences with no learned component. Neither is
    # heuristic_proxy -- both are real measurements with a real
    # reprojection error, which is precisely what distinguishes them from
    # the goalkeeper/referee inference documented in
    # docs/pipeline_architecture.md.
    method = Column(Enum(MetricMethod), nullable=False)

    computed_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    match = relationship("Match", back_populates="calibration_status")

    def __repr__(self) -> str:
        return (f"<CalibrationStatus match={self.match_id} "
                f"frames={self.frame_start}-{self.frame_end} valid={self.valid} "
                f"source={self.source} conf={self.confidence:.3f}>")


# ---------------------------------------------------------------------------
# Pre-Match Health Intelligence
#
# Self-reported pre-match questionnaire -> deterministic readiness scoring
# (ai/performance_ai/match_readiness_predictor/). A performance-readiness
# estimate, not a medical assessment.
#
# Split into two tables on the same principle as ProcessingJob ->
# AnalysisResult above: what was submitted is kept separate from what was
# computed from it. That separation is what makes an assessment auditable --
# the raw answers stay exactly as the player gave them, so any score can be
# recomputed and checked against them later, and a formula change (tracked by
# schema_version) can be re-run over historical submissions without the
# originals having been overwritten.
#
# Note this player_id is a String, unlike the Integer player_id on
# PlayerDetection / PlayerTracking / Event / PlayerMetric. Those are ByteTrack
# tracking IDs scoped to one processed video, carrying no cross-match
# identity (see backend/api/player_intelligence.py's module comment). A
# questionnaire is filled in before any tracking exists, so there is no
# tracking ID to reuse; this is a plain external identifier supplied by the
# caller. The two ID spaces are unrelated and must not be joined.
# ---------------------------------------------------------------------------

class PreMatchQuestionnaire(Base):
    """The raw self-report, exactly as submitted and validated."""
    __tablename__ = "prematch_questionnaires"
    id = Column(String(36), primary_key=True, default=new_id)
    player_id = Column(String(64), nullable=False, index=True)

    # Nullable by design, not as a workaround: Match rows are only created at
    # video-upload time (backend/api/main.py::upload_video), and a genuine
    # pre-match questionnaire is submitted before that video exists. "No
    # match linked yet" is a normal, expected state here -- unlike
    # Video.match_id above, where the gap lasts one flush.
    match_id = Column(String(36), ForeignKey("matches.match_id"), nullable=True, index=True)

    # One JSON blob rather than ~19 scalar columns. Same justification as
    # AnalysisResult.result_json and Video.metadata_json: these answers are
    # write-once and always read back as a whole, nothing queries or
    # aggregates across an individual answer, and the questionnaire's field
    # set is expected to grow. The columns that ARE queried (player_id,
    # match_id, submitted_at) are real indexed columns, not JSON lookups.
    # Validation happens before this is written (Pydantic at the API edge,
    # PreMatchQuestionnaireInput in the engine), so this never holds an
    # out-of-range answer.
    questionnaire_json = Column(JSON, nullable=False)

    submitted_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    match = relationship("Match", back_populates="prematch_questionnaires")
    assessment = relationship(
        "PreMatchHealthAssessment",
        back_populates="questionnaire",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<PreMatchQuestionnaire id={self.id} player_id={self.player_id} match_id={self.match_id}>"

class PreMatchHealthAssessment(Base):
    """The computed output for exactly one questionnaire."""
    __tablename__ = "prematch_health_assessments"
    id = Column(String(36), primary_key=True, default=new_id)

    # unique -> the 1:1 with the questionnaire is enforced by the database,
    # not just by uselist=False on the relationship. Same shape as
    # AnalysisResult.job_id.
    questionnaire_id = Column(
        String(36),
        ForeignKey("prematch_questionnaires.id"),
        nullable=False,
        unique=True,
        index=True,
    )

    # Denormalized from the questionnaire so "latest assessment for this
    # player" and "assessments for this match" are single-table indexed
    # lookups instead of a join on every dashboard read -- the same
    # denormalization PlayerTracking.match_id already uses.
    player_id = Column(String(64), nullable=False, index=True)
    match_id = Column(String(36), ForeignKey("matches.match_id"), nullable=True, index=True)

    # 1-based, per player, assigned at write time. This is what "latest" and
    # the history ordering actually sort on -- NOT computed_at.
    #
    # computed_at cannot order these reliably: datetime.utcnow() inherits the
    # OS clock granularity, which on Windows is ~15ms, so two submissions a
    # few milliseconds apart get byte-identical timestamps and "most recent"
    # becomes whichever row the database happens to return first. That is not
    # hypothetical -- a player who submits, spots a typo, and resubmits does
    # it well inside one tick, and would then be shown their stale assessment.
    # An explicit counter makes the order exact regardless of clock
    # resolution, and doubles as a human-meaningful "submission #N" for the
    # dashboard. computed_at is still stored and returned; it just answers
    # "when", not "in what order".
    submission_index = Column(Integer, nullable=False, default=1, index=True)

    # The full normalized feature vector (inputs + the three headline scores),
    # stored so a future ML/DL model can train on exactly what the heuristic
    # saw, and so any score here can be traced back through the formulas
    # without recomputing from the raw answers.
    features = Column(JSON, nullable=False)

    physical_readiness = Column(Float, nullable=False)
    fatigue_score = Column(Float, nullable=False)
    recovery_score = Column(Float, nullable=False)

    performance_risk = Column(Enum(RiskLevel), nullable=False)
    workload_risk = Column(Enum(RiskLevel), nullable=False)

    # Per-dimension "positive"/"negative"/"neutral" classification with the
    # sub-score behind each one. This is the explainability record: kept so
    # the reasoning survives alongside the headline number instead of being
    # discarded once the score is computed.
    factors = Column(JSON, nullable=False)

    # Reuses the existing MetricMethod enum rather than introducing a parallel
    # one -- heuristic_proxy today, ml_trained the day a real model replaces
    # HeuristicReadinessScorer behind the same ReadinessScorer interface.
    method = Column(Enum(MetricMethod), nullable=False)

    # Which version of the formulas produced these numbers. Assessments
    # computed under different constants are not comparable, so this is what
    # a historical comparison has to check first. Same role as
    # PlayerMetric.schema_version.
    schema_version = Column(String(16), nullable=False)

    # No confidence column, unlike PlayerMetric/TeamMetric. MetricConfidence's
    # three states are all about upstream measurement quality --
    # low_sample (too few tracked events) and low_upstream_confidence (poor
    # homography/tracking). Neither can occur here: the input is one complete,
    # validated self-report, so there is no sample size to be short of and no
    # upstream tracking to be uncertain about. Storing "normal" on every row
    # would be a constant dressed up as a measurement. What genuinely varies
    # -- which scoring tier ran -- is already carried by `method`, and the
    # inherent limitation of the data source is stated in the API response and
    # the UI copy instead.
    computed_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    questionnaire = relationship("PreMatchQuestionnaire", back_populates="assessment")
    match = relationship("Match", back_populates="prematch_assessments")

    def __repr__(self) -> str:
        return (
            f"<PreMatchHealthAssessment id={self.id} player_id={self.player_id} "
            f"readiness={self.physical_readiness} risk={self.performance_risk}>"
        )

# ---------------------------------------------------------------------------
# Pre-Match Psychology Intelligence
#
# Self-reported 13-item questionnaire -> deterministic mental-readiness scoring
# (ai/psychology_ai/). A mental-READINESS estimate: NOT emotion detection, NOT
# a psychological or clinical assessment, and NOT a diagnosis. The column names
# and every string stored in them stay in performance vocabulary
# ("pressure_response", "mental_performance_risk") for that reason.
#
# One table, not the two PreMatchHealthAssessment uses. The split there exists
# so a formula change can be re-run over the original answers; here the same
# guarantee is met by storing `responses` (the raw validated answers) on the
# assessment row itself. The health questionnaire justified its own table by
# being ~19 fields expected to grow; this one is a fixed 13-item form whose
# answers are only ever read back whole, alongside the scores computed from
# them. A second table would be a join with nothing on the other side of it.
#
# Note this player_id is a String, unlike the Integer player_id on
# PlayerDetection / PlayerTracking / Event / PlayerMetric. Those are ByteTrack
# tracking IDs scoped to one processed video and carry no cross-match identity
# (see backend/api/player_intelligence.py's module comment). A questionnaire is
# filled in before any tracking exists, so there is no tracking ID to reuse.
# The two ID spaces are unrelated and are never joined by guesswork -- which is
# exactly what cv_player_id below is for.
# ---------------------------------------------------------------------------

class PsychologyAssessment(Base):
    """One player's mental-readiness assessment for one questionnaire."""
    __tablename__ = "psychology_assessments"
    assessment_id = Column(String(36), primary_key=True, default=new_id)

    player_id = Column(String(64), nullable=False, index=True)

    # The explicit, caller-supplied bridge into CV-derived data. Nullable and
    # never inferred: the only way a psychology submission is linked to
    # PlayerMetric rows is if whoever submitted it stated the tracking ID
    # outright. Guessing a mapping between the two ID spaces would silently
    # attribute one player's on-pitch data to another's self-report.
    cv_player_id = Column(Integer, nullable=True)

    # Nullable by design: Match rows are created at video-upload time, and a
    # genuine pre-match questionnaire is submitted before that video exists.
    match_id = Column(String(36), ForeignKey("matches.match_id"), nullable=True, index=True)

    # The raw validated answers, exactly as submitted. Kept so any score here
    # can be recomputed and checked against its inputs later, and so a formula
    # change (tracked by schema_version) can be re-run over historical
    # submissions without the originals having been overwritten.
    responses = Column(JSON, nullable=False)

    # The normalized 0-100 feature vector -- the shape a future ML/DL model
    # would train on and predict from, stored so it can train on exactly what
    # the heuristic saw.
    features = Column(JSON, nullable=False)

    # Top-level columns rather than JSON keys because these four are what
    # history charts and trend queries actually select and order by. Everything
    # else that varies lives in the JSON columns.
    mental_readiness = Column(Float, nullable=False)
    focus = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    stress = Column(Float, nullable=False)

    # Reuses RiskLevel's vocabulary (low/moderate/high) but stored as plain
    # String, matching how the ai/ engine emits them. Kept as strings rather
    # than Enum(RiskLevel) only because these two fields are nullable in the
    # degraded case, and a null enum reads worse than a null string.
    pressure_risk = Column(String(16), nullable=False)
    mental_performance_risk = Column(String(16), nullable=False)

    # Per-dimension "positive"/"neutral"/"negative" classification. This is the
    # explainability record: the reasoning survives alongside the headline
    # numbers instead of being discarded once they are computed.
    factors = Column(JSON, nullable=False)

    # The per-domain PlayerMetric-shaped rows (focus, confidence, stress,
    # motivation, pressure index, readiness) behind the headline scores.
    sub_scores = Column(JSON, nullable=False)

    # Reuses the existing MetricMethod enum rather than introducing a parallel
    # one -- heuristic_proxy today, ml_trained the day a real model replaces
    # HeuristicReadinessModel behind the same PsychologyReadinessModel interface.
    method = Column(Enum(MetricMethod), nullable=False)

    # Unlike PreMatchHealthAssessment, which deliberately has no confidence
    # column, this one does -- and the difference is real rather than
    # inconsistency. That module scores one complete self-report through a
    # single formula, so its confidence could only ever be the constant
    # "normal". This module composes six independently-gated domain scorers and
    # can optionally fold in CV-derived history, so the tier genuinely varies:
    # low_sample when a domain's answers are missing, low_upstream_confidence
    # when requested historical proxies came back unusable. Storing what varies
    # is the whole point of the column.
    confidence_level = Column(Enum(MetricConfidence), nullable=False)

    # Which version of the formulas produced these numbers. Assessments
    # computed under different constants are not comparable, so this is what a
    # historical comparison has to check first. Same role as
    # PlayerMetric.schema_version.
    schema_version = Column(String(16), nullable=False)

    # 1-based, per player, assigned at write time. This is what "latest" and
    # the history ordering sort on -- NOT computed_at.
    #
    # Not in the original column spec, and added anyway for the reason
    # PreMatchHealthAssessment.submission_index documents at length:
    # datetime.utcnow() inherits the OS clock granularity, ~15ms on Windows, so
    # two submissions a few milliseconds apart get byte-identical timestamps
    # and "most recent" becomes whichever row the database happens to return
    # first. A player who submits, spots a mistyped answer, and resubmits does
    # it well inside one tick, and would then be shown their stale assessment.
    # Ordering on an explicit counter is exact regardless of clock resolution.
    submission_index = Column(Integer, nullable=False, default=1, index=True)

    submitted_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    computed_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    match = relationship("Match", back_populates="psychology_assessments")

    def __repr__(self) -> str:
        return (
            f"<PsychologyAssessment id={self.assessment_id} player_id={self.player_id} "
            f"readiness={self.mental_readiness} risk={self.mental_performance_risk}>"
        )
