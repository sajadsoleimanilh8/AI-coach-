import enum
import uuid
from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Enum, Float, ForeignKey, Integer, JSON, String, Text
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
    """Banding shared by both risk fields on PreMatchHealthAssessment."""
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
    duration = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    frames = relationship("Frame", back_populates="match", cascade="all, delete-orphan")
    player_trackings = relationship("PlayerTracking", back_populates="match", cascade="all, delete-orphan")
    events = relationship("Event", back_populates="match", cascade="all, delete-orphan")
    player_metrics = relationship("PlayerMetric", back_populates="match", cascade="all, delete-orphan")
    team_metrics = relationship("TeamMetric", back_populates="match", cascade="all, delete-orphan")
    videos = relationship("Video", back_populates="match")
    calibration_status = relationship("CalibrationStatus", back_populates="match",
                                      cascade="all, delete-orphan")

    prematch_questionnaires = relationship("PreMatchQuestionnaire", back_populates="match")
    prematch_assessments = relationship("PreMatchHealthAssessment", back_populates="match")
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

    frame_id = Column(Integer, nullable=False, index=True)
    team_id = Column(String(64), nullable=True)
    pixel_x = Column(Float, nullable=False)
    pixel_y = Column(Float, nullable=False)
    pitch_x_m = Column(Float, nullable=True)
    pitch_y_m = Column(Float, nullable=True)
    homography_confidence = Column(Float, nullable=True)
    speed = Column(Float, nullable=True)
    distance = Column(Float, nullable=True)
    acceleration = Column(Float, nullable=True)

    body_orientation_deg = Column(Float, nullable=True)
    body_orientation_confidence = Column(Float, nullable=True)

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


class CalibrationSourceKind(str, enum.Enum):
    """Where a homography came from. Mirrors
    ai/computer_vision/frame_data.py::CalibrationSource exactly -- the CV
    layer's enum is the source of truth and this one persists it.
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

    frame_start = Column(Integer, nullable=False, index=True)
    frame_end = Column(Integer, nullable=False)

    valid = Column(Boolean, nullable=False, index=True)
    invalid_reason = Column(Text, nullable=True)

    confidence = Column(Float, nullable=False)
    reprojection_error_m = Column(Float, nullable=True)
    n_points = Column(Integer, nullable=False, default=0)
    source = Column(Enum(CalibrationSourceKind), nullable=False)

    solved_on_frame = Column(Integer, nullable=True)

    camera_motion = Column(String(16), nullable=True)
    camera_shift_px = Column(Float, nullable=True)

    homography_matrix = Column(JSON, nullable=True)

    method = Column(Enum(MetricMethod), nullable=False)

    computed_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    match = relationship("Match", back_populates="calibration_status")

    def __repr__(self) -> str:
        return (f"<CalibrationStatus match={self.match_id} "
                f"frames={self.frame_start}-{self.frame_end} valid={self.valid} "
                f"source={self.source} conf={self.confidence:.3f}>")



class PreMatchQuestionnaire(Base):
    """The raw self-report, exactly as submitted and validated."""
    __tablename__ = "prematch_questionnaires"
    id = Column(String(36), primary_key=True, default=new_id)
    player_id = Column(String(64), nullable=False, index=True)

    match_id = Column(String(36), ForeignKey("matches.match_id"), nullable=True, index=True)

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

    questionnaire_id = Column(
        String(36),
        ForeignKey("prematch_questionnaires.id"),
        nullable=False,
        unique=True,
        index=True,
    )

    player_id = Column(String(64), nullable=False, index=True)
    match_id = Column(String(36), ForeignKey("matches.match_id"), nullable=True, index=True)

    submission_index = Column(Integer, nullable=False, default=1, index=True)

    features = Column(JSON, nullable=False)

    physical_readiness = Column(Float, nullable=False)
    fatigue_score = Column(Float, nullable=False)
    recovery_score = Column(Float, nullable=False)

    performance_risk = Column(Enum(RiskLevel), nullable=False)
    workload_risk = Column(Enum(RiskLevel), nullable=False)

    factors = Column(JSON, nullable=False)

    method = Column(Enum(MetricMethod), nullable=False)

    schema_version = Column(String(16), nullable=False)

    computed_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    questionnaire = relationship("PreMatchQuestionnaire", back_populates="assessment")
    match = relationship("Match", back_populates="prematch_assessments")

    def __repr__(self) -> str:
        return (
            f"<PreMatchHealthAssessment id={self.id} player_id={self.player_id} "
            f"readiness={self.physical_readiness} risk={self.performance_risk}>"
        )


class PsychologyAssessment(Base):
    """One player's mental-readiness assessment for one questionnaire."""
    __tablename__ = "psychology_assessments"
    assessment_id = Column(String(36), primary_key=True, default=new_id)

    player_id = Column(String(64), nullable=False, index=True)

    cv_player_id = Column(Integer, nullable=True)

    match_id = Column(String(36), ForeignKey("matches.match_id"), nullable=True, index=True)

    responses = Column(JSON, nullable=False)

    features = Column(JSON, nullable=False)

    mental_readiness = Column(Float, nullable=False)
    focus = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    stress = Column(Float, nullable=False)

    pressure_risk = Column(String(16), nullable=False)
    mental_performance_risk = Column(String(16), nullable=False)

    factors = Column(JSON, nullable=False)

    sub_scores = Column(JSON, nullable=False)

    method = Column(Enum(MetricMethod), nullable=False)

    confidence_level = Column(Enum(MetricConfidence), nullable=False)

    schema_version = Column(String(16), nullable=False)

    submission_index = Column(Integer, nullable=False, default=1, index=True)

    submitted_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    computed_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    match = relationship("Match", back_populates="psychology_assessments")

    def __repr__(self) -> str:
        return (
            f"<PsychologyAssessment id={self.assessment_id} player_id={self.player_id} "
            f"readiness={self.mental_readiness} risk={self.mental_performance_risk}>"
        )
