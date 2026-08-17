import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Annotated
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from kombu.exceptions import OperationalError
from sqlalchemy.orm import Session

from backend.api import calibration_debug, player_intelligence, prematch_health, psychology, simulation, tactical, tracking
from backend.pipeline.overlay_video import find_overlay_output, media_type_for
from backend.api.schemas import (
    AnalysisResultCreate,
    AnalysisResultResponse,
    JobStatusUpdate,
    PipelineLatencyReport,
    ProcessingStatusResponse,
    VideoUploadResponse,
)
from backend.database.models import AnalysisResult, Match, ProcessingJob, ProcessingStatus, Video, new_id
from backend.database.session import Base, engine, get_db
from backend.tasks import process_video_job

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "storage" / "uploads"
ALLOWED_VIDEO_TYPES = {
    "video/mp4",
    "video/mpeg",
    "video/quicktime",
    "video/x-msvideo",
    "application/octet-stream",
}

app = FastAPI(
    title="SportsStrategyCoachAI Backend",
    version="0.1.0",
    description="Video upload, processing status, and analysis JSON API.",
)

_default_cors_origins = "http://localhost:3000,http://127.0.0.1:3000"
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", _default_cors_origins).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tactical.router)
app.include_router(tactical.team_intel_router)
app.include_router(player_intelligence.router)
app.include_router(tracking.router)
app.include_router(prematch_health.router)
app.include_router(psychology.router)
app.include_router(simulation.router)
app.include_router(calibration_debug.router)


@app.on_event("startup")
def startup():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)


@app.get("/health")
def health():
    return {"status": "ok", "service": "sports-strategy-coach-ai"}


@app.post("/api/videos/upload", response_model=VideoUploadResponse, status_code=status.HTTP_201_CREATED)
def upload_video(
    file: Annotated[UploadFile, File()],
    metadata: Annotated[str | None, Form()] = None,
    db: Session = Depends(get_db),
):
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}",
        )
    metadata_json = None
    if metadata:
        try:
            metadata_json = json.loads(metadata)
        except json.JSONDecodeError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"metadata must be valid JSON: {error.msg}",
            ) from error
    file_id = new_id()
    extension = Path(file.filename or "video.mp4").suffix or ".mp4"
    stored_filename = f"{file_id}{extension.lower()}"
    storage_path = UPLOAD_DIR / stored_filename
    with storage_path.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    file_size = storage_path.stat().st_size

    meta = metadata_json or {}
    match = Match(
        home_team=meta.get("team", "Home"),
        away_team=meta.get("opponent", "Away"),
        video_path=str(storage_path),
        duration=0.0,
    )
    db.add(match)
    db.flush()

    video = Video(
        id=file_id,
        original_filename=file.filename or stored_filename,
        stored_filename=stored_filename,
        content_type=file.content_type,
        file_size=file_size,
        storage_path=str(storage_path),
        metadata_json=metadata_json,
        match_id=match.match_id,
    )
    job = ProcessingJob(
        video_id=video.id,
        status=ProcessingStatus.queued,
        progress=0,
        message="Video uploaded and queued for processing.",
    )
    db.add(video)
    db.add(job)
    db.commit()
    db.refresh(job)
    try:
        process_video_job.delay(job.id)
    except OperationalError:
        job.status = ProcessingStatus.failed
        job.error = "Processing queue unavailable — could not reach the task broker."
        job.message = "Upload saved, but processing could not be queued."
        job.completed_at = datetime.utcnow()
        db.commit()
        db.refresh(job)
    return VideoUploadResponse(
        video_id=video.id,
        job_id=job.id,
        match_id=match.match_id,
        filename=video.original_filename,
        status=job.status.value,
        message=job.message or "Video uploaded.",
    )


@app.get("/api/videos/{video_id}/file")
def get_video_file(video_id: str, db: Session = Depends(get_db)):
    """Streams the raw uploaded clip back for the tracking-overlay player
    (see TabMatchAnalysis.jsx) -- the video is written to disk at upload
    time (see upload_video() above), before the pipeline ever runs, so
    this is available immediately and independent of processing status."""
    video = db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    storage_path = Path(video.storage_path)
    if not storage_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video file missing on disk")
    return FileResponse(storage_path, media_type=video.content_type or "video/mp4")


@app.get("/api/videos/{video_id}/processed")
def get_processed_video_file(video_id: str, db: Session = Depends(get_db)):
    """Streams the ANNOTATED render -- the source clip with the pipeline's own
    tracking boxes, track ids and team colours burned in.
    """
    video = db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")

    rendered = find_overlay_output(video.storage_path, video.id)
    if rendered is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No annotated render for this video. It is written by the last "
                "pipeline stage, so it exists only after a completed run "
                "(and not at all when RENDER_OVERLAY_VIDEO=0). When a run "
                "completed but the render did not, the reason is on the job's "
                "analysis result under `overlay_render`."
            ),
        )
    return FileResponse(
        rendered,
        media_type=media_type_for(rendered),
        headers={"Accept-Ranges": "bytes"},
    )


@app.get("/api/processing/{job_id}", response_model=ProcessingStatusResponse)
def get_processing_status(job_id: str, db: Session = Depends(get_db)):
    job = db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processing job not found")
    return ProcessingStatusResponse(
        job_id=job.id,
        video_id=job.video_id,
        match_id=job.video.match_id if job.video else None,
        status=job.status.value,
        progress=job.progress,
        message=job.message,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@app.patch("/api/processing/{job_id}", response_model=ProcessingStatusResponse)
def update_processing_status(
    job_id: str, payload: JobStatusUpdate, db: Session = Depends(get_db)
):
    job = db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processing job not found")
    next_status = ProcessingStatus(payload.status)
    job.status = next_status
    job.progress = payload.progress
    job.message = payload.message
    job.error = payload.error
    job.updated_at = datetime.utcnow()
    if next_status == ProcessingStatus.processing and job.started_at is None:
        job.started_at = datetime.utcnow()
    if next_status in {ProcessingStatus.completed, ProcessingStatus.failed}:
        job.completed_at = datetime.utcnow()
    db.commit()
    db.refresh(job)
    return ProcessingStatusResponse(
        job_id=job.id,
        video_id=job.video_id,
        status=job.status.value,
        progress=job.progress,
        message=job.message,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@app.post("/api/processing/{job_id}/result", response_model=AnalysisResultResponse)
def save_analysis_result(
    job_id: str, payload: AnalysisResultCreate, db: Session = Depends(get_db)
):
    job = db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processing job not found")
    existing = db.query(AnalysisResult).filter(AnalysisResult.job_id == job_id).first()
    if existing:
        existing.result_json = payload.result
        existing.summary = payload.summary
    else:
        db.add(AnalysisResult(job_id=job_id, result_json=payload.result, summary=payload.summary))
    job.status = ProcessingStatus.completed
    job.progress = 100
    job.message = "Analysis result saved."
    job.error = None
    job.completed_at = datetime.utcnow()
    job.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(job)
    return AnalysisResultResponse(
        job_id=job.id,
        video_id=job.video_id,
        status=job.status.value,
        summary=payload.summary,
        result=payload.result,
    )


@app.get("/api/processing/{job_id}/result", response_model=AnalysisResultResponse)
def get_analysis_result(job_id: str, db: Session = Depends(get_db)):
    job = db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processing job not found")
    result = db.query(AnalysisResult).filter(AnalysisResult.job_id == job_id).first()
    return AnalysisResultResponse(
        job_id=job.id,
        video_id=job.video_id,
        status=job.status.value,
        summary=result.summary if result else None,
        result=result.result_json if result else None,
    )


@app.get("/api/pipeline/latency/{job_id}", response_model=PipelineLatencyReport)
def get_pipeline_latency(job_id: str, db: Session = Depends(get_db)):
    """
    Serves the REAL per-stage timing captured by backend/pipeline/latency.py
    during this job's run (backend/tasks.py writes it into
    AnalysisResult.result_json['pipeline_latency']). See that module's
    docstring for why "real" is load-bearing here: this replaces a
    """
    job = db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processing job not found")

    result = db.query(AnalysisResult).filter(AnalysisResult.job_id == job_id).first()
    if not result or "pipeline_latency" not in (result.result_json or {}):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No latency report for job_id={job_id} yet -- job may still be processing or failed "
                   f"before completing a stage.",
        )
    return PipelineLatencyReport(**result.result_json["pipeline_latency"])
