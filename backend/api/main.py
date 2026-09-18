import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from kombu.exceptions import OperationalError
from sqlalchemy.orm import Session

from backend.api import (
    calibration_debug,
    player_intelligence,
    prematch_health,
    psychology,
    simulation,
    tactical,
    tracking,
)
from backend.api.schemas import (
    AnalysisResultCreate,
    AnalysisResultResponse,
    JobStatusUpdate,
    PipelineLatencyReport,
    ProcessingStatusResponse,
    VideoUploadResponse,
)
from backend.auth.api_key import require_api_key
from backend.database.models import AnalysisResult, Match, ProcessingJob, ProcessingStatus, Video, new_id
from backend.database.session import Base, engine, get_db
from backend.pipeline.overlay_video import find_overlay_output, media_type_for
from backend.tasks import process_video_job

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "storage" / "uploads"
ALLOWED_VIDEO_TYPES = {
    "video/mp4",
    "video/mpeg",
    "video/quicktime",
    "video/x-msvideo",
    "video/webm",
}
# Some clients (curl without -H, a few browsers) send octet-stream for a
# perfectly ordinary .mp4. Rather than accept that blanket -- which made the
# allowlist mean "any file at all" -- accept it only when the filename still
# claims a video extension.
AMBIGUOUS_CONTENT_TYPES = {"application/octet-stream", None, ""}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mpeg", ".mpg", ".mov", ".avi", ".webm", ".mkv"}

# Without a cap, a single request could fill the storage volume. Override with
# MAX_UPLOAD_BYTES; default 2 GiB, comfortably above a full-match clip.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
UPLOAD_CHUNK_BYTES = 1024 * 1024


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup work. Replaces the deprecated @app.on_event("startup")."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # Schema bootstrap for SQLite only -- the local dev/test database, where a
    # migration step would just be friction. On any other backend (Postgres in
    # docker-compose.yml) the schema is owned by Alembic: creating tables here
    # too would let the running app and the migration history disagree without
    # anything noticing. Run: alembic upgrade head
    if engine.dialect.name == "sqlite":
        Base.metadata.create_all(bind=engine)
    else:
        logger.info("Non-SQLite database: schema is managed by Alembic (alembic upgrade head).")
    yield


app = FastAPI(
    title="SportsStrategyCoachAI Backend",
    version="0.1.0",
    description="Video upload, processing status, and analysis JSON API.",
    lifespan=lifespan,
    # Applied on the constructor, not per-router, so a router added later cannot
    # accidentally ship unauthenticated. Inert unless SSC_API_KEY is set.
    dependencies=[Depends(require_api_key)],
)

# The frontend calls this API from another origin (the Vite dev server on
# :3000, this app on :8000), so every browser request is cross-origin. Without
# CORSMiddleware the browser blocks the response, or the preflight OPTIONS
# request, before any application code runs, and nothing in this app can log
# it.
#
# allow_origins is explicit, never "*": browsers reject wildcard origins with
# credentials, and an explicit list is the one place to update when a deployed
# frontend origin is added.
#
# The list comes from CORS_ALLOWED_ORIGINS (see backend/.env.example) because
# `npm run dev` silently moves to 3001/3002 when 3000 is taken, and the browser
# then reports only "Failed to fetch". Add the real origin there instead of
# editing this file.
_default_cors_origins = "http://localhost:3000,http://127.0.0.1:3000"
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", _default_cors_origins).split(",")
    if origin.strip()
]

# Explicit lists rather than "*": allow_credentials=True means a wildcard here
# would let any origin named in CORS_ALLOWED_ORIGINS drive authenticated,
# cookie-bearing requests with arbitrary headers.
CORS_ALLOWED_METHODS = ["GET", "POST", "PATCH", "DELETE", "OPTIONS"]
CORS_ALLOWED_HEADERS = ["Accept", "Authorization", "Content-Type", "X-API-Key"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=CORS_ALLOWED_METHODS,
    allow_headers=CORS_ALLOWED_HEADERS,
)

# Register Phase 3 routers
app.include_router(tactical.router)
app.include_router(tactical.team_intel_router)
app.include_router(player_intelligence.router)
app.include_router(tracking.router)
app.include_router(prematch_health.router)
app.include_router(psychology.router)
app.include_router(simulation.router)
app.include_router(calibration_debug.router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "sports-strategy-coach-ai"}


@app.post("/api/videos/upload", response_model=VideoUploadResponse, status_code=status.HTTP_201_CREATED)
def upload_video(
    file: Annotated[UploadFile, File()],
    metadata: Annotated[str | None, Form()] = None,
    db: Session = Depends(get_db),
):
    extension = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    if file.content_type in AMBIGUOUS_CONTENT_TYPES:
        accepted = extension in ALLOWED_VIDEO_EXTENSIONS
    else:
        accepted = file.content_type in ALLOWED_VIDEO_TYPES
    if not accepted:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type} ({extension})",
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
    stored_filename = f"{file_id}{extension}"
    storage_path = UPLOAD_DIR / stored_filename
    file_size = 0
    try:
        with storage_path.open("wb") as output:
            while chunk := file.file.read(UPLOAD_CHUNK_BYTES):
                file_size += len(chunk)
                if file_size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=f"Upload exceeds the {MAX_UPLOAD_BYTES} byte limit.",
                    )
                output.write(chunk)
    except BaseException:
        # Never leave a partial file behind -- on the 413 path especially, that
        # would hand an attacker the disk fill the limit exists to prevent.
        storage_path.unlink(missing_ok=True)
        raise

    # Create the Match row here, at the one point in the system that knows this
    # Video exists, and return its id in the upload response so the frontend
    # can carry it forward to every /api/*_intelligence/{match_id} call.
    #
    # home_team/away_team/duration aren't known at upload time unless the
    # caller supplies them via `metadata` -- default to honest placeholders
    # rather than guessing team names; duration gets filled in for real
    # once the pipeline actually opens the video (see
    # backend/pipeline/runner.py::_estimate_fps and the corresponding
    # Match.duration update after ingest).
    meta = metadata_json or {}
    match = Match(
        home_team=meta.get("team", "Home"),
        away_team=meta.get("opponent", "Away"),
        video_path=str(storage_path),
        duration=0.0,
    )
    db.add(match)
    db.flush()  # need match.match_id before constructing Video below

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
    # The upload itself has already succeeded and been committed above -- the
    # file is on disk and Match/Video/ProcessingJob rows exist. Only the
    # hand-off to Celery can still fail here, and it fails by raising
    # kombu.exceptions.OperationalError when the broker (Redis) is
    # unreachable. Left unhandled that escaped as a raw 500 and stranded the
    # job at status=queued forever: nothing would ever pick it up, but the
    # frontend's status poller would keep waiting on it indefinitely. Mark the
    # job failed instead, so the poller sees a terminal state and the user is
    # told their upload was kept.
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

    Separate from /file rather than replacing it. The raw upload is available
    the instant the upload finishes; this one only exists after a successful
    run, and conflating them would mean the player either had nothing to show
    during processing or silently swapped the user's video for a different
    file. A 404 here means "not rendered", which the caller distinguishes
    from "no video at all" by /file still working.

    H.264/MP4, with WebM/VP8 as the fallback rung and as the format older
    runs were written in -- see backend/pipeline/overlay_video.py's CODEC
    note. The container is whatever actually got written, so the media type
    is derived from the file on disk rather than asserted here: serving an
    mp4 as video/webm makes the browser refuse a file it can decode.

    RANGE REQUESTS. Starlette's FileResponse honours the Range header, which
    this endpoint depends on rather than merely benefits from: OpenCV writes
    the mp4 index (moov) at the END of the file, so a browser must fetch the
    tail before it can play or seek at all. Replacing this with a plain
    streaming response that ignores Range would break seeking in the
    annotated video and, on a long clip, playback itself.
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
        # Lets the browser cache the render across seeks within a session
        # without re-fetching a file that, by construction, never changes
        # for a given video_id unless the video is re-processed.
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
    docstring for why "real" is load-bearing here: these numbers can only exist
    if this exact job actually ran this exact pipeline code.
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
