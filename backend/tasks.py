from datetime import datetime

from backend.api.cache import invalidate_match_cache
from backend.celery_app import celery_app
from backend.database.models import AnalysisResult, ProcessingJob, ProcessingStatus
from backend.database.session import SessionLocal
from backend.pipeline.latency import PipelineTimer
from backend.pipeline.runner import PipelineAssetError, run_pipeline


@celery_app.task(name="backend.tasks.process_video_job", bind=True)
def process_video_job(self, job_id: str):
    """
    Async entry point for the video processing pipeline.
    """
    db = SessionLocal()
    try:
        job = db.get(ProcessingJob, job_id)
        if job is None:
            return {"job_id": job_id, "status": "failed", "error": "job not found"}

        job.status = ProcessingStatus.processing
        job.progress = 5
        job.message = "Worker picked up job."
        job.started_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        db.commit()

        def on_progress(pct: int, message: str) -> None:
            job.progress = pct
            job.message = message
            job.updated_at = datetime.utcnow()
            db.commit()

        timer = PipelineTimer(job_id=job_id, match_id=job.video.match_id if job.video else None)

        try:
            result = run_pipeline(db, job, timer, progress_cb=on_progress)
        except PipelineAssetError as asset_error:
            job.status = ProcessingStatus.failed
            job.error = f"Missing pipeline asset: {asset_error}"
            job.message = "Processing failed -- missing required asset."
            job.updated_at = datetime.utcnow()
            job.completed_at = datetime.utcnow()
            db.commit()
            return {"job_id": job_id, "status": "failed", "error": str(asset_error)}

        latency_report = timer.to_report_dict()
        db.add(AnalysisResult(
            job_id=job_id,
            result_json={
                "match_id": result.match_id,
                "frames_processed": result.frames_processed,
                "players_tracked": result.players_tracked,
                "events_detected": result.events_detected,
                "player_metrics_written": result.player_metrics_written,
                "team_metrics_written": result.team_metrics_written,
                "homography_confidence": result.homography_confidence,
                "calibration_valid_fraction": result.calibration_valid_fraction,
                "overlay_render": result.overlay_render,
                "pipeline_latency": latency_report,
            },
            summary=(
                f"Processed {result.frames_processed} frames, tracked {result.players_tracked} "
                f"players, detected {result.events_detected} events. Homography confidence: "
                f"{result.homography_confidence:.2f}."
            ),
        ))

        invalidate_match_cache(result.match_id)

        job.status = ProcessingStatus.completed
        job.progress = 100
        job.message = "Analysis complete."
        job.error = None
        job.completed_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        db.commit()

        return {"job_id": job_id, "status": "completed", "match_id": result.match_id,
                "latency_total_seconds": latency_report["total_seconds"]}

    except Exception as error:
        job = db.get(ProcessingJob, job_id)
        if job is not None:
            job.status = ProcessingStatus.failed
            job.error = str(error)
            job.updated_at = datetime.utcnow()
            job.completed_at = datetime.utcnow()
            db.commit()
        raise

    finally:
        db.close()
