"""
Run a queued/failed processing job in THIS process, with no Celery broker.
"""

from __future__ import annotations

import argparse
import sys
import time

from backend.database.models import ProcessingJob
from backend.database.session import SessionLocal
from backend.tasks import process_video_job


def latest_job_id() -> str | None:
    db = SessionLocal()
    try:
        job = db.query(ProcessingJob).order_by(ProcessingJob.created_at.desc()).first()
        return job.id if job else None
    finally:
        db.close()


def describe(job_id: str) -> None:
    db = SessionLocal()
    try:
        job = db.get(ProcessingJob, job_id)
        if not job:
            print(f"  no job with id={job_id}")
            return
        print(f"  job     {job.id}")
        print(f"  video   {job.video_id}")
        print(f"  match   {job.video.match_id if job.video else '(unlinked)'}")
        print(f"  status  {job.status.value}  progress={job.progress}")
        if job.error:
            print(f"  error   {job.error}")
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", nargs="?", help="job to run")
    parser.add_argument("--latest", action="store_true", help="run the most recent job")
    args = parser.parse_args()

    job_id = args.job_id or (latest_job_id() if args.latest else None)
    if not job_id:
        print("Give a job_id, or --latest. Nothing to do.")
        return 2

    print("BEFORE")
    describe(job_id)

    print(f"\nRunning pipeline locally for job_id={job_id} (no broker) ...")
    print("This is the real pipeline: model loading and per-frame inference on CPU")
    print("take minutes, not seconds. Progress updates land in the DB as it goes.\n")

    started = time.time()
    result = process_video_job.apply(args=[job_id])
    elapsed = time.time() - started

    print(f"\nFinished in {elapsed:.1f}s")
    if result.failed():
        print("  the task raised:")
        print(f"  {result.traceback}")
    else:
        print(f"  returned: {result.result}")

    print("\nAFTER")
    describe(job_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
