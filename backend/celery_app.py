import os

from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "sports_strategy_coach_ai",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["backend.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    # By default Celery retries connecting to the broker indefinitely
    # (broker_connection_retry / task_publish_retry), which is exactly what
    # made process_video_job.delay() hang the upload request for a very
    # long time with Redis down instead of raising quickly. Cap it hard:
    # one attempt, no retry, short timeout — so main.py's try/except around
    # .delay() actually fires within ~2s instead of never.
    broker_connection_retry_on_startup=False,
    broker_connection_retry=False,
    broker_connection_timeout=2,
    task_publish_retry=False,
    # Nothing in this codebase calls AsyncResult.get() -- backend/tasks.py
    # writes status/progress straight to Postgres, which is what the API
    # polls. Without this, every .delay() call still opens a Redis pubsub
    # connection for result tracking that nobody reads, and that pubsub
    # connection has its OWN ~19s retry-limit timeout independent of the
    # broker_connection_* settings above -- that's what was actually
    # causing the long hang when Redis was down, not the broker publish.
    task_ignore_result=True,
)