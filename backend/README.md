# SportsStrategyCoachAI Backend

FastAPI backend for video upload, processing status, database storage, and JSON analysis results.

## Database

Tables are defined in SQLAlchemy models and documented in `backend/database/schema.sql`.

- `videos`: uploaded video file records
- `processing_jobs`: queue/status/progress for each video
- `analysis_results`: JSON output from computer vision or tactical analysis

## Run locally

Four separate processes, started in this order -- each one depends on the
one before it being up already (starting the celery worker before Redis
exists will fail its broker connection; the frontend needs the backend
listening to avoid "Failed to fetch"). `backend/.env.example` documents
every variable read by `os.getenv(...)` below, but nothing in this repo
loads a `.env` file automatically (no `python-dotenv`) -- treat it as a
reference and set real environment variables in each shell you use
(`$env:NAME = "value"`, as shown in step 2 below), not as a file you can
just drop in and have picked up.

### 1. Redis

Required by `backend/celery_app.py`'s `REDIS_URL` -- there's no fallback
if it isn't reachable; `.delay()` calls fail fast instead of hanging (see
that file's own comment), and `backend/api/cache.py`'s metrics cache also
needs it.

```powershell
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

(Already running via `docker compose up redis` if you're using the
Docker Compose stack instead -- see "Run with Docker" below.)

### 2. Backend API

```powershell
$env:DATABASE_URL = "sqlite:///backend/database/sports_strategy.db"
pip install -r backend/requirements-full.txt
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

`*.pt` checkpoint files are gitignored, so a fresh checkout has none. Train
them (`python -m training.train_player`, and one per model) or point
`SSC_MODEL_ROOT` at a directory that has them before uploading a video.
Without them the job fails with a clean `PipelineAssetError` naming exactly
what's missing -- that's correct, expected behaviour, not a bug to work
around here.

Use `backend/requirements-full.txt`, not `backend/requirements.txt`, for
this step -- the plain `requirements.txt` is missing numpy/opencv-python/
scipy/mediapipe/ultralytics, which `backend.api.main` needs transitively
(via `backend.tasks` -> `backend.pipeline.runner` -> the `ai.*` modules)
just to import, let alone run a pipeline job. Installing only
`requirements.txt` means `uvicorn` never starts at all --
`ModuleNotFoundError: No module named 'numpy'` on the first request, not
a network/CORS issue despite what the frontend's error looks like.
`ultralytics` pulls in `torch`/`torchvision` as transitive deps -- expect
this install to be multi-GB and take several minutes; that's expected,
not a hang.

The local default database is SQLite:

```text
backend/database/sports_strategy.db
```

### 3. Celery worker

Must be started separately, or every uploaded job sits at `status:
queued` forever with no error surfaced anywhere -- `POST
/api/videos/upload` succeeds and queues the job regardless of whether
anything is listening to actually run it.

```powershell
celery -A backend.celery_app worker --pool=solo --loglevel=info
```

`--pool=solo` is required on Windows: Celery's default `prefork` pool
uses `os.fork()`, which doesn't exist on Windows. Without this flag the
worker either fails to start or silently never picks up tasks (depends on
Celery version) -- either way, nothing points back to "wrong pool for
this OS," so it just looks like uploads are permanently stuck.

### 4. Frontend

```powershell
cd frontend/web
npm install
npm run dev
```

Set `VITE_API_BASE_URL` (and `VITE_NEXUS_BASE_URL`) if those services
aren't on `http://localhost:8000` / `http://localhost:8100`.

## Run with Docker

```powershell
docker compose up --build
```

This starts:

- FastAPI backend on `http://localhost:8000`
- PostgreSQL on `localhost:5432`
- Redis on `localhost:6379`
- A Celery worker (Linux container, so no `--pool=solo` needed here --
  that flag is a Windows-host-only workaround, see step 3 above)

`docker-compose.yml` bind-mounts the five trained checkpoints into both the
`backend` and `celery-worker` services under `SSC_MODEL_ROOT=/app/models`.
Those `*.pt` files are gitignored, so a fresh clone does not have them and
`docker compose up` fails at the mount -- deliberately, rather than starting
and then failing on the first upload. Train them, or drop those mounts to run
the API without inference.

## API examples

Upload a video:

```powershell
curl.exe -X POST "http://localhost:8000/api/videos/upload" `
  -F "file=@samples/test_clip_270_570.mp4" `
  -F "metadata={\"team\":\"home\",\"opponent\":\"away\",\"match_date\":\"2026-07-20\"}"
```

Check processing status:

```powershell
curl.exe "http://localhost:8000/api/processing/<job_id>"
```

Update processing status:

```powershell
curl.exe -X PATCH "http://localhost:8000/api/processing/<job_id>" `
  -H "Content-Type: application/json" `
  -d "{\"status\":\"processing\",\"progress\":45,\"message\":\"Detecting players and ball\"}"
```

Save analysis JSON:

```powershell
curl.exe -X POST "http://localhost:8000/api/processing/<job_id>/result" `
  -H "Content-Type: application/json" `
  -d "{\"summary\":\"High press created weak-side space.\",\"result\":{\"players_detected\":22,\"ball_tracks\":184,\"tactical_notes\":[\"pressing intensity dropped after minute 60\"]}}"
```

Read analysis JSON:

```powershell
curl.exe "http://localhost:8000/api/processing/<job_id>/result"
```
