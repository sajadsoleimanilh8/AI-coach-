# CLAUDE.md

Guidance for AI coding agents working in this repository. Humans: start with
README.md and RUN.md.

## Non-negotiables

- **Never delete or relax `conftest.py` (repo root) or
  `backend/tests/test_database_isolation.py`.** A green `pytest` run once
  deleted real `Match`/`Video`/`ProcessingJob`/`PlayerMetric` rows. The root
  conftest points `DATABASE_URL` at a temp SQLite file at import time and
  aborts with exit code 3 if that fails. If you see exit code 3, fix the import
  order that caused it, never the guard.
- **Never loosen `.gitignore`.** It keeps ~270 MB of `.pt` checkpoints, videos,
  datasets and `runs/` out of git, and each rule is commented.
- **Do not re-enable the `python` tool by default** (`nexus/config/nexus.yaml`).
  It executes model-written code with no sandbox.
- Model and dataset paths come only from `configs/registry.py` +
  `configs/models.yaml` / `configs/datasets.yaml`. Do not hardcode a `.pt` path.

## Environment

- Python **3.11**, not 3.12/3.13 (`mediapipe==0.10.14` pin).
- The virtualenv is `venv/` (not `.venv/`). Commands use
  `venv\Scripts\python.exe -m ...` explicitly.
- The project is installed editable: `pip install -e ".[dev,cv]"`. Imports are
  absolute from the repo root (`from ai.computer_vision...`). Do **not** add
  `sys.path.insert` hacks; there are none left.
- `ai/` is an implicit namespace package (no top-level `__init__.py`).
- Windows Celery worker needs `--pool=solo`.
- Use `127.0.0.1`, not `localhost`, for local URLs. Vite and both APIs bind
  IPv4, and `localhost` can resolve to `::1` first.
- The overlay renderer tries `avc1` (.mp4) and falls back to `VP80` (.webm),
  depending on which codec the local OpenCV build can write.
- The working checkout is `D:\ssc`. Datasets, `football_v1/` and training
  `runs/` are in `D:\ssc-data\`; the extracted Docker and Ollama installers
  are in `D:	ools\`. Nothing outside `D:\ssc` is part of the repo.

## Commands

| Task | Command |
|---|---|
| Backend API | `venv\Scripts\python.exe -m uvicorn backend.api.main:app --port 8000` |
| NEXUS API | `venv\Scripts\python.exe -m uvicorn nexus.api.main:app --port 8100` |
| Worker | `venv\Scripts\python.exe -m celery -A backend.celery_app worker --pool=solo` |
| Frontend | `cd frontend/web && npm run dev` |
| Python tests | `venv\Scripts\python.exe -m pytest -q` (~2.5 min) |
| Python lint | `venv\Scripts\python.exe -m ruff check .` |
| Types (report only) | `venv\Scripts\python.exe -m mypy backend nexus configs ai` |
| Frontend lint / build | `npm run lint` / `npm run build` in `frontend/web` |
| Frontend unit tests | `npm run test:unit` in `frontend/web` (Vitest, <1 s) |
| Browser tests | `npm test` in `frontend/web` (needs both APIs running and a processed match) |
| Pipeline regression | `pytest tests/test_pipeline_e2e.py` (~25 s; needs `samples/` + checkpoints) |

## Security config

- `SSC_API_KEY` (backend) and `NEXUS_API_KEY` (NEXUS): unset means auth off.
  When set, send `X-API-Key`. `/health` and `/api/health` stay public. The two
  GET video routes also accept `?api_key=` because `<video src>` cannot send
  headers.
- `MAX_UPLOAD_BYTES` caps uploads (default 2 GiB).
- See `backend/.env.example`, `nexus/.env.example`, `frontend/web/.env.example`.

## Layout notes

- **The pipeline is split by stage.** `backend/pipeline/runner.py` is the
  orchestrator only; stage code lives in `detection.py`, `calibration.py`,
  `trajectories.py`, `events.py`, `team_scoring.py`, `player_scoring.py`,
  `persistence.py`, `results.py`. `runner.py` re-exports them, so
  `runner._some_stage` still resolves. Patch a constant on the module that
  owns it (e.g. `calibration.CALIBRATION_DIR`), not on `runner`.
- **NEXUS services** are built in `nexus/api/services.py` and reached with
  `get_services(request)`. Tests override with
  `app.state.services.router = fake`.
- **Scorers** return `ai/common/metrics.py::metric_result(...)`, and share
  `clamp` / `scale_1_to_10` / `invert` from `ai/common/scoring.py`.
- **Schema** is owned by the models + Alembic (`alembic upgrade head`).
  Startup `create_all()` runs for SQLite only.
- `ai/` contains only implemented modules; `ai/README.md` records the
  blueprint slots that have no code.

## Known state (2026-09-20)

- `ruff check .` passes; coverage 87.3% overall (1,481 tests), pipeline
  stages 86%.
- **`tests/test_pipeline_e2e.py` fingerprints the real pipeline's output.** If
  it fails after a change you did NOT intend to alter results, you changed
  behaviour. If you did intend it, re-run with `SSC_UPDATE_PIPELINE_GOLDEN=1`
  and commit the new `tests/golden/pipeline_sample_15s.json` with that change.
- Import test helpers in `tests/` as sibling modules (`from pipeline_snapshot
  import ...`): ultralytics installs a top-level `tests` package that shadows
  `tests.*`.
- mypy baseline: 346 errors (CI reports but does not gate). Roughly 239 of
  those are in test fakes; app code is at 107.
- ESLint: 0 errors, 0 warnings. The React Compiler rules that had been
  downgraded (`set-state-in-effect`, `refs`) are back at error level -- the
  hooks now adjust state during render instead of from an effect.
- Redis is optional. When it is not running the cache trips a circuit breaker
  after a 250 ms timeout and skips it for 30 s, so a missing Redis costs
  milliseconds, not seconds. `REDIS_URL` must be `127.0.0.1`, not `localhost`.
- Pipeline memory is NOT a blocker: measured on a real clip, every whole-video
  structure together is ~1.9 MB per 15 s, about 0.7 GB for a 90-minute match.
  Streaming would require changing global algorithms (re-id merge, team
  clustering), so it was deliberately not done.
- Both Docker images build and run non-root with working healthchecks
  (backend 3.6 GB with CPU torch, nexus 487 MB).
