# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Security
- The NEXUS `python` tool is now **disabled by default**. It runs model-written
  code with no sandbox, and together with the open `/api/chat` endpoint that
  allowed remote code execution. Opt in with `NEXUS_ENABLE_PYTHON_TOOL=1`.
- Optional API-key auth on both services: `SSC_API_KEY` and `NEXUS_API_KEY`,
  sent as the `X-API-Key` header. It is off when unset, so local dev does not
  change. Health endpoints stay public. The dashboard sends the key via
  `VITE_SSC_API_KEY` / `VITE_NEXUS_API_KEY`.
- Video uploads are capped by `MAX_UPLOAD_BYTES` (default 2 GiB), and a
  rejected upload leaves no partial file behind. `application/octet-stream` is
  accepted only when the filename has a video extension.
- CORS now lists allowed methods and headers explicitly instead of `*`.

### Fixed
- `GET /api/tactical/formation` returned 404 and `team_shape` returned `[]` for
  every processed match. Both required `team_id=team-home`, but the pipeline
  writes whichever team labels clustering produced (often `unassigned`). The
  parameter is now optional: formation picks the best-ranked team present and
  `team_shape` returns every team's rows.
- Each Redis call blocked ~4 seconds per attempt when no server was running,
  about 8 s on a cached endpoint, because the client used the library's default
  timeouts. Connections now time out in 250 ms and a tripped circuit skips
  Redis entirely for 30 s, logging once per outage rather than per request. The
  tactical test module went from 57 s to 1.6 s.
- The `REDIS_URL` default was `localhost`, which resolves to IPv6 `::1` first on
  Windows and waits for that to fail; it is now `127.0.0.1`.
- The tracking overlay re-requested its window on every render while the
  playhead sat in the last 40 frames of one: the auto-pager stepped the window
  past the playhead and the backward-seek rule pulled it straight back. A run
  against the old code made 6,083 requests in 3 seconds and left the backend
  unresponsive; the same clip now makes 5. The paging rule is one pure function
  (`matchAnalysis/paging.js`) with tests that assert it settles.
- `transformers` 5 removed the `torch_dtype` alias, and `SFTTrainer` honours
  `max_length` only when its args are an `SFTConfig`, so `max_seq_length`
  reached the VRAM planner and nothing else -- the "1024 -> 512" OOM remedy it
  prints was a no-op.
- `VectorStore.init()` was declared on two implementations but not on the ABC.
- `backend/Dockerfile` did not copy `configs/`, so the image failed on
  `from configs import registry` when the first job ran.
- `nexus/Dockerfile` did not copy `ai/`, so that container exited at startup
  with `ModuleNotFoundError: No module named 'ai'` — it had not been able to
  start since `nexus/sports/game_plan.py` began importing
  `ai.opponent_intelligence`. `tests/test_dockerfile_copies.py` now fails if
  any service imports a first-party package its image does not copy.
- The backend image pulled the CUDA build of torch on Linux (several GB of
  `nvidia-*` wheels, no GPU in that image); it now installs CPU torch
  explicitly.
- Restored the comments and docstrings that `8e3a993` stripped from ~190
  Python files and 24 frontend files, keeping every code change made since.
  History-log notes ("FIXED (Phase 3 audit): …", "Day 9 & Day 10") were
  rewritten as present-tense rationale, and several that described removed
  code were deleted.
- A comment above `GET /api/tactical/formation` claimed the endpoint returns
  `value=None`; it raises 404. Corrected.
- `test_dry_run_imports_no_ml_package` failed or passed depending on test
  order. It now checks in a fresh subprocess. The full suite passes: 1,330 tests.

### Changed
- `backend/database/models.py` uses SQLAlchemy 2.0 typed declarations
  (`DeclarativeBase`, `Mapped[...]`, `mapped_column`) for all 157 columns, each
  with an explicit `nullable=` so the annotation cannot change the schema. The
  migration-drift test and the pipeline fingerprint both confirm the schema is
  unchanged. mypy: 498 errors to 346.
- The deprecated `@app.on_event("startup")` hook is now a `lifespan` context
  manager.
- The dashboard's data-loading hooks adjust state during render instead of in
  an effect, so a tab shows its loading state on the first render rather than
  one stale frame of the previous match's data. The two ESLint rules that had
  been downgraded to warnings for this are back at error level: 22 warnings to
  0.
- Coverage 79.2% to 87.3% (1,481 tests), covering first-touch scoring, manual
  calibration, the phased training splits and the psychology loader.
- `backend/pipeline/runner.py` (1,968 lines) is now the orchestrator only.
  Stage code moved verbatim into `detection.py`, `calibration.py`,
  `trajectories.py`, `events.py`, `team_scoring.py`, `player_scoring.py`,
  `persistence.py` and `results.py`; `runner.py` re-exports them. Verified by
  running the real pipeline on a sample clip before and after: every tracking
  point, event, metric and calibration row is byte-identical.
- NEXUS built ~40 components inline in `lifespan()` and stored them as untyped
  `app.state` attributes. They now come from `nexus/api/services.py`
  (`build_services()` / `NexusServices` / `get_services()`), so mypy checks
  every route's access. `nexus/api/main.py`: ~400 lines to 92.
- `TabMatchAnalysis.jsx` (1,475 lines) split into `matchAnalysis/*`; the tab
  file is now 91 lines. A crash in a tab no longer takes down the dashboard:
  `ui/ErrorBoundary.jsx` contains it and the other tabs stay reachable.
- The metric envelope, previously hand-built in 37 places, comes from
  `ai/common/metrics.py::metric_result()`. `clamp` / `scale_1_to_10` /
  `invert` are defined once in `ai/common/scoring.py` (the two copies had
  already drifted).
- `method` and `confidence` are `Literal` types, so a typo like
  `confidence="lowsample"` is now a type error rather than a value that
  reaches the database.
- Five near-identical YOLO trainers collapsed into `training/cli.py`.
  `python -m training.train <model>` is the entry point; the per-model
  modules remain as wrappers because in-progress phased runs print them.
- The schema is owned by the models plus Alembic. The stale
  `backend/database/schema.sql` (7 tables, while the models define 15) and the
  unrunnable `migrations/001_calibration_status.sql` were removed, and startup
  `create_all()` now runs for SQLite only.
- GSAP is vendored in `frontend/web/public/vendor/gsap` (byte-identical to the
  cdnjs files it replaces), and the ~1,000-line inline `<style>` block moved to
  `src/styles/landing.css`. `index.html`: 1,966 lines to 949.
- `test_homography.py` was one script-style `run()` with 26 assertions behind a
  single pytest wrapper, so the first failure hid the rest. It is now 13
  independent tests; the assertions are unchanged.
- The `_memory_db_env` fixture, copy-pasted into 15 NEXUS test modules in five
  variants, lives once in `nexus/tests/conftest.py`.
- `RUN.md` pointed at the old repository URL; the repo is now
  `sajadsoleimanilh8/AI-coach-`. Three superseded design docs carry a banner
  pointing to `docs/database_schema.md`, and `docs/pipeline_architecture.md`
  covers the September additions.
- `ai/` no longer contains 58 empty `.gitkeep` directories; `ai/README.md`
  records which blueprint modules have no code.
- Docker images are multi-stage, run as a non-root user, and declare
  healthchecks. A `.dockerignore` keeps `venv/`, datasets, weights and sample
  videos out of the build context.

### Added
- `tests/test_pipeline_e2e.py`: runs the real pipeline on `samples/sample_15s.mp4`
  and compares every output row against a recorded fingerprint
  (`tests/golden/pipeline_sample_15s.json`). Pipeline-stage coverage went from
  roughly 12-27% to 86%. Skips where the clip or checkpoints are absent (CI).
- Frontend unit tests (Vitest, `npm run test:unit`, 33 tests) for the API
  client and `useAsync`/`useAction`, including a regression test for the
  StrictMode bug where every submit hung forever. Wired into CI.
- Coverage measurement (`pytest --cov`); baseline 79.2% of 14,730 statements.
- `pyproject.toml`: the project can be installed with `pip install -e ".[dev,cv]"`.
- Ruff (passes clean), a mypy baseline, and ESLint + Prettier for the frontend.
- GitHub Actions CI: ruff, pytest with coverage, frontend lint and build, and
  mypy as a report.
- `CLAUDE.md`, `CONTRIBUTING.md`, this changelog, `nexus/.env.example`,
  `frontend/web/.env.example`, and `requirements-lock-windows.txt`.

### Removed
- All `sys.path.insert` import hacks in tracked code (production modules,
  tests, scripts, and trainers). Imports are now absolute.
