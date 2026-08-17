# Verification Guide

Requirements: Python 3.11+ and Node 18+.

Steps 1–2 run on a plain CPU laptop in a couple of minutes: no GPU, no Redis,
no Postgres, no Docker, no video file, and no model checkpoints. Step 3 shows
the UI. Step 4 processes a real clip and does need the CV stack.

---

## 1. Backend + AI test suite

```bash
pip install -r backend/requirements.txt
pytest
```

Expected: **1223 passed**.

`pytest` with no arguments, from the repo root, is now the correct command.
Two things previously made it wrong, both fixed:

- **It reported 54 errors.** `nexus/pytest.ini` sets `asyncio_mode = auto`, but
  pytest reads one config from one rootdir — running from the root meant that
  file was never read and every async test under `nexus/` errored at setup.
  There is now a root `pytest.ini`.
- **It silently deleted your database.** The fixtures in `backend/tests` issue
  unqualified `DELETE`s (`query(Match).delete()` and friends), which are safe
  only against a throwaway database. `backend/tests/conftest.py` redirects
  `DATABASE_URL` to one — but from the repo root that conftest loads *after*
  something under `ai/` has already imported `backend.database.session`, so the
  redirection was dead code and its `pytest_sessionstart` guard never ran.
  Measured by probing the bound URL from inside a backend test:

  ```
  pytest                 -> sqlite:///.../backend/database/sports_strategy.db   # real
  pytest backend/tests   -> sqlite:///<temp>/test_suite.db                      # safe
  ```

  A green suite destroyed real `Match` / `Video` / `ProcessingJob` /
  `PlayerMetric` rows. The repo-root `conftest.py` now performs the
  redirection where pytest is guaranteed to load it first, and
  `backend/tests/test_database_isolation.py` fails loudly if that ever breaks
  again. **Do not delete either file.**

Worth reading on its own:

```bash
pytest backend/tests/test_video_upload.py -v      # broker-down upload path
pytest backend/tests/test_overlay_video.py -v     # annotated render + codec chain
pytest backend/tests/test_events_api.py -v        # events endpoint + space contract
pytest ai/computer_vision/tactical_analysis/tests/test_image_space_possession.py -v
```

---

## 2. Frontend build

```bash
cd frontend/web
npm install
npm run build
```

---

## 3. Run the dashboard

```bash
pip install -r backend/requirements-full.txt   # adds numpy/opencv/ultralytics
python -m uvicorn backend.api.main:app --port 8000
```

```bash
cd frontend/web && npm run dev
```

Open <http://localhost:3000>.

- **Port 3000 is deliberate.** Both services default their CORS allowlist to
  `localhost:3000` (`backend/api/main.py::CORS_ALLOWED_ORIGINS`,
  `nexus/api/main.py::NEXUS_CORS_ALLOWED_ORIGINS`) and `vite.config.js` pins
  the port. Serving on another port needs that origin added to
  `CORS_ALLOWED_ORIGINS`.
- The **Coach Chat** tab additionally needs NEXUS on port 8100
  (`python -m uvicorn nexus.api.main:app --port 8100`). Every other tab works
  without it.

### Browser tests

These drive a real Chromium against a running frontend. The third needs a
backend holding at least one completed match with a processed render, and
exits 2 (skipped) rather than passing vacuously if there is none.

```bash
cd frontend/web
node tests/navigation.spec.mjs        # 33 assertions
node tests/match-selection.spec.mjs   # 27 assertions
node tests/match-analysis.spec.mjs    # 23 assertions
```

`match-analysis.spec.mjs` is the one that covers the video pipeline's output
end to end: that the player loads the **processed** render rather than the raw
upload, that the browser actually decodes it, that the tactical overlay shares
the video's box and uses the **source** frame's coordinate system, that it
stays locked to `currentTime` across play / pause / backward seek / 2x
playback / crossing a window boundary, and that the events timeline seeks the
video when clicked.

---

## 4. Process a real clip

Needs the CV stack and the five trained checkpoints under `models/`. A GPU is
optional; the pipeline probes it and falls back to CPU (see
`backend/pipeline/device.py`).

```bash
# Upload through the dashboard, or POST /api/videos/upload, then:
python -m scripts.run_job_local --latest
```

`run_job_local` executes the same task body a Celery worker would, in-process,
with no broker. Progress lands in the `ProcessingJob` row as it goes, so the
dashboard's poller tracks it live.

Measured on an RTX 5070 Ti Laptop, `samples/test_clip_270_570.mp4` (300 frames,
1280×720, 29.97 fps): **26.5 s total**, of which detection+tracking 5.2 s,
frame synchronisation 7.8 s, pose 9.8 s, annotated render 2.5 s.

A 15-second sample is at `samples/sample_15s.mp4`. Note that it is a close-up
shot: the ball model fires on 10 of its 375 frames and no player is ever
within the control radius of the ball, so it correctly produces **zero**
events. `samples/test_clip_270_570.mp4` is a wide broadcast shot and produces 8.
