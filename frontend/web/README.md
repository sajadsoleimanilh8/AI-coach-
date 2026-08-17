# Frontend — SportsStrategyCoachAI

A single-page app. `index.html` is the **only** HTML file: the marketing shell
*is* the app shell. The functional console mounts into `#dashboard-root` inside
that same document and the two views swap by flipping `body[data-view]` — no
router, no navigation, no second page.

## Run

```bash
cd frontend/web
npm install
npm run dev          # http://localhost:3000
```

The port is pinned with `strictPort`. If 3000 is taken, Vite **fails to start**
rather than sliding to 3001 — because the backend's `CORS_ALLOWED_ORIGINS`
defaults to `http://localhost:3000`, and a silent port change turns every API
call into an unexplained `Failed to fetch`.

`npm run build` emits to `dist/`; `npm run preview` serves that build.

## Running the whole stack

Five processes. Only the first three are needed for the dashboard to work at
all; Redis + worker are what make a *new* upload process automatically, and
NEXUS is only for Coach Chat.

```bash
# 1. Redis — Celery broker and the metrics cache
docker run -d --name ssc-redis -p 6379:6379 --restart unless-stopped redis:7-alpine

# 2. Celery worker — picks up uploads.  --pool=solo is REQUIRED on Windows.
venv/Scripts/python -m celery -A backend.celery_app worker --loglevel=info --pool=solo

# 3. Core backend
venv/Scripts/python -m uvicorn backend.api.main:app --port 8000

# 4. NEXUS (optional, Coach Chat only)
venv/Scripts/python -m uvicorn nexus.api.main:app --port 8100

# 5. Frontend
cd frontend/web && npm run dev
```

**Use `venv/`, not `.venv/`.** Only `venv/` has `ultralytics` + `torch`. Running
the backend from `.venv/` makes the Calibration tab return 500 on every frame
with `ModuleNotFoundError: No module named 'ultralytics'`.

Check everything at once:

```bash
venv/Scripts/python -m scripts.dev_environment          # per-tab readiness report
venv/Scripts/python -m scripts.dev_environment --seed   # + seed the questionnaire tabs
```

If Redis was down when you uploaded, the file is still saved — only the queue
hand-off failed. Run that job without a broker:

```bash
venv/Scripts/python -m scripts.run_job_local --latest
```

## Services it talks to

| Service | Default | Used by | Required? |
|---|---|---|---|
| Core backend (FastAPI) | `http://localhost:8000` | Tabs 1–7 | Yes |
| NEXUS (LLM coach) | `http://localhost:8100` | Coach Chat only | No |

The two have separate clients (`src/api/client.js`, `src/api/nexus.js`) with
separate "unreachable" messages, because they start and stop independently —
NEXUS being down says nothing about the core backend and vice versa.

Override with `VITE_API_BASE_URL` / `VITE_NEXUS_BASE_URL` in a `.env` file here.

### CORS

Running the dev server on a non-default port means adding that origin:

```dotenv
# backend/.env
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://localhost:<your-port>
NEXUS_CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:<your-port>
```

## Layout

```
index.html                  the shell + design tokens + GSAP + the view toggle
src/
  main.jsx                  createRoot(#dashboard-root)
  api/client.js             core backend  (AbortSignal on every call)
  api/nexus.js              NEXUS + SSE streaming chat
  hooks/useAsync.js         useAsync (load) / useAction (submit), both abortable
  lib/motion.js             bridge to window.SSCMotion — no second animation lib
  styles/dashboard.css      built entirely on index.html's :root tokens
  components/
    DashboardApp.jsx        tab shell; owns the shared matchId
    ui/                     Primitives / States / Form / Pitch
    tabs/                   the eight tabs
```

## Video playback: processed vs original

The player defaults to the **annotated render** — the clip with the pipeline's
own boxes, track ids and team colours burned in — and falls back to the raw
upload when no render exists yet. `processed_video_exists` on the match
summary decides which, re-asked on `dataEpoch` so it flips the moment a run
finishes.

| Source | Endpoint | Exists |
|---|---|---|
| Processed | `GET /api/videos/{id}/processed` | after a completed run |
| Original | `GET /api/videos/{id}/file` | the instant the upload lands |

The live SVG overlay is suppressed on the processed clip — its boxes are
already in those pixels, and drawing them twice would invite reading the two
as independent confirmations of each other.

It is WebM/VP8, not MP4/H.264. This machine's OpenCV ships an FFmpeg that
advertises H.264 but cannot initialise libopenh264, and `mp4v` fails in the
browser with `MEDIA_ERR_SRC_NOT_SUPPORTED` (verified). VP8 both encodes here
and plays there. `OVERLAY_MAX_WIDTH` (default 960) caps the render;
`RENDER_OVERLAY_VIDEO=0` skips the stage entirely.

## Heatmap coordinate spaces

Two, and they are not interchangeable:

- **pitch** (default) — metres on a 105×68 model, via a validated homography.
  Comparable across matches. Empty on footage whose calibration never solved,
  which is currently all of it.
- **image** — where the player appeared in the video *frame*. Needs no
  calibration, so it actually returns cells, but it moves with the camera,
  carries no distance in metres, and cannot be compared across matches.

The view renders the space the **response** reports, never the one it asked
for, and in image space it drops the pitch markings and the "105m × 68m"
legend. A grid of frame pixels labelled in metres is the one claim this
screen must never make.

## Two conventions worth keeping

**A missing number is words, never a zero.** `null` from the backend renders as
"Not Yet Computed" / "Not Available", never `0` or `--`. A readiness score of
zero and a score that was never computed look identical as "0", and only one of
them is a real reading.

**Provenance travels with the value.** Where the backend labels its own output —
`method: heuristic_proxy`, `source: measured`, `is_reinforcement_learning: false`,
`confidence`, `calibration_valid` — that label is rendered next to the number,
not dropped on the way to the screen.

## Ambient backdrop

`src/components/ambient/StreamingBackdrop.jsx` is a decorative word-by-word
reveal behind the landing hero, adapted from a `streaming-text` component. It
mounts into `#ambient-root` (inside `#hero`) from its own `createRoot`, and it
is `aria-hidden` + `pointer-events: none`.

Its citations, action buttons, and follow-up prompts were deliberately dropped,
and its copy is capability description rather than measurement. All three imply
the text is a real answer with real provenance — on a decorative layer it is
not, and an ambient layer is the worst place to put a number a reader could
mistake for a reading.

It pauses when `body[data-view="dashboard"]` is set or the tab is hidden.

## This project is not shadcn / Tailwind / TypeScript

It is Vite + React 19 + **plain JSX**, styled by hand-written CSS on the tokens
declared in `index.html`. Components live in `src/components/` (`ui/` for
primitives, `tabs/` for views, `ambient/` for decoration).

That is a deliberate choice, not an oversight: the design system is fixed by
`index.html`, and a utility-class framework alongside it would mean two styling
systems in one bundle for no gain. Drop-in `.tsx` components therefore need
porting, not pasting — and any component carrying its own `--surface` / `--ink`
light palette needs restyling onto the existing tokens first.

If you do want the shadcn stack later:

```bash
cd frontend/web

# TypeScript — allowJs keeps the 18 existing .jsx/.js files compiling
npm i -D typescript @types/react @types/react-dom
npx tsc --init --jsx react-jsx --allowJs --moduleResolution bundler

# Tailwind 4
npm i -D tailwindcss @tailwindcss/vite
# then add tailwindcss() to plugins in vite.config.js
# and `@import "tailwindcss";` at the top of a stylesheet

# shadcn — writes components.json and its own tokens
npx shadcn@latest init
```

`shadcn init` expects a `@/` alias. Add it to `vite.config.js`:

```js
import path from 'node:path';
// ...
resolve: { alias: { '@': path.resolve(__dirname, './src') } }
```

Note that `shadcn init` will write **its own** `:root` palette into your CSS.
Keep it namespaced or scoped to a wrapper, or it will sit next to the neon-dark
tokens this app is built on and the two will drift.

## Shared state

`matchId` lives in `DashboardApp`. Match-scoped tabs (Player, Team,
Calibration, Simulation) read it from props and show the match picker while it
is null. No tab derives a match id of its own.

It is set three ways, all of which route through the backend rather than
guessing: an upload in Match Analysis, the picker on any match-scoped tab
(`GET /api/matches` to list, `GET /api/matches/{id}` to resolve), or a
`?match=` link.

### `?match=` in the URL

The selection is mirrored into the query string, so it survives a refresh and
a tab is linkable with its match:

```
http://localhost:3000/?match=<match_id>#dashboard/simulation
```

Only the **id** is stored there. `video_id` / `job_id` are re-resolved from
`GET /api/matches/{id}` on load — a hand-edited or stale link must not be able
to pair one match's clip with another match's tracking rows. An id that no
longer resolves is dropped from the address bar rather than left sitting there
looking like a live selection.

Written with `replaceState`, not `pushState`: selecting a match is not a
navigation, and pushing would make Back undo the selection instead of
returning to the previous tab.
