# Pipeline architecture — what actually exists today

_Audit date: 2026-08-12. Every edge below was verified by reading the code,
not inferred from directory names. Edges that do **not** exist are marked
**MISSING** rather than described aspirationally._

## Reading this document

| Mark | Meaning |
|---|---|
| ✅ | Implemented and wired into the production path (`backend/pipeline/runner.py`) |

> **Module layout (2026-09-17).** `backend/pipeline/runner.py` was 1,968 lines
> and is now just the orchestrator: `run_pipeline()` plus the stage order. The
> stage code moved, unchanged, into siblings — `detection.py`, `calibration.py`,
> `trajectories.py`, `events.py`, `team_scoring.py`, `player_scoring.py`,
> `persistence.py`, `results.py`. Older references below that read
> `runner.py::_some_stage()` mean the function in its new module; `runner.py`
> re-exports them, so existing imports still resolve.

## Added since the 2026-08-12 audit

The audit below predates the September work. Same status marks as the table
that follows.

| Component | Status | What it is |
|---|---|---|
| Re-identification merge (`ai/computer_vision/player_tracking/reid_merge.py`) | ✅ | Appearance-based re-association of tracks ByteTrack dropped during occlusions, so one player is not scored as several half-players. Wired as pipeline stage `reid_merge`; disable with `SSC_REID_MERGE=0` to measure without it. |
| Opponent weakness map (`ai/opponent_intelligence/weakness_map/`) | ✅ | Ranks named weaknesses and strengths from already-measured metrics. Never invents a number; an unmeasured metric is an absence, not a weakness. |
| Counter-strategy generator (`ai/opponent_intelligence/counter_strategy_generator/`) | ✅ | Maps each weakness to a coachable adjustment. The trigger is data; the phrasing is a fixed playbook. |
| Deterministic game plan (`nexus/sports/game_plan.py`) | ✅ | Builds the coach report's game-plan section from the two modules above with no LLM call; the model only narrates what is already decided. |
| Narrative guard (`nexus/sports/narrative_guard.py`) | ✅ | Reads the generated narrative back and flags chronology or event claims the measured context cannot support (e.g. "in the first half…" on a 15-second clip). The coach then regenerates once; if the retry is still not clean it keeps the attempt with fewer such claims and returns it with those warnings attached, rather than silently. |
| Coach report UI (`frontend/web/src/components/ui/CoachReport.jsx`) | ✅ | Renders the report with measured values and generated narrative shown separately. |
| Tactical timeline (`nexus/sports/timeline.py`) | 🚧 | Defines and parses the timeline shape the coach expects, but no backend route serves it yet (`ai/team_intelligence/{possession,transition}_analysis/` are not built), so it currently returns `None` and the coach says so. |
| End-to-end regression test (`tests/test_pipeline_e2e.py`) | ✅ | Runs the real pipeline on `samples/sample_15s.mp4` and compares the full output against a recorded fingerprint. Skips where the clip or checkpoints are absent (CI).
| ⚠️ | Implemented but with a stated limitation that affects correctness |
| 🚧 | Code exists but is dormant / not reachable from the production path |
| ❌ | **MISSING** — does not exist anywhere in the repo |

---

## 1. End-to-end flow

```
 VIDEO ──► FRAME ──► DETECTION ──► TRACKING ──► CALIBRATION ──► TEAM SEP.
                                        │            │              │
                                        └────────────┴──────────────┤
                                                                    ▼
   FRONTEND ◄── API ◄── LLM COACH ◄── SIMULATION ◄── INTELLIGENCE ◄── EVENTS
                                          ❌            (player/team)
```

### Stage-by-stage

| # | Stage | Status | Implemented by |
|---|---|---|---|
| 1 | VIDEO → FRAME | ✅ | `backend/api/main.py::upload_video` (`POST /api/videos/upload`) → `backend/tasks.py` → `runner.run_pipeline()`; decode via `cv2.VideoCapture` |
| 2 | FRAME → DETECTION | ✅ | `ai/computer_vision/player_tracking/tracker.py::track_video()` |
| 3 | DETECTION → TRACKING | ✅ | same call — ultralytics' built-in ByteTrack (`.track(tracker="bytetrack.yaml")`) |
| 3b | offline re-tracker | 🚧 | `player_tracking/bytetrack.py` via `track_detections()`. **Decision 2026-08-13: kept deliberately, not dead code** — see §6.5. Not in the production path by design |
| 4 | TRACKING → CALIBRATION | ⚠️ | **Automatic**: `tactical_analysis/auto_calibration.py::AutoCalibrator` runs the pose model per frame, with camera-motion-driven reuse and jump rejection. **Manual override**: `manual_calibration.py` still wins when `calibrations/{match_id}.json` exists. Both produce the same `HomographyResult`/`CalibrationState`. ⚠️ the model does not yet generalise to broadcast footage — §6.3 |
| 4b | per-frame synchronisation | ✅ | `ai/computer_vision/frame_data.py::FrameData` — one object per frame holding all five models' output; built by `runner.py::_build_frame_data()` |
| 4c | calibration validity + history | ✅ | `CalibrationState.valid` (confidence gate + field-region cross-check), persisted per episode to the `calibration_status` table — §6.6 |
| 5 | CALIBRATION math | ✅ | `tactical_analysis/homography.py::compute_homography / homography_confidence / pixel_to_pitch` — solid, outputs `reprojection_error_m` and a 0–1 confidence |
| 6 | pixel → pitch | ✅ | `player_tracking/trajectory.py::enrich_with_pitch_coordinates()`; **uses `det.foot_point()` (line 130), already correct** — not bbox-centre |
| 7 | TEAM SEPARATION | ✅ | `tactical_analysis/team_assignment.py::assign_teams()` — HSV + k-means jersey clustering, returns real per-run confidence |
| 8 | EVENTS | ⚠️ | `possession.py` (radius heuristic), `pass_detection/pass_heuristics.py`, `shot_detection/shot_heuristics.py` |
| 9 | TEAM INTELLIGENCE | ⚠️ | `ai/team_intelligence/{formation_stability,pressing_structure_analysis,weak_zone_detection}` |
| 10 | PLAYER INTELLIGENCE | ✅ | 9 scorers under `ai/player_intelligence/*/score.py` |
| 11 | SIMULATION | ❌ | `ai/simulation_ai/*` is **empty scaffolding**; there is no simulation trainer (the empty `train_simulation.py` stub was removed 2026-09-16) |
| 12 | LLM COACH | ⚠️ | `nexus/sports/coach.py::CoachAssistant` + `tactical.py::derive_findings()` — **not** `ai/llm_coach/`, which is empty scaffolding |
| 13 | API | ✅ | `backend/api/main.py`, `backend/api/player_intelligence.py`, … |
| 14 | FRONTEND | ✅ | `frontend/web/src/` |

---

## 2. Detection models

### Before this overhaul

A **single combined 4-class detector** — `0=ball, 1=goalkeeper, 2=player,
3=referee` — trained by a since-removed Phase 0-1 script, loaded through one
`$YOLO_MODEL_PATH` env var. `docker-compose.gpu.yml` pointed that variable
at `/app/models/yolo/yolov8s.pt`, a **stock untrained COCO checkpoint**.

### After

Five independently trained and versioned models declared in
[`configs/models.yaml`](../configs/models.yaml), resolved through
`configs/registry.py`:

| Model | Task | Dataset (on disk) | Images | Purpose |
|---|---|---|---|---|
| `player` | detect | `player_datasets/{1,2}` | 13,480 | players |
| `ball` | detect | `ball_datasets/{1..6}` | 1,912 | ball (small-object tuned) |
| `field` | segment | `calibration_dataset` | 976 | *where is the pitch* |
| `calibration` | pose (32 kpt) | `field_datasets/2` | 317 | *how do pixels map to pitch metres* |
| `goalpost` | detect | `goalpost_datast/1` | 3,161 | goal-mouth geometry for shots |

> ### ⚠️ The `field` and `calibration` directory names are inverted
>
> Verified by reading the labels, not the directory names:
> - `field_datasets/2/` declares `kpt_shape: [32, 3]` and every label row
>   has 101 fields (1 class + 4 bbox + 32×3 keypoints) → **pitch landmarks
>   = calibration**.
> - `calibration_dataset/` has exactly 9 fields per row (1 class + a
>   4-point polygon) → **pitch region = field detection**.
>
> The registry binds these **by content**. Do not "correct" the mapping to
> match the directory names — that would train a pose model on polygon
> labels and a segmentation model on keypoint labels.

### Goalkeeper / referee

The new `player` dataset is `nc: 1, names: ['player']` — it has no
goalkeeper or referee class, unlike the old 4-class model.

**Decision (2026-08-12): player-only detection + downstream heuristics.**
Referees are recovered as jersey-cluster outliers (they match neither
team's kit) and goalkeepers as outliers that are additionally inside a
penalty area and spatially isolated. This extends the existing special-case
handling in `team_assignment.py` (which already leaves goalkeepers
unassigned).

Both **must** be written with `MetricMethod.heuristic_proxy` — they are not
model predictions.

### ✅ IMPLEMENTED 2026-08-13 — `tactical_analysis/role_inference.py`

The carried risk above ("until implemented, referees will be clustered as
ordinary players and pollute team assignment") is closed.
`runner.py` Stage 3.5 calls `infer_roles()` and removes referee tracks from
the trajectories fed to team-level scoring.

Two supporting changes made this possible:

- `assign_teams_with_stats()` (new; `assign_teams()` is now a thin
  wrapper, so the seven existing call sites and the test suite are
  unchanged) returns per-track `TrackKitStats` including
  **`dist_to_nearest_centroid`**. The pre-existing `margin` is a *ratio*
  and therefore cannot distinguish "equidistant because it sits between
  the two kits" (a player with noisy crops) from "equidistant because it
  is far from both" (an official). Only the raw distance separates them,
  and both previously came out as `team_id=None`.
- The outlier test is **relative to each video's own population**
  (median + 3·MAD), never an absolute colour distance — kit palettes and
  lighting do not transfer between matches. Median/MAD rather than
  mean/std because the officials *are* the outliers being looked for, and
  1–3 of them among ~22 players would inflate a standard deviation enough
  to hide themselves.

**Goalkeeper inference requires a valid calibration and does not guess
without one.** Separating a keeper from a referee is a positional test
(inside their own penalty area, spatially isolated), and which part of the
*image* is a penalty area depends entirely on camera angle. With
`calibration.valid == False` the role is reported as `unknown` with a
stated reason — never defaulted to `player`. On the broadcast footage in
`storage/uploads` the calibration model produces no valid homography at
all (see below), so goalkeeper inference genuinely does not fire there.

Thresholds (`REFEREE_KIT_OUTLIER_MADS`, `GOALKEEPER_MIN_ISOLATION_M`,
`GOALKEEPER_MAX_DEPTH_M`) are **uncalibrated starting points** — the same
caveat as `SCANS_PER_MINUTE_TARGET`. The mechanism is sound; the specific
numbers are not yet fitted to labelled officials. Do not present per-role
output as validated.

---

## 3. Known-broken edges

_Reconciled 2026-08-13 (Phase 3). This section had drifted out of step with
the rest of the document: two of its ❌ entries were resolved by Phase 2 and
contradicted §4c/§6.6, and its line references pointed at a version of
`runner.py` that no longer exists. Resolved items are listed with what
closed them; the still-open list is now genuinely still open._

### ✅ Closed by Phase 3

| Was | Closed by |
|---|---|
| `nexus/sports/video.py` called `POST /api/video/process` and `GET /api/video/status/{job_id}` — neither exists | Now calls `POST /api/videos/upload` (multipart) and `GET /api/processing/{job_id}`. Covered by `nexus/tests/test_sports_video.py` (7 tests) including the full upload → poll → `build_report()` path |
| Team metrics pooled every player regardless of `team_id` | `backend/pipeline/team_scoring.py::_score_team_intelligence()` computes each metric once per team over that team's own tracks and tags every row with `team_id` |
| `attacking_direction` a hardcoded `left_to_right` global | `tactical_analysis/attacking_direction.py`, inferred per team from tracked positions; `unknown` when unmeasurable, and consumers decline rather than default |
| `NORMALIZATION_CONSTANT = 8.0` placeholder | Derived at import from template geometry — `formation_detection.py::_derive_normalization_constant()`, **7.89 m** |
| `_build_players_by_frame()` keyed by list index | Keys by real `frame_id`; `_positions_by_frame()` had the same defect and was fixed with it |
| Shot detection had no goalpost geometry | `shot_heuristics.build_goal_mouths()` measures the mouth from `goalpost_v1` detections, gated at confidence ≥ 0.5, falling back to nominal geometry with the source recorded per event |

### ✅ Already closed by Phase 2 — this section was stale

- **"No calibration table in the database"** — contradicted §4c and §6.6.
  The `calibration_status` table exists and is written per calibration
  episode by `backend/pipeline/calibration.py::_persist_calibration_status()`.
- **"No automatic calibration, no temporal stability"** — contradicted §4
  and §6.3. `tactical_analysis/auto_calibration.py::AutoCalibrator` runs
  per frame with camera-motion-driven reuse and jump rejection.

### ⚠️ Found during Phase 3, not previously documented

**Two call sites were already looking up real `frame_id`s in the
index-keyed dict.** `_detect_events()` passed `ball_pt["frame_id"]` and
`_enrich_first_touch_metadata()` passed `touch_frame` into a structure
keyed by point-list index. Before the first tracking gap these coincide, so
it worked on clean clips and silently returned the wrong frame's players
after any occlusion — possession attribution and first-touch pressure both
read from it. Fixing the keying repaired both; they were never listed as
broken because the mismatch was invisible from either end.

**`detect_formation()` was fed the first ten trajectory points, not ten
players.** `all_positions[:10]` is ten consecutive samples of whichever one
or two players sorted first, matched by the Hungarian algorithm against a
full 10-slot formation template. Now one mean position per player, per
team.

**`detect_formation()` still truncates templates by slice — STILL OPEN.**
When fewer than 10 players are tracked it uses `slots_m[:n_players]`, which
takes the first N slots in template declaration order — the back four
first. An 8-player sample is therefore matched against "the defenders and
midfielders of each template" rather than a representative subset, biasing
towards whichever template has the most defensive slots. Not fixed here:
choosing which slots to drop is a modelling decision, not a bug fix.

**`NORMALIZATION_CONSTANT` lives in `formation_detection.py`,** not
`formation_templates.py` as §3 previously stated.

### ⚠️ Still open

- **Pressing intensity has no real input.** `compute_pressing_intensity()`
  is called with an empty per-frame defender-to-ball distance list, so it
  always answers through its own `low_sample` gate. It needs possession
  attributed to a team over time, which nothing in the pipeline computes
  yet. It is wired and honest, not measured.
- **`_nearest_other_player_distance()` uses nearest *player*, not nearest
  *opponent*.** Real `team_id` now exists, so this can be tightened; it
  still feeds first-touch pressure and off-ball separation as a stated
  approximation.
- **Off-ball possession-window sub-scores remain 0/0** for the same
  missing team-scoped possession windows, and report `None` rather than a
  guess.
- **Goalkeeper role inference requires a valid calibration** and therefore
  does not fire on real broadcast footage at all (§6.3).

---

## 4. Honesty contract

`backend/database/models.py:17-25` defines the enums every emitted metric
must carry:

```python
class MetricMethod(str, enum.Enum):
    ml_trained = "ml_trained"          # a trained model produced this
    deterministic = "deterministic"    # exact computation (e.g. homography)
    heuristic_proxy = "heuristic_proxy"  # threshold/proxy — most tactical metrics

class MetricConfidence(str, enum.Enum):
    normal, low_sample, low_upstream_confidence
```

Rules already enforced in `runner.py` and worth preserving:

- `HOMOGRAPHY_CONFIDENCE_MIN = 0.6` gates every pitch-coordinate read.
  Below it, frames are **dropped, not imputed** (`trajectory.py`).
- Missing assets raise `PipelineAssetError` — never substituted with a
  placeholder.
- The LLM is forbidden from inventing or recomputing numeric values
  (`nexus/sports/coach.py`); findings are derived deterministically first
  by `tactical.py::derive_findings()`.
- `nexus/sports/adapter.py` talks to `backend/api/*` over HTTP only and
  never imports `ai/` or `backend/` directly. Keep that decoupling.

---

## 5. Data-quality constraints that limit what the models can achieve

From [`docs/dataset_audit/`](dataset_audit/):

| Dataset | Images | Labelled | Note |
|---|---:|---:|---|
| player | 13,480 | 99.9% | clean, no duplicates, no leakage |
| ball | 1,912 | **38.6%** | 1,174 images have **no label file** |
| field | 976 | 100% | 8.7% within-split duplication |
| calibration | 317 | 100% | clean, but the smallest set |
| goalpost | 3,161 | 100% | 1 file exceeds Windows `MAX_PATH` |

Two findings that required remediation before training:

1. **Ball cross-split leakage.** The 6 ball source directories each did
   their own train/valid/test split, so 114 byte-identical images appeared
   in more than one split. Training on the raw directories and reporting
   mAP would have been reporting memorisation. Fixed by
   `scripts/dataset_dedupe.py`, which emits deduplicated split manifests
   (test > valid > train priority) without mutating the source data.
2. **Ball is only 38.6% labelled.** Ultralytics silently treats the other
   1,174 images as background negatives. This is the primary constraint on
   ball recall — the metric the entire possession/pass/shot chain depends
   on.

---

## 6. Phase 2 engineering — verified findings (2026-08-13)

Everything in this section was measured, not assumed.

### 6.1 Field segmentation: mAP50 == mAP50-95 == 0.9950 is real, not a reporting bug

`models/yolo/field_v1/metrics.json` carries ultralytics' raw `results_dict`
alongside the parsed values, and the raw keys agree — `metrics/mAP50(B)` and
`metrics/mAP50-95(B)` are both `0.995`, as are both mask keys. The two fields
were not accidentally written from one value.

Measured directly, per test image (mask IoU against the ground-truth polygon):

```
n test images: 15
mask IoU: min=0.9628  median=0.9864  mean=0.9855  max=0.9984
images with IoU >= 0.95: 15/15
```

Every prediction clears IoU 0.95, so every AP threshold from 0.50 to 0.95
scores identically and their average equals AP50. (0.995, not 1.0, is
ultralytics' interpolation ceiling for a perfect PR curve.)

**Why it is achievable here, and why it is less impressive than it looks:**
the label is a 4-point convex quadrilateral covering a median 62% of the
frame. A large, simple, convex shape has a very low boundary-to-area ratio,
so boundary error barely moves IoU. This is a near-trivial segmentation
target, not evidence of an exceptional model.

**Caveats that must travel with this number:** the test split is **15
images**, and `docs/dataset_audit/field.md` reports 46 duplicate groups
within train. `min IoU = 0.9628` sits only just above the 0.95 threshold — a
marginally worse model would drop mAP50-95 sharply. Treat 0.995 as "the pitch
region is an easy target", not as a headline result.

### 6.2 Ball: the 38.6% figure is dataset contamination, not "ball out of frame"

738 label files against 1,912 images = **38.6%**, confirmed. The other 61.4%
are images with **no `.txt` file at all** (`empty labels: 0` in every split),
so they train as pure negatives.

Sampling 8 orphan images at random and looking at them: **8/8 contain a
clearly visible ball**, and 7/8 are not football at all — a baseball with bat
and glove, two basketballs in a posed family photo, a volleyball on a beach,
product/stock shots of balls filling the frame. The eighth is a real match
frame with a plainly visible, unlabelled ball.

Quantified on 250 random orphans, the same sample both times:

| model | fires on | median conf |
|---|---|---|
| trained `ball_v1` (conf>=0.10) | **6 / 250 (2.4%)** | 0.277, none >=0.5 |
| stock COCO `yolov8s`, class `sports ball` (conf>=0.25) | **63 / 250 (25.2%)** | 0.827, 53 >=0.5 |

A generic untrained model finds a ball in at least a quarter of the images
our model was taught to call empty. The trained model has been driven to
near-total suppression on exactly those images.

**It gets worse after deduplication.** The generated train manifest
(`runs/_data/ball/train.txt`) is **824 images of which 644 (78.2%) are
unlabelled negatives** — 180 positive training images. The cross-split
dedupe correctly removed leaked duplicates, but the images it kept in train
were disproportionately the unlabelled ones.

> **Corrected 2026-08-13.** This paragraph previously read "698 (84.7%)
> unlabelled … only 126 positive". That was never true of this manifest.
> The 824 files are the same files; 180/644 is the correct composition and
> was already correct when this section was written. Evidence: a fresh
> `scripts/dataset_dedupe.py` run reproduces the manifests byte-for-byte,
> the split is 180/644 under four independent definitions of "labelled",
> no label file has been modified since the manifest was written, and
> ultralytics' own label cache — written at 14:57 on 2026-08-12, four
> minutes before `ball_v1` training began — records
> `(nf=180, nm=644, ne=0, nc=0, total=824)` with 318 instances. See
> `docs/dataset_audit/cv_dataset_preparation_report.md` §2. The
> substantive point survives: train is still starved of positives, just
> less severely than stated.

**Verdict: a fixable dataset defect, not expected occlusion.** It is the
direct explanation for recall 0.842, and it matters because Phase 3 leans on
ball trajectory. Fix before Phase 3: drop the off-domain imagery (other
sports, studio/product shots) rather than labelling it, and label the
in-domain frames that do have a visible ball.

### 6.3 Automatic calibration: the mapping table is right, the model is not

`tactical_analysis/pitch_keypoints.py` cited
`python -m scripts.validate_pitch_keypoints` for a median 0.247 m
reprojection error — **that script did not exist**. It has been written.

Running it (`--split train`, ground-truth keypoints through the table):

```
images fitted=255   median = 0.295 m   mean = 0.327 m
90th pct = 0.510 m  under 1.0 m = 254/255 (99.6%)
```

Substantively reproduces the claim. **The index -> pitch-metre table is
correct.**

The model's own keypoints are a different story.
`python -m scripts.validate_auto_calibration --split test` fits a homography
from *predicted* keypoints, then projects the *ground-truth* landmarks
through it and measures the true positional error in metres:

```
images: 28   calibrated: 28
TRUE positional error:  median 2.004 m   mean 14.597 m   max 58.770 m
  under 1 m:  2/28      under 3 m: 19/28      over 10 m: 7/28
fits clearing HOMOGRAPHY_CONFIDENCE_MIN (0.6):  1/28
  of those: TRUE error median 1.539 m, max 1.539 m
  cleared the gate but >5 m wrong:  0/1
```

The dominant failure mode is a **mirrored reading of the pitch**: the model
emits a self-consistent full 32-keypoint template for the *wrong end*. On one
test image the left-side landmarks 0-12 land within 3-29 px of ground truth
while indices 13-16 and 30 are **640-690 px** off, and it hallucinates
right-side landmarks not in frame. RANSAC then fits the *wrong* consensus set
to a tidy 0.468 m internal error.

`flip_idx` **is** correctly present in the generated `data.yaml`, so this is
not the augmentation misconfiguration `models.yaml` warns about — it is a
capability limit of a 255-image training set that cannot reliably tell which
end of the pitch it is looking at when only one end is visible.

**Two consequences that are load-bearing and must not be "optimised" away:**

1. `compute_homography()` reports `reprojection_error_m` over **all** input
   points, including RANSAC outliers. That is what makes the mirrored fits
   score ~0 confidence and get rejected — the wrong consensus set leaves the
   other half of the keypoints as huge outliers. "Improving" this to report
   inlier-only error would make the catastrophic mirrored fits look excellent
   (0.468 m) and let them straight through. **Do not change it.**
2. The confidence gate is therefore very conservative — it rejects 27/28
   in-domain fits — but it admitted **zero** catastrophic errors. That trade
   is the right one for a value feeding every pitch-space metric.

**Off-domain generalisation is worse still.** On the broadcast clips in
`storage/uploads` the calibration model fires on ~1 sampled frame in 6 at
1280x720 and **0 in 6** at 854x480, and the resulting homographies reproject
at 7-25 m. A full 140-frame pipeline run produced
`calibration_valid_fraction = 0.0` — correctly, with a populated
`invalid_reason` on every episode.

**This is why `manual_calibration.py` is retained as an override and why
`calibration.valid` is a hard gate rather than a warning.** Automatic
calibration is wired, correct, and honest about failing; it is not yet usable
on broadcast footage. Improving the calibration model (more data, more camera
angles, both ends visible) is the highest-value Phase 3 input.

### 6.4 Tracking anchor — verified correct, no change needed

`TrackedDetection.foot_point()` returns bottom-centre and `trajectory.py:130`
already calls it — verified by asserting on the function's own source, not by
reading its docstring. Concrete effect for a 40x110 px box under a real
homography:

```
foot_point  = (420.0, 410.0)  ->  (52.833, 36.682) m
bbox centre = (420.0, 355.0)  ->  (52.882, 29.956) m
difference  =  6.726 m of position error avoided
```

`trajectory.py` computes speed/distance/acceleration from these projected
points, so the anchor was the only thing to check there; it is correct. The
ball uses its **centre**, which is right for a sphere (documented in
`detectors.py` — a ball in flight is off the z=0 plane anyway, and the centre
keeps that error symmetric rather than biased).

### 6.5 Two trackers — kept, with a stated rationale

`bytetrack.py` (hand-rolled) is **kept**, not deleted. Reason: ultralytics'
`.track()` only tracks the live output of the one model it is running and has
no entry point accepting externally-computed boxes, so offline re-tracking of
merged/ensembled detections — a real workflow now that five models exist —
needs its own tracker. It also carries two regression suites
(`test_bytetrack_fixes.py`, `test_tracking_pipeline.py`) covering two real
fixed bugs.

**The rule this creates:** `run_pipeline()` uses `track_video()` and nothing
else. A second tracker in the production path is what was rejected, not the
existence of an offline re-tracking tool with its own tests.

Separately, `ai/computer_vision/pipeline.py` and `ai/task.py` were
**deleted**. They were a duplicate pipeline that (a) did not import at all
(`ModuleNotFoundError: No module named 'computer_vision'`), (b) was
unreachable — `celery_app.py` has `include=["backend.tasks"]` and nothing
imported `ai.task` — and (c) registered a *conflicting* Celery task name,
`backend.tasks.process_video_job`. It also hardcoded `yolov8n.pt` and held
one of the four duplicated `homography_confidence >= 0.6` gates.

### 6.6 The single calibration gate

`calibration.valid` (`frame_data.CalibrationState`) is now the one source of
truth. It is the `HOMOGRAPHY_CONFIDENCE_MIN` gate **and** a geometric
cross-check against the field model's detected pitch polygon, computed once
per frame and read everywhere.

Previously four sites re-derived `homography_confidence >= 0.6`
independently:

| site | now |
|---|---|
| `backend/pipeline/runner.py` | reads `calibration.valid` per frame |
| `tactical_analysis/possession.py` | `get_ball_possessor(..., calibration_valid=)`; the old comparison remains only as the fallback when the caller passes `None` |
| `backend/api/tracking.py` | queries the `calibration_status` table; falls back to the per-row comparison only for matches processed before that table existed |
| `ai/computer_vision/pipeline.py` | deleted (see 6.5) |

**Known limit of the geometric cross-check:** it rejects keypoints falling
*outside* the detected pitch (landmarks matched to the crowd or hoardings).
It does **not** catch the mirrored-pitch failure in 6.3, because those
keypoints are all legitimately on the pitch — just with the wrong identities.
The confidence gate is what catches that one.

### 6.7 Reproducing everything in this section

```
python -m scripts.validate_pitch_keypoints --split train
python -m scripts.validate_auto_calibration --split test
python -m scripts.print_model_env --check
pytest ai/computer_vision/player_tracking/tests/ \
       ai/computer_vision/tactical_analysis/tests/ backend/tests/ -q
```

---

## 7. Phase 3 engineering (2026-08-13)

Scope so far: the §3 known-broken edges. See §3 for what each fix closed
and what it uncovered. Files changed:

| Area | Files |
|---|---|
| NEXUS video flow | `nexus/sports/video.py`, `nexus/tests/test_sports_video.py` |
| Per-team metrics | `backend/pipeline/runner.py`, `ai/team_intelligence/weak_zone_detection/weak_zones.py` |
| Attacking direction | `ai/computer_vision/tactical_analysis/attacking_direction.py` (new), `runner.py`, `ai/player_intelligence/passing_vision_score/score.py` |
| Formation constant | `ai/computer_vision/tactical_analysis/formation_detection.py` |
| frame_id keying | `backend/pipeline/runner.py` |
| Goalpost-backed shots | `ai/computer_vision/shot_detection/shot_heuristics.py` |
| Tests | `backend/tests/test_team_split_and_direction.py` (new), `ai/computer_vision/tactical_analysis/tests/test_event_heuristics.py` |

**Everything above is verified on synthetic input, not on real footage.**
All of it operates on pitch coordinates, which exist only when
`calibration.valid` is True — never, on the clips in `storage/uploads`
(§6.3). That is a real limit on the evidence, stated rather than papered
over: the arithmetic is tested, the end-to-end behaviour on broadcast video
is not.

Two behaviour changes are worth knowing about before reading any output:

1. **`detect_shots()` no longer defaults to `left_to_right`.** Without a
   known direction a fast ball is emitted as `uncertain`, not as a shot.
   On real footage, where direction is unmeasurable, this means **zero
   shots** rather than a plausible-looking count scored against a guessed
   end. Events are now typed `shot | cross | clearance | uncertain`.
2. **`forward_passes` is `None` when direction is unknown**, and
   `score_passing_vision()` reweights over its remaining sub-scores rather
   than counting every +x pass as forward for both teams.

```
pytest ai/ backend/tests/ nexus/tests/test_sports_video.py -q   # 319 passed
```

### 7.1 End-to-end validation on real footage (2026-08-15)

`python -m scripts.validate_pipeline_e2e --clips 3 --frames 140 --start-frame 200`
runs the real `run_pipeline()` over real uploads and writes
`docs/dataset_audit/phase3_e2e_results.json`.

**Only two distinct clips exist.** `storage/uploads` holds 17 files but just
**2 unique MD5s** — every other file is a byte-identical duplicate. The
requested "2–3 clips" is therefore satisfied at 2, and no third source
footage exists to run.

| | clip A | clip B |
|---|---:|---:|
| resolution | 854x480 | 1280x720 |
| frames processed | 140 | 140 |
| players tracked | 37 | 59 |
| player_tracking rows | 592 | 2,225 |
| player metrics | 333 | 531 |
| team metrics | **10** | **10** |
| teams found | `team-home`, `team-away` | `team-home`, `team-away` |
| events detected | 0 | 0 |
| **calibration_valid_fraction** | **0.0** | **0.0** |
| honesty-contract violations | **0** | **0** |
| wall clock (CPU-only torch) | 45.8 s | 62.5 s |

**The per-team fix is confirmed on real video.** 10 team-metric rows = 5
metrics x 2 teams, tagged `team-home`/`team-away`. Before Phase 3 this was
5 rows under a single pooled `unassigned` bucket. Team assignment is
jersey-colour clustering and does **not** need calibration, which is why it
works here while everything pitch-derived does not.

**Everything pitch-derived degrades honestly.** Every team metric is
`value=None` with `low_sample`, and `calibration_status` records
`invalid_reason = "no pitch keypoints above the visibility floor"` for the
whole run. Zero metrics carried an invalid `MetricMethod`/`MetricConfidence`.

**Zero events is correct, not a regression.** No valid calibration means no
pitch coordinates, so there is no ball trajectory to detect passes, shots or
turnovers from. Note the Phase 3 change compounds this deliberately:
`detect_shots()` no longer assumes `left_to_right`, so even given a ball
trajectory it would emit `uncertain` rather than a shot count scored against
a guessed end.

**Calibration debugging (§7.2) isolates the failure.** On a broadcast frame
the player model returns 22 detections and the field model fires at 0.930,
while the calibration model returns **0 keypoints above the visibility
floor**. Detection and field segmentation are healthy; the pose model alone
fails. On the one in-domain frame that clears the gate
(`42ba34_0_4`, confidence 0.792 — consistent with §6.3's 1/28) the same tool
projects 26 detected players to plausible pitch metres.

**The what-if engine refuses to invent.** Run on clip B's real team-split
trajectories it returns `baseline=None, simulated=None, delta=None,
confidence=low_upstream_confidence` for every metric, with an explicit
`unavailable` reason — rather than moving players around a coordinate space
that does not exist.

### 7.2 Newly found and fixed during validation

**The tracking payload dropped calibration validity.**
`GET /api/matches/{id}/tracking` returned `pitch_x_m`/`pitch_y_m` per player
with nothing indicating whether they were trustworthy. On current footage
every one is `None`, and the frontend rendered that as an ordinary empty
overlay. Added `TrackingWindowResponse.calibration`
(`backend/api/schemas.py`, `backend/api/tracking.py`) carrying
`valid_frame_ranges`, `valid_in_window` and
`rows_with_pitch_coordinates`; `TabMatchAnalysis.jsx` now shows
**"Calibration unavailable"** with the reason after a run completes.
Covered by `backend/tests/test_tracking_calibration_field.py`.

Worth noting how this nearly shipped silently: the endpoint declares
`response_model=TrackingWindowResponse`, so FastAPI would have **stripped
the new field** from the HTTP response while the direct function call in a
test still returned it. The schema had to change too.

### 7.3 NOT VERIFIED

* **Any pitch-coordinate metric on real footage** — compactness, formation,
  stability, weak zones, pressing, shots, distances in metres. All are
  gated to `None` because `calibration_valid_fraction = 0.0`. Their
  arithmetic is verified on synthetic input
  (`backend/tests/test_team_split_and_direction.py`,
  `ai/simulation_ai/tests/test_what_if_engine.py`), not on broadcast video.
* **Goalpost-backed shot classification on real footage.** `build_goal_mouths()`
  needs goalpost detections *with a valid calibration* to place a mouth in
  pitch metres, so it fell back to nominal geometry and no shots were
  emitted. Verified by unit test only.
* **Attacking-direction inference on real footage** — needs pitch
  coordinates; returned `unknown`, which is the designed behaviour.
* **LLM coach narration.** The flow was exercised up to the LLM boundary
  (`nexus/tests/test_sports_video.py`: multipart upload → poll
  `/api/processing/{job_id}` → `build_report()` on the backend-assigned
  match id). No live provider call was made — that is an outward-facing,
  billable request and was not run unprompted. The deterministic half of
  the report (`tactical.py::derive_findings()`) is what produces the
  numbers; the LLM only narrates and is forbidden from inventing values.
