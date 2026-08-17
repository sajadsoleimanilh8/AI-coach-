# CV dataset preparation audit

Generated 2026-08-13 from the registry-resolved sources. Revised the same day
to close the three gaps raised in review. No training was started, no
checkpoint was promoted, no registered source image or label was modified,
and no production inference or pipeline module was touched.

## Scope and existing code

The audit reused `configs/models.yaml`, `configs/datasets.yaml`,
`configs/registry.py`, `frame_data.py`, `pitch_keypoints.py`,
`auto_calibration.py`, `homography.py`, `scripts/validate_pitch_keypoints.py`,
`scripts/validate_auto_calibration.py`, and `scripts/dataset_dedupe.py`.
`compute_homography()` remains unchanged: its confidence uses reprojection
error over every supplied point, not only RANSAC inliers.

Player, field, and goalpost datasets were not changed.

New in this revision:

* `scripts/build_zero_leakage_splits.py` — group-aware split proposal.
* `scripts/audit_calibration_landmark_coverage.py` — landmark deficit analysis.
* `scripts/audit_ball_dataset.py` — batched detector pass; COCO class filter.

---

## 1. Environment and the detector-assisted triage

### Root cause: the script had never been executed

The previous report attributed the empty detector columns to a project venv
pointing at a removed Python executable. That is not what happened. The venv
at `venv/` is intact and always was: `venv/Scripts/python.exe` runs, its
`pyvenv.cfg` resolves to an existing 3.11.0 base install, and Ultralytics
8.4.116, Torch 2.13.0+cpu, PIL 12.3.0, OpenCV 5.0.0 and NumPy 2.4.6 all
import from it. No repair was required and none was made.

The actual cause is visible in the artifact itself. The previous
`triage.csv` began with a UTF-8 BOM, quoted every field, and used CRLF —
none of which Python's `csv.DictWriter` with `encoding="utf-8"` produces. Its
`heuristic` column contained `not_run_python_unavailable` for all 1,404 rows,
a value that appears nowhere in `scripts/audit_ball_dataset.py`; the script
can only emit `football_context_candidate`, `non_football_or_unknown_candidate`
or `unavailable`. `calibration_coverage.csv` showed the same signature plus
uppercase SHA-1 digests, which `hashlib.hexdigest()` never returns, and it was
missing the `width_height` column the script emits.

Both CSVs were assembled by a separate PowerShell pass. The Python tools were
written but never run, so `not_run` was literal: the `--trained-model` and
`--coco-model` arguments were never supplied, and `_detector_signal()`
returned `("not_run", 0.0, "model_not_configured")` before any inference was
attempted. The environment was never the blocker.

### Two fixes to the tool before re-running

**The COCO signal was measuring the wrong thing.** `_detector_signal()` took
the maximum confidence over *all* boxes the general detector returned. On
`yolov8s.pt` that is all 80 COCO classes, so a crowd of players scored as a
ball signal. The call now passes `classes=[32]` (`sports ball`). Without this
the trained-vs-COCO comparison the review asked for would have been
meaningless.

**Inference is now batched.** The per-image `model.predict()` call re-entered
the predictor 1,404 times per model. Detection is batched at 8, with one
model load, and the trained model runs at `imgsz=1280` to match the imgsz
recorded in `runs/train/ball_v1/args.yaml`.

Bucket assignment is unchanged. Every unlabeled image is still `ambiguous`
and `human_decision` is still `REVIEW_REQUIRED` regardless of any score. The
one behavioural change is that the `flags=` hint in the `reason` column now
fires at a named threshold (0.25, recorded inline in the row) instead of at
`conf>=0.01`, where it fired on essentially everything and ordered nothing.

### Before / after

| | before | after |
|---|---:|---:|
| unlabeled rows with a trained-model signal | 0 / 853 | **853 / 853** |
| unlabeled rows with a COCO signal | 0 / 853 | **853 / 853** |
| unlabeled rows with a scene-heuristic signal | 0 / 853 | **853 / 853** |
| all rows populated | 0 / 1,404 | **1,404 / 1,404** |

### Reconciling 2.4% vs 25.2% against the full unlabeled set

Measured over all 853 deduplicated unlabeled images, not a 250-image sample:

| detector | >=0.10 | >=0.25 | >=0.50 |
|---|---:|---:|---:|
| trained `ball_v1` | 28 (3.3%) | 16 (1.9%) | **0 (0.0%)** |
| stock COCO `yolov8s`, class 32 | 192 (22.5%) | 167 (19.6%) | 139 (16.3%) |

The prior 250-image sample reported 6/250 (2.4%) for the trained model at
`>=0.10` and 63/250 (25.2%) for COCO at `>=0.25`. The full set gives 3.3% and
19.6% at those same thresholds. The sample was directionally right and
roughly right in magnitude; the gap narrows from about 10.5x to about 6x but
does not close.

The sharper number is the last column. **The trained model does not produce a
single detection at or above 0.50 on any of the 853 unlabeled images**, while
COCO produces 139. The suppression claim survives at full scale in a stronger
form than the sample showed.

One result changes how the gap should be read. On the 551 *labeled* images:

| detector | >=0.25 | >=0.50 |
|---|---:|---:|
| trained `ball_v1` | 508 (92.2%) | 259 (47.0%) |
| stock COCO `yolov8s`, class 32 | 45 (8.2%) | 30 (5.4%) |

COCO finds a sports ball in only 8.2% of the images that are *known* to
contain a labelled football, but in 19.6% of the unlabeled ones — and at
`>=0.25` on unlabeled images the two detectors overlap on **zero** rows
(16 trained-only, 167 COCO-only, 670 neither). The two are firing on disjoint
phenomena. That is consistent with the earlier 8/8 manual sample, in which 7
of 8 orphans were not football at all: COCO is detecting large, close,
generic balls — basketballs, volleyballs, product shots — and is largely blind
to the small distant footballs that dominate the labelled set.

So the 25.2% figure should not be read as "a quarter of the orphans are
missed football balls". The review queue this produces is:

* **16 rows** flagged by the trained model — the most likely genuine missed
  footballs, highest-value to label.
* **167 rows** flagged only by COCO — most likely off-domain imagery to drop
  rather than label.
* **670 rows** flagged by neither.

**Still requires human judgement.** Every one of the 853 remains `ambiguous`.
The three-way split above orders the queue; it does not decide it. The 551
existing positives still need box-correctness QA.

---

## 2. The train-manifest discrepancy is resolved

**They are the same 824 files, and 180/644 is correct. The 698/126 figure was
never correct for this manifest.**

Three independent lines of evidence:

1. **The file lists are identical.** Re-running `scripts/dataset_dedupe.py`
   today reproduces `runs/_data/ball/{train,val,test}.txt` exactly — 824/380/200,
   zero files added, zero removed, for all three splits. The dedupe is
   deterministic (SHA-1 with a fixed `test > valid > train` priority, no random
   seed), so this was expected but needed confirming.

2. **Label state is 180/644 under every plausible counting rule.** Label file
   exists; exists and non-empty; any member of the SHA-1 duplicate group
   labelled; all members labelled — all four give 180 labelled / 644 unlabelled.
   The number is not an artifact of how "labelled" was defined.

3. **Ultralytics recorded the same thing on 2026-08-12, before this audit
   existed.** Training the ball model left a label cache at
   `ball_datasets/1/train/labels.cache` (written 14:57:26, four minutes before
   `ball_v1` training began at 15:01). Its file list is identical to the current
   train manifest, and its scan result is `(nf=180, nm=644, ne=0, nc=0,
   total=824)` with 318 instances — matching today's measurement exactly, from
   a tool that is not mine and had no stake in the question.

Additionally, **no label file under any of the six ball sources has been
modified since the manifest was written**. Nothing changed between the two
audit runs; there was no merge, no relabel, and no different dedup pass.

Both figures were therefore *not* "correct for their respective moments".
180/644 was already true when 698/126 was published.

Where 698/126 came from is inference rather than proof. The claim lives in
`docs/pipeline_architecture.md:319-323`. The arithmetic is suggestive: 180
minus 126 is 54, and 54 is exactly the number of labelled train images
contributed by `ball_datasets/1` — the first source in registry order and the
one whose directory happens to host the cache. A count that skipped that one
source's train labels while keeping the full 824 denominator would produce
698/126. I could not reconstruct the code that did it, so I am flagging the
mechanism as probable, not established.

**Requires human action:** `docs/pipeline_architecture.md:319-323` states a
figure this audit shows to be wrong, and it carries a "Verdict" other
decisions rest on. I have not edited that file — correcting a load-bearing
architecture record is a call for its owner. The correct sentence is that the
generated train manifest is 824 images of which **644 (78.2%) are unlabelled
negatives, with 180 positives**. The substantive point it was making survives:
train is still starved of positives, just less severely than stated.

---

## 3. Leakage in the proposed splits

### It was open

The previous report contained a splitting *policy* in prose — "no group may
cross train/validation/test" — but no proposed split manifests existed on
disk. Nothing had been generated, so nothing had eliminated the 114 groups.
It is resolved now.

### What a near-duplicate check found that SHA-1 could not

`scripts/build_zero_leakage_splits.py` groups images by SHA-1, by 64-bit dHash
within a Hamming threshold, and — where filenames validly encode one — by clip
id, then assigns whole groups.

Run with near-duplicate detection disabled it reproduces the prior audit
exactly (1,404 unique, 500 duplicate groups, 114 cross-split), which is the
control confirming the tool agrees with the existing numbers before adding
anything.

Threshold sensitivity, measured rather than assumed:

| Hamming | ball groups | ball cross-split | calib groups | calib cross-split |
|---:|---:|---:|---:|---:|
| exact only | 1,404 | 114 | 317 | **0** |
| 0 | 1,389 | 115 | 312 | 2 |
| 3 | 1,262 | 125 | 290 | 6 |
| **5** | **1,188** | **127** | **276** | **11** |
| 8 | 1,093 | 134 | 253 | 17 |

No cliff, and the largest group stays at 8 images (ball) / 5 (calibration) at
threshold 5, so the grouping is not collapsing unrelated imagery. 5 is the
default; a `--hamming` flag exposes the choice.

**Calibration was the significant finding.** The prior report's "zero exact
duplicate groups" was true and badly misleading. The 317 images carry only
**18 distinct clip ids** (`<clip>_<segment>_<frame>` in the export filenames,
2 to 77 frames each), and **17 of those 18 clips were split across
train/valid/test**. Adjacent frames of one broadcast clip were being used to
evaluate a model trained on their neighbours. That is a much larger leak than
the 11 dHash groups, and exact SHA-1 could not see any of it.

Clip grouping is applied to calibration only. Ball filenames were tested for
the same signal and rejected: its six independent exports reuse generic
original stems (`-2-_jpeg`, `4_jpg`) and **58.6% of same-stem ball pairs are
more than 20 dHash bits apart** — genuinely different photographs. Grouping
ball by filename would have merged unrelated images and overstated the fix.

### A bug in my own tool, found by independent verification

The first implementation bucketed dHash into four 16-bit bands and only
compared images sharing a band. That prefilter is exact only for Hamming <= 3
by pigeonhole; at 5 it silently missed pairs. A separate verifier that
re-hashes the written manifests from scratch found **11 cross-split
near-duplicate pairs still present** in a proposal the builder had certified
as clean. The prefilter is gone — these datasets are ~2k images, so exact
pairwise comparison costs seconds — and the re-verification is below.

### Final counts

Ball, `datasets/derived/splits_zero_leakage_v1/ball/`:

| split | images | labelled | unlabelled | instances |
|---|---:|---:|---:|---:|
| train | 824 | 323 | 501 | 619 |
| val | 379 | 149 | 230 | 297 |
| test | 201 | 79 | 122 | 142 |
| total | 1,404 | 551 | 853 | 1,058 |

Calibration, `datasets/derived/splits_zero_leakage_v1/calibration/`:

| split | images | clips |
|---|---:|---:|
| train | 257 | 12 |
| val | 33 | 3 |
| test | 27 | 3 |
| total | 317 | 18 |

Split *sizes* are held at each dataset's own current proportions (ball
58.7/27.1/14.2, calibration 80.4/10.7/8.8), measured from the registry rather
than imposed. This changes which images are in each split, not how big the
splits are; resizing them is a separate decision.

Reassignment: **127 of 127** previously cross-split ball groups are now
confined to one split, and **17 of 17** calibration clips. In total 694 of
1,175 ball groups and all 18 calibration clips sit in a different split than
before, which is expected — the assignment is re-derived from scratch rather
than patched.

A side effect worth noting: the current registry splits put positives very
unevenly — 21.8% of train is labelled against 61.6% of val and 68.5% of test.
The proposal stratifies on label presence, giving 39.2% / 39.3% / 39.3%. The
starvation of positives in train that `pipeline_architecture.md` flagged is
substantially a splitting artifact, and it is fixed here without labelling
anything.

### Independent verification of the proposed splits

Re-hashing the written manifests from scratch, not reusing the builder's
objects:

| check | ball | calibration |
|---|---|---|
| images in more than one split | 0 | 0 |
| exact SHA-1 groups spanning splits | 0 | 0 |
| near-dup pairs across splits, Hamming <= 3 | 0 | 0 |
| near-dup pairs across splits, Hamming <= 5 | **0** | **0** |
| clips spanning splits | n/a | **0** (was 17/18) |

**Zero cross-split duplicate groups remain in the new proposed splits.**

**Open for human decision.** At a looser Hamming of 8, ball still has 124
cross-split pairs, since the proposal targets 5. A threshold-8 variant is
generated at `datasets/derived/splits_zero_leakage_v1_hamming8/`
(ball 823/384/197) for comparison. I recommend 5 for ball and against 8 for
calibration, where clip id already does the work and 8 merges two pairs of
genuinely distinct clips, cutting the independent evaluation units from 18 to
16. **Calibration val/test rest on 3 clips each** — clip-disjoint is correct,
but three broadcasts is a thin basis for a gate, and this is an argument for
collection rather than for re-splitting.

---

## 4. Calibration collection, re-aimed at the measured deficit

### Index 29 is not rare because its end is under-collected

Cross-referencing landmark visibility against framing
(`scripts/audit_calibration_landmark_coverage.py`, joining each index to its
pitch coordinate from the same table `homography` consumes):

* **End coverage is already balanced.** The left end is in frame in 187 of 317
  images, the right end in 188. Framings split 127 left-end-only, 128
  right-end-only, 60 both-ends-wide, 2 centre-only.
* **The deficit is the near touchline, not either end.** At matched pitch
  positions differing only in which touchline they sit on:

  | far landmark | images | near landmark | images | ratio |
  |---|---:|---|---:|---:|
  | `left_corner_far` | 161 | `left_corner_near` | 27 | 5.96x |
  | `right_corner_far` | 156 | `right_corner_near` | 11 | **14.18x** |
  | `halfway_far` | 285 | `halfway_near` | 234 | 1.22x |

* The pattern is systematic across every goal-line and penalty-area landmark,
  not just the corners, and the gap widens with distance from the pitch centre
  line — 131 vs 111 where the goal area meets the goal line, 146 vs 73 where
  the penalty area meets it, 161 vs 27 at the corner. That is the signature of
  the near touchline falling out of the bottom of frame.
* **Conditioned on its own end being in frame**, index 5 is visible in 14.4%
  of those images and index 29 in only **5.9%**.

So index 29 is off-camera in the framings this camera actually produces. It is
cause (a): collecting more of the same broadcast framings, at either end,
cannot fix it. Note also that the square 960x960 export does not explain it —
squaring 16:9 footage would cut the left and right of frame, which is where
the *ends* are, and end coverage is fine.

### What the mirrored-pitch failure actually keys on

Measuring how far each image's visible landmark set is invariant under the
registry `flip_idx` — i.e. how well the mirrored pitch explains the same
points:

| flip-symmetry of visible set | images |
|---|---:|
| 1.00 (fully ambiguous) | 11 |
| 0.75–0.99 | 32 |
| 0.50–0.75 | 41 |
| 0.01–0.50 | 209 |
| 0.00 (unambiguous) | 24 |

Mean symmetry by framing: **both-ends-wide 0.746**, centre-only 1.00,
left-end-only 0.287, right-end-only 0.328.

The wide broadcast view is the *most* mirror-ambiguous framing in the set, and
43 of 317 images already sit at 0.75 or above. The previous plan gave two of
its eight buckets (48 of 192 images) to "both ends in wide broadcast view" —
that is, a quarter of the collection budget spent adding more of the case
where a mirrored fit is least penalised.

### Revised allocation

Total held at 192 new images (509 after collection) so the GPU-hour estimate
and the existing plan stay comparable. Only the allocation changes.

| bucket | images | requirement | acceptance |
|---|---:|---|---|
| `near_corner_right_in_frame` | 64 | index 29 visible | index 29 in >=90%; >=8 distinct clips |
| `near_corner_left_in_frame` | 48 | index 5 visible | index 5 in >=90%; >=6 distinct clips |
| `mirror_stress_wide_unseen_venue` | 48 | both ends, fixture/venue outside the existing 18 clips | >=50% carry an end-exclusive near-side anchor; >=6 clips |
| `venue_lighting_camera_diversity` | 32 | >=4 venues, >=3 lighting, varied height | no venue >40% of bucket; >=4 clips |
| ~~`end_asymmetric_alternating`~~ | ~~32~~ → **0** | retired — already supplied by existing data | curate the existing 167 instead |

Reasoning:

* **112 of 192 images (58%) now require a near-side corner in frame**, against
  zero before. The counts are sized from the deficit: index 29 at 11 needs 64
  frames at 90% yield to reach ~69; index 5 at 27 needs 48 to reach ~70. That
  brings both to the mid-tier where indices 9–12 already sit, instead of
  leaving one landmark the model has effectively never seen.
* **The two dedicated low-camera buckets are dropped.** A lower camera sees
  *less* of the near touchline, so 48 of the previous 192 images were aimed
  directly against the landmark the collection needs to recover. Camera-height
  variation is retained inside the diversity bucket rather than given a
  quarter of the budget.
* **Splitting every bucket by pitch end is dropped.** End coverage is already
  187/188; that dimension is not the deficit and does not need eight-way
  stratification.
* **Wide both-ends collection is re-scoped** to unseen fixtures with an
  asymmetry requirement, so it stresses the mirrored case instead of merely
  repeating it, and raised from 32 to 48 with the images freed below.
* Venue/lighting diversity stays at 32. It is real generalisation cover, but
  it does not address the measured failure and was crowding out what does.

### `end_asymmetric_alternating` is retired: the data already has it

Scoring every existing image against the bucket predicates
(`scripts/audit_calibration_landmark_coverage.py`) shows how much of each
bucket the current 317 images already supply, and — the part that matters —
from how many independent clips:

| bucket | existing images | clips supplying | most from one clip |
|---|---:|---:|---:|
| `end_asymmetric` (one end, symmetry <=0.35) | **167** | **18** | 27 |
| `mirror_stress` (both ends, symmetry >=0.75) | 31 | 9 | 9 |
| `near_corner_left` (index 5) | 27 | 10 | 7 |
| `near_corner_right` (index 29) | **11** | **4** | **7** |

`end_asymmetric_alternating` asked for 32 images of a framing the set already
holds **167** of, drawn from all 18 clips. Collecting 32 more would have added
a fifth copy of something already abundant. The model produced mirrored fits
*despite* that supply, which is direct evidence that a shortage of
low-symmetry examples was never the cause. The bucket is retired and its 32
images reallocated to the two real deficits. What it needs instead is
curation: tag the existing 167 so they are used deliberately rather than
incidentally.

The other three buckets cannot be filled from existing data:

* `near_corner_right` is the worst case in the entire audit. Eleven images,
  from **four clips, seven of them from a single clip**. Its effective scene
  count is closer to four than to eleven, and one broadcast supplies 64% of
  everything the model has ever seen of landmark 29.
* `near_corner_left` is better but still thin: 27 images over 10 clips.
* `mirror_stress` has 31 qualifying existing images, but the bucket requires
  *unseen* venues by definition, so the existing ones are disqualified by
  construction rather than by count.

### Clip diversity is now an explicit requirement

The first revision of this plan specified framing and said nothing about
scene count. That was a real hole: all 192 images could have been collected
from three or four new broadcasts, satisfying every bucket while reproducing
the exact effective-diversity failure this audit diagnosed, with new bucket
labels attached. The near-corner numbers above show the trap is not
hypothetical — landmark 29's existing "11 images" is really four scenes.

| requirement | value |
|---|---|
| minimum distinct **new** clips | **24** |
| preferred | 32–48 |
| maximum images from any one new clip | **8** |
| minimum clips per bucket | 6 (8 for `near_corner_right`) |

**Derivation, from the same evidence as the image counts.** 192 images capped
at 8 per clip forces at least 24 independent scenes. The cap of 8 is the
observed per-clip yield for these framings in the current data — the most any
single existing clip contributes is 7 images for index 5, 7 for index 29 and
9 for mirror-stress — so it is a ceiling on what one broadcast plausibly
supplies, not an arbitrary round number. It bounds any one clip at 4.2% of
the batch, against 24.3% for the worst existing clip (`42ba34`, 77 of 317).

**A "new clip" means a distinct broadcast of a distinct fixture.** Another
segment of a match already among the existing 18, or a second camera cut of
the same passage of play, does not count. Without that wording, `08fd33_0_*`
and `08fd33_2_*` — already two "segments" of one broadcast in the current
data — would qualify as two clips.

**24 is a floor, not a target.** 24 × 8 is exactly 192, so a 24-clip batch
puts every clip at the cap: the least diverse collection the policy permits.
Prefer more clips at 4–6 frames each. Frames within one clip are
near-duplicates of each other by construction, so the marginal value of the
eighth frame from a clip is far below the first frame of a new one.

The post-collection target becomes **509 images across at least 42 clips**
(18 existing + 24 new), against 317 across 18 today. Image count rises 61%;
independent scene count rises at least 133%. Given that clip-level leakage
rather than raw size is the likeliest driver of the generalisation failure,
the second number is the one to judge the collection by.

Every collected image must record venue, fixture, pitch end, camera height,
camera angle, lighting and clip id — the metadata whose absence forced the
current audit to report proxies. Whole clips only during splitting.

### The policy is enforced in code

`scripts/build_zero_leakage_splits.py` now carries a `CLIP_POLICY` for
calibration and checks every run against it, writing the result into
`summary.json` and printing failures to stderr. Run on today's data it
correctly fails all three checks:

```
CLIP POLICY FAIL [calibration] min_distinct_clips: required 42, got 18
CLIP POLICY FAIL [calibration] max_share_from_one_clip: required 0.05, got 0.2429
CLIP POLICY FAIL [calibration] min_clips_per_split: required 6, got 3
```

Reporting is the default and `--enforce-clip-policy` makes a failure exit
non-zero, for use on a collection batch or in CI. The check reports rather
than raises by default on purpose: the existing 317 images predate the policy
and would fail it, and a tool that refuses to audit the data it exists to
audit is useless. The `min_clips_per_split` floor of 6 is what makes the
acceptance gate meaningful — with today's 3 clips in test, one bad broadcast
moves the pass rate by a third.

This closes the regression path. A future batch that adds 192 calibration
images from four clips will fail the check at split-build time rather than
being discovered after another training run.

**Still requires human judgement.** Three things here are production work,
not analysis, and nothing in this audit simulates or substitutes for them:

* **Sourcing 24+ new broadcasts is the hard part of this plan, and it is
  entirely human.** The clip target is derived from the data; whether that
  many distinct fixtures are licensable and archivable is not a question the
  dataset can answer. If only a handful of new broadcasts are obtainable, say
  so before collection starts — the correct response is to re-scope the
  acceptance gate, not to hit 192 images out of six clips and call the
  diversity requirement met.
* **Whether near-corner framings exist in the available footage at all.** If
  corner and behind-goal angles cannot be sourced, indices 5 and 29 stay rare
  and the mapping table's coverage assumption should be revisited instead.
* **Curating the existing 167 `end_asymmetric` images.** The tagging is a
  human pass over images this audit only counted.

The 192/509 image target is unchanged and remains a minimum operational
specification derived from the 28-frame failure set. The 24-clip floor is new
and is the binding constraint of the two.

---

## Calibration diagnosis (unchanged)

Registry resolution: `field_datasets/2`, content is 32-keypoint pose data.
317 images, 255/34/28, all 960x960. Label visibility 6–22 landmarks per image.
`scripts/audit_calibration_dataset.py` and its artifacts were regenerated by
the script this time rather than by hand; `width_height` is now populated.

The prior end-to-end real-world result remains the controlling diagnosis:
28 frames, median TRUE positional error 2.004 m, mean 14.597 m, max 58.770 m,
and only 1/28 frames clearing `HOMOGRAPHY_CONFIDENCE_MIN`. The gap between a
2.0 m median and a 58.8 m maximum is a subset of frames failing
catastrophically rather than uniform imprecision, which is what both the
mirror-ambiguity and near-side-extrapolation findings above would predict. The
table and `flip_idx` checks are separate and remain intact. A fresh per-image
model validation should be rerun with the existing validation scripts and the
same TRUE-error calculation.

## Splits, augmentation, and training plan (planning only)

Group-aware splitting is now implemented rather than described. For ball,
retain small-object settings (1280 imgsz, modest scale jitter, mosaic closed
early, tiling/crops for tiny balls) after human cleaning; do not inherit
player augmentation. For calibration, keep mosaic off, use symmetry-aware
flips with the registry `flip_idx`, modest photometric/camera perturbations,
and treat real end/venue/camera collection as non-negotiable. Horizontal flips
create symmetry pairs but do not create venues, camera heights, or broadcast
texture — and given that 43 images are already flip-ambiguous, flip
augmentation cannot substitute for the near-side collection.

Post-review starting configurations: ball 1280px, batch 6, 120–180 epochs,
cosine/one-cycle schedule, recall-first inference conf 0.10 and IoU 0.30;
calibration 960px, batch 8, 100–150 epochs, mosaic off, conf 0.30, keypoint
floor 0.50. Plans only.

Acceptance gates before replacing checkpoints:

* Ball: clean held-out test precision >=0.90, recall >=0.93, mAP50 >=0.95,
  mAP50-95 >=0.70; on a held-out orphan/control sample, the trained model's
  >=0.5 firing rate must be <=5% and materially below the COCO detector, with
  manually reviewed false positives <=5%.
* Calibration: on unseen real broadcast footage, median TRUE error <=1.0 m,
  mean <=2.5 m, max <=8.0 m, and >=90% of frames must clear
  `HOMOGRAPHY_CONFIDENCE_MIN` end-to-end. Any mirrored fit that clears the gate
  is a hard failure; all-point reprojection scoring remains load-bearing.

After approval, estimate 2–4 GPU-hours for a cleaned ball run and 1–3
GPU-hours for a 509-image calibration run on the recorded RTX 5070 Ti-class
environment. Note that the installed Torch in this venv is CPU-only
(2.13.0+cpu); a CUDA build is required before any training run and its absence
is another reason nothing was started here.

---

## Artifacts and how to reproduce them

`datasets/` is gitignored (`.gitignore:12`), so the triage table and the split
manifests are **local build output and will not arrive with a commit or a
PR**. The scripts that produce them are tracked, which is the intended
workflow — the artifacts are regenerated, not shipped. A reviewer on another
machine runs:

```
python -m scripts.audit_ball_dataset \
    --out datasets/derived/Ball_dataset_cleaned_v1 \
    --trained-model models/yolo/ball_v1/weights/best.pt \
    --coco-model  models/yolo/yolov8s.pt \
    --trained-imgsz 1280 --coco-imgsz 640
python -m scripts.build_zero_leakage_splits ball calibration --hamming 5
python -m scripts.audit_calibration_dataset
python -m scripts.audit_calibration_landmark_coverage
```

After a calibration collection batch lands, gate it on clip diversity:

```
python -m scripts.build_zero_leakage_splits calibration --enforce-clip-policy
```

Tracked (in git): the four scripts above, this report, and
`docs/dataset_audit/calibration_coverage_v1/`.

Untracked (regenerate locally): `datasets/derived/Ball_dataset_cleaned_v1/`
and `datasets/derived/splits_zero_leakage_v1{,_hamming8}/`.

**If the human labelling review happens on a different machine, `triage.csv`
has to be transferred out of band or the review will start from an empty
directory.** The full ball pass takes about 13 minutes on this CPU-only
install.

## Go/no-go

| item | status |
|---|---|
| 1. Environment + detector triage | **Resolved.** Root cause was that the script had never been run, not a broken venv. All 1,404 rows now carry real trained, COCO and heuristic signals; 853/853 unlabeled rows populated. |
| 2. Train-manifest discrepancy | **Resolved.** Same 824 files; 180 labelled / 644 unlabelled is correct and was already correct on 2026-08-12. 698/126 was never correct. |
| 3. Cross-split leakage | **Resolved.** 127/127 ball groups and 17/17 calibration clips confined to one split; independently verified zero cross-split duplicates at the stated threshold. |
| 4. Collection plan | **Revised twice.** 112 of 192 images now require a near-side corner; low-camera buckets dropped; `end_asymmetric` retired as already-supplied (167 existing images) and its budget reallocated. |
| 5. Clip diversity | **Added.** Minimum 24 new clips, max 8 images per clip, enforced by `CLIP_POLICY` in the split builder (`--enforce-clip-policy`). Post-collection target is 509 images across >=42 clips. |

Carried forward for human decision, none of them blocking items 1–4:

* `docs/pipeline_architecture.md:319-323` states a figure this audit
  disproves; I did not edit it.
* Ball near-duplicate threshold 5 vs 8 (both variants generated).
* Calibration val/test rest on 3 broadcast clips each — correct, but thin;
  the collection's 24-clip floor is what lifts it to the 6-clip policy minimum.
* Sourcing 24+ genuinely new broadcasts, and confirming near-corner framings
  exist in the available footage at all.
* Whether near-corner framings are obtainable from the available footage.
* All 853 unlabeled images and all 551 existing positives still need
  per-image human sign-off. Nothing was auto-labelled.

**Training and registry promotion remain blocked pending explicit human
approval.** No GPU training was started, no checkpoint was promoted, no
production inference or pipeline module was modified, and the proposed splits
were written to `datasets/derived/` — deliberately not to `runs/_data/`, which
is what the trainers read — so nothing here can be picked up by a training run
without a human moving it.

**Do you approve the derived ball triage artifact, the zero-leakage split
proposal, and the revised calibration collection specification — including
the 24-clip minimum and the enforced clip policy — for human review, and only
then planned training?**
