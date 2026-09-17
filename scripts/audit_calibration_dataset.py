"""Coverage and leakage audit for the registered 32-keypoint calibration set.

The source export does not carry explicit venue/end/camera metadata.  This
script therefore reports measurable proxies (source stem tokens, visible
landmark subsets, image dimensions, and SHA-1 duplicate groups) and marks
camera/end/venue as ``human_metadata_required`` rather than inventing labels.
It also emits a quantified collection specification from the observed 317
images and the existing real-world validation result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
from configs import registry as R  # noqa: E402


def _sha1(p: Path) -> str: return hashlib.sha1(p.read_bytes()).hexdigest()


# Revised 2026-08-13. The previous eight-bucket plan split every bucket by
# pitch end and gave 24 images to each. `scripts/audit_calibration_landmark_
# coverage.py` shows that end coverage is ALREADY balanced (left end in frame
# in 187 of 317 images, right end in 188), so splitting by end buys nothing
# on a dimension that is not the deficit. The measured deficit is the NEAR
# touchline: at matched pitch positions the far corner is visible 5.96x more
# often than the near corner on the left (161 vs 27) and 14.18x more on the
# right (156 vs 11), while at the halfway line near/far is nearly even
# (285 vs 234, 1.22x). Index 29 is visible in only 5.9% of the images where
# its own end is already in frame -- it is a camera-geometry deficit, not an
# end-coverage deficit, so more of the same framings cannot fix it.
#
# Two dedicated low-camera buckets were also dropped: a lower camera sees
# LESS of the near touchline, so they worked directly against the landmark
# the collection is meant to recover. Camera-height variation is retained
# inside the diversity bucket instead.
#
# Total stays at 192 (509 after collection) so the GPU-hour estimate and the
# existing plan remain comparable; only the allocation changes.
#
# CLIP DIVERSITY (added 2026-08-13, second revision). The first revision
# specified framing but not scene count, which left the plan satisfiable out
# of three or four new broadcasts -- reproducing the 18-clip effective-
# diversity problem under new bucket labels. Per-clip measurement also
# retired one bucket: `end_asymmetric_alternating` asked for 32 images of a
# framing the set already holds 167 of, spread across all 18 clips. More of
# it was never the missing thing; those 32 images are reallocated to the two
# deficits that are real. See the report for the full derivation.
_COLLECTION_SPEC = {
    "revision": "2026-08-13 rev2; landmark deficit + clip diversity",
    "new_images_total": 192,
    "target_total_after_collection": 509,
    "clip_diversity": {
        "min_new_clips": 24,
        "preferred_new_clips": "32-48",
        "max_images_per_new_clip": 8,
        "min_clips_per_bucket": 6,
        "floor_is_not_a_target":
            "24 clips x 8 images is exactly 192, so collecting 24 clips means "
            "every clip sits at the cap -- the least diverse batch the policy "
            "permits. Prefer more clips at 4-6 frames each; scene count is the "
            "quantity that was actually missing, and frames within one clip "
            "are near-duplicates of each other by construction.",
        "new_clip_definition":
            "a distinct broadcast of a distinct fixture. Another segment of a "
            "match already among the existing 18 clips does NOT count as new, "
            "and neither does a second camera cut of the same passage of play.",
        "rationale":
            "192 images capped at 8 per clip forces >=24 independent scenes. "
            "8 is the observed per-clip yield for these framings in the "
            "current set (max 7 for index 5, 7 for index 29, 9 for "
            "mirror-stress), so it is a ceiling on what one broadcast "
            "plausibly supplies rather than an arbitrary cut. It bounds any "
            "one clip at 4.2% of the batch, against 24.3% for the worst "
            "existing clip (42ba34, 77 of 317).",
        "enforced_by": "scripts/build_zero_leakage_splits.py::CLIP_POLICY "
                       "(--enforce-clip-policy)",
    },
    "buckets": [
        {"name": "near_corner_right_in_frame", "images": 64,
         "requires": "landmark index 29 (right_corner_near) visible",
         "acceptance": "index 29 in >=90% of accepted frames; >=8 distinct clips",
         "rationale": "index 29 at 11/317 and those 11 come from only 4 clips, "
                      "7 of them from one -- the worst deficit in both image "
                      "count and scene count; 64 frames lifts it to ~69"},
        {"name": "near_corner_left_in_frame", "images": 48,
         "requires": "landmark index 5 (left_corner_near) visible",
         "acceptance": "index 5 in >=90% of accepted frames; >=6 distinct clips",
         "rationale": "index 5 at 27/317 across 10 clips; 48 frames lifts it to ~70"},
        {"name": "mirror_stress_wide_unseen_venue", "images": 48,
         "requires": "both ends in frame, fixture and venue outside the existing 18 clips",
         "acceptance": ">=50% carry one end-exclusive near-side anchor; >=6 distinct clips",
         "rationale": "wide both-ends framing has mean flip-symmetry 0.746, the "
                      "highest of any framing, and 43 of 317 images already sit "
                      "at >=0.75; this is where a mirrored fit is unpenalised. "
                      "The existing 31 qualifying images cannot serve this "
                      "bucket -- they are by definition the already-seen venues"},
        {"name": "venue_lighting_camera_diversity", "images": 32,
         "requires": ">=4 distinct venues, >=3 lighting conditions, "
                     "camera height varied incl. some low",
         "acceptance": "no single venue >40% of the bucket; >=4 distinct clips",
         "rationale": "genuine generalisation cover; does not address the "
                      "landmark deficit, so it is not given more"},
    ],
    "buckets_retired": [
        {"name": "end_asymmetric_alternating", "was_images": 32, "now_images": 0,
         "reason": "167 existing images already satisfy it (exactly one end in "
                   "frame, flip-symmetry <=0.35), drawn from all 18 clips, "
                   "against a target of 32. The model failed at mirroring "
                   "despite that supply, so more of it was not the fix.",
         "action": "curate and tag the existing 167 rather than collect; "
                   "ensure train retains them across all available clips"},
    ],
    "record_per_image": ["venue", "fixture", "pitch_end", "camera_height",
                         "camera_angle", "lighting", "clip_id"],
    "splitting_rule": "whole clips only; see scripts/build_zero_leakage_splits.py",
}


def _row(src: Path, split: str, image: Path) -> dict:
    lab = src / split / "labels" / f"{image.stem}.txt"
    vals = lab.read_text(encoding="utf-8").split() if lab.exists() else []
    visible = [i for i in range(32) if len(vals) >= 5 + 32 * 3 and int(float(vals[5 + 3 * i + 2])) == 2]
    # Names often encode source clip/frame tokens; preserve them as evidence,
    # never as a claimed venue or pitch-end class.
    tokens = re.findall(r"[A-Za-z]+|\d+", image.stem)
    try:
        from PIL import Image
        with Image.open(image) as im:
            wh = f"{im.width}x{im.height}"
    except Exception as exc:
        wh = f"unavailable={exc}"
    return {"split": split, "image": str(image.resolve()), "sha1": _sha1(image),
            "width_height": wh, "label": str(lab.resolve()),
            "visible_count": len(visible), "visible_indices": ",".join(map(str, visible)),
            "stem_tokens": "|".join(tokens), "pitch_end": "human_metadata_required",
            "camera_angle": "human_metadata_required", "venue": "human_metadata_required",
            "review_notes": ""}


def build(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for src in R.dataset_source_dirs("calibration"):
        for split in ("train", "valid", "test"):
            img_dir = src / split / "images"
            if img_dir.is_dir():
                rows.extend(_row(src, split, p) for p in sorted(img_dir.iterdir()) if p.suffix.lower() in R.IMAGE_SUFFIXES)
    by_hash = defaultdict(list)
    for r in rows: by_hash[r["sha1"]].append(r)
    dup_groups = sum(1 for g in by_hash.values() if len(g) > 1)
    index_counts = Counter(i for r in rows for i in r["visible_indices"].split(",") if i)
    summary = {"images": len(rows), "splits": dict(Counter(r["split"] for r in rows)),
               "duplicate_groups": dup_groups, "cross_split_duplicate_groups": sum(1 for g in by_hash.values() if len({x["split"] for x in g}) > 1),
               "visible_keypoint_counts": dict(sorted(Counter(r["visible_count"] for r in rows).items())),
               "keypoint_index_coverage": dict(sorted(index_counts.items(), key=lambda x: int(x[0]))),
               "metadata_status": "pitch_end/camera_angle/venue absent from source; human annotation required",
               "real_world_validation_reference": {"images": 28, "median_true_error_m": 2.004, "mean_true_error_m": 14.597, "max_true_error_m": 58.770, "gate_pass": "1/28"},
               "collection_spec": _COLLECTION_SPEC}
    fields = list(rows[0]) if rows else []
    with (out_dir / "calibration_coverage.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text("# Calibration coverage audit\n\nMetadata fields are intentionally human-review placeholders; the source labels provide keypoint visibility only. No images or labels were changed. See `summary.json` for the quantified collection specification.\n", encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "dataset_audit" / "calibration_coverage_v1")
    args = ap.parse_args(); print(json.dumps(build(args.out), indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
