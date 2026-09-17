"""Why are some pitch landmarks rare, and what collection would fix it?

The coverage audit established that landmark visibility is very uneven
(indices 13-15 in ~285-290 of 317 images, index 29 in 11).  A raw histogram
does not say what to COLLECT, because it does not distinguish two very
different causes:

  (a) the landmark is off-camera in almost every framing the broadcast
      camera physically produces -- collecting more of the same framings
      cannot fix it, and
  (b) the landmark is visible only from framings that happen to be
      under-collected -- more of THOSE framings fixes it directly.

This script separates the two by joining each landmark index to its pitch
coordinate (ai/computer_vision/tactical_analysis/pitch_keypoints.py, the
same table homography uses) and asking, per image, which pitch REGION the
framing actually covers.  It then reports each rare landmark's hit rate
*conditioned on its own region being in frame*, which is the discriminating
measurement.

Reads labels only.  Writes a report; changes no dataset and no model.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
from ai.computer_vision.tactical_analysis.pitch_keypoints import (  # noqa: E402
    FLIP_IDX,
    PITCH_KEYPOINT_NAMES,
    PITCH_KEYPOINTS_32,
    L,
    W,
)
from configs import registry as R  # noqa: E402

RARE_THRESHOLD = 30

# Must stay in step with CLIP_PATTERNS["calibration"] in
# scripts/build_zero_leakage_splits.py -- both read the same export naming.
CLIP_PATTERN = r"^([0-9a-fA-F]+)_"

# Bucket membership tests, expressed against measurable label content so the
# same predicate can score an existing image and an incoming collected one.
MIRROR_STRESS_MIN_SYMMETRY = 0.75
END_ASYMMETRIC_MAX_SYMMETRY = 0.35

# Region membership straight from the coordinate table, so these cannot
# drift from what homography actually consumes.
LEFT_END = {i for i, (x, _) in PITCH_KEYPOINTS_32.items() if x < L / 3}
RIGHT_END = {i for i, (x, _) in PITCH_KEYPOINTS_32.items() if x > 2 * L / 3}
CENTRE = set(PITCH_KEYPOINTS_32) - LEFT_END - RIGHT_END
# "Near" = the touchline the camera sits on (y -> W); "far" = y -> 0.
NEAR_SIDE = {i for i, (_, y) in PITCH_KEYPOINTS_32.items() if y > W * 2 / 3}
FAR_SIDE = {i for i, (_, y) in PITCH_KEYPOINTS_32.items() if y < W / 3}


def _visible(label: Path) -> set[int]:
    vals = label.read_text(encoding="utf-8").split()
    if len(vals) < 5 + 32 * 3:
        return set()
    return {i for i in range(32) if int(float(vals[5 + 3 * i + 2])) == 2}


def _records() -> list[dict]:
    out = []
    for src in R.dataset_source_dirs("calibration"):
        for split in ("train", "valid", "test"):
            img_dir = src / split / "images"
            if not img_dir.is_dir():
                continue
            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() not in R.IMAGE_SUFFIXES:
                    continue
                lab = src / split / "labels" / f"{img.stem}.txt"
                if not lab.exists():
                    continue
                m = re.match(CLIP_PATTERN, img.name.split(".rf.")[0])
                out.append({"split": split, "image": img, "visible": _visible(lab),
                            "clip": m.group(1) if m else "unmatched"})
    return out


def _framing(vis: set[int]) -> str:
    left, right = bool(vis & LEFT_END), bool(vis & RIGHT_END)
    if left and right:
        return "both_ends_wide"
    if left:
        return "left_end_only"
    if right:
        return "right_end_only"
    return "centre_only"


def build(out_dir: Path) -> dict:
    recs = _records()
    n = len(recs)
    counts = Counter(i for r in recs for i in r["visible"])

    rare = sorted(i for i in range(32) if counts.get(i, 0) < RARE_THRESHOLD)

    # The discriminating measurement: for each landmark, how often is it
    # visible GIVEN that its own end is already in frame? A low conditional
    # rate means (a) framing/camera geometry; a high conditional rate with a
    # small denominator means (b) that end is simply under-collected.
    conditional = {}
    for i in range(32):
        region = LEFT_END if i in LEFT_END else (RIGHT_END if i in RIGHT_END else CENTRE)
        in_region = [r for r in recs if r["visible"] & region]
        hits = sum(1 for r in in_region if i in r["visible"])
        conditional[i] = {
            "name": PITCH_KEYPOINT_NAMES[i],
            "pitch_xy": list(PITCH_KEYPOINTS_32[i]),
            "visible_in": counts.get(i, 0),
            "region": "left_end" if i in LEFT_END else ("right_end" if i in RIGHT_END else "centre"),
            "side": "near" if i in NEAR_SIDE else ("far" if i in FAR_SIDE else "middle"),
            "region_in_frame_images": len(in_region),
            "visible_given_region_in_frame": hits,
            "conditional_rate": round(hits / len(in_region), 4) if in_region else None,
            "flip_partner": FLIP_IDX[i],
            "flip_partner_visible_in": counts.get(FLIP_IDX[i], 0),
        }

    framings = Counter(_framing(r["visible"]) for r in recs)

    # Near/far asymmetry at matched pitch positions. These pairs share an
    # x-coordinate and differ only in which touchline they sit on, so a gap
    # between them isolates the camera-side effect from the end effect.
    matched_pairs = [(0, 5), (24, 29), (13, 16)]
    side_gap = [{
        "far_index": f, "far_name": PITCH_KEYPOINT_NAMES[f], "far_visible_in": counts.get(f, 0),
        "near_index": nr, "near_name": PITCH_KEYPOINT_NAMES[nr], "near_visible_in": counts.get(nr, 0),
        "ratio_far_over_near": (round(counts.get(f, 0) / counts.get(nr, 0), 2)
                                if counts.get(nr, 0) else None),
    } for f, nr in matched_pairs]

    end_totals = {
        "left_end_in_frame": sum(1 for r in recs if r["visible"] & LEFT_END),
        "right_end_in_frame": sum(1 for r in recs if r["visible"] & RIGHT_END),
        "centre_in_frame": sum(1 for r in recs if r["visible"] & CENTRE),
        "near_side_any": sum(1 for r in recs if r["visible"] & NEAR_SIDE),
        "far_side_any": sum(1 for r in recs if r["visible"] & FAR_SIDE),
    }

    # Co-occurrence for the rarest landmark: what ELSE is in frame when it
    # appears? This describes the framing that would have to be collected.
    cooccur = {}
    for i in rare:
        with_i = [r for r in recs if i in r["visible"]]
        c = Counter(j for r in with_i for j in r["visible"] if j != i)
        cooccur[i] = {
            "name": PITCH_KEYPOINT_NAMES[i], "n_images": len(with_i),
            "top_cooccurring": [{"index": j, "name": PITCH_KEYPOINT_NAMES[j],
                                 "n": k, "rate": round(k / len(with_i), 3)}
                                for j, k in c.most_common(8)] if with_i else [],
            "median_visible_count": (sorted(len(r["visible"]) for r in with_i)[len(with_i) // 2]
                                     if with_i else None),
        }

    # Mirror ambiguity. A homography fit is free to land on the flipped pitch
    # when the VISIBLE landmark set is (near-)invariant under FLIP_IDX: the
    # mirrored hypothesis then explains the same points about as well. This
    # is the measurement that speaks to the observed mirrored-end failure,
    # as opposed to end coverage, which is already balanced.
    def flip(s: set[int]) -> set[int]:
        return {FLIP_IDX[i] for i in s}

    sym_rows = []
    for r in recs:
        v = r["visible"]
        if not v:
            continue
        overlap = len(v & flip(v)) / len(v)
        sym_rows.append({"framing": _framing(v), "symmetry": overlap,
                         "n_visible": len(v)})
    bands = Counter()
    for s in sym_rows:
        if s["symmetry"] >= 0.99:
            bands["1.00_fully_flip_ambiguous"] += 1
        elif s["symmetry"] >= 0.75:
            bands["0.75-0.99_high"] += 1
        elif s["symmetry"] >= 0.50:
            bands["0.50-0.75_moderate"] += 1
        elif s["symmetry"] > 0.0:
            bands["0.01-0.50_low"] += 1
        else:
            bands["0.00_unambiguous"] += 1
    sym_by_framing = defaultdict(list)
    for s in sym_rows:
        sym_by_framing[s["framing"]].append(s["symmetry"])

    # Per-clip bucket coverage. The image-count deficit says how many frames
    # to collect; this says how many INDEPENDENT SCENES currently supply each
    # bucket. A bucket fed by one or two clips is not covered, however many
    # frames it holds -- that is the same effective-diversity trap the 18-clip
    # finding exposed, one level down.
    def _sym(v: set[int]) -> float:
        return len(v & {FLIP_IDX[i] for i in v}) / len(v) if v else 0.0

    def _buckets(v: set[int]) -> set[str]:
        b = set()
        if 5 in v:
            b.add("near_corner_left")
        if 29 in v:
            b.add("near_corner_right")
        f = _framing(v)
        if f == "both_ends_wide" and _sym(v) >= MIRROR_STRESS_MIN_SYMMETRY:
            b.add("mirror_stress")
        if f in ("left_end_only", "right_end_only") and _sym(v) <= END_ASYMMETRIC_MAX_SYMMETRY:
            b.add("end_asymmetric")
        return b

    per_clip: dict[str, dict] = {}
    for r in recs:
        c = per_clip.setdefault(r["clip"], {"images": 0, "splits": set(), "buckets": Counter()})
        c["images"] += 1
        c["splits"].add(r["split"])
        for b in _buckets(r["visible"]):
            c["buckets"][b] += 1

    bucket_names = ["near_corner_left", "near_corner_right", "mirror_stress", "end_asymmetric"]
    bucket_supply = {
        b: {
            "images_available": sum(c["buckets"].get(b, 0) for c in per_clip.values()),
            "clips_supplying": sum(1 for c in per_clip.values() if c["buckets"].get(b, 0)),
            "max_from_one_clip": max((c["buckets"].get(b, 0) for c in per_clip.values()), default=0),
        } for b in bucket_names
    }

    summary = {
        "images": n,
        "rare_threshold": RARE_THRESHOLD,
        "clips": {
            "pattern": CLIP_PATTERN,
            "distinct_clips": len(per_clip),
            "images_per_clip": {k: v["images"] for k, v in sorted(per_clip.items())},
            "clip_size_min": min(c["images"] for c in per_clip.values()),
            "clip_size_max": max(c["images"] for c in per_clip.values()),
            "clips_spanning_registry_splits": sum(1 for c in per_clip.values() if len(c["splits"]) > 1),
        },
        "existing_bucket_supply": bucket_supply,
        "per_clip_bucket_counts": {
            k: dict(v["buckets"]) for k, v in sorted(per_clip.items())},
        "mirror_ambiguity": {
            "note": "share of visible landmarks that map onto another VISIBLE "
                    "landmark under FLIP_IDX; 1.0 means the mirrored pitch "
                    "explains the same point set",
            "bands": dict(bands),
            "mean_symmetry_by_framing": {
                k: round(sum(v) / len(v), 4) for k, v in sorted(sym_by_framing.items())},
            "images_at_or_above_0.75": sum(1 for s in sym_rows if s["symmetry"] >= 0.75),
        },
        "rare_indices": rare,
        "index_detail": conditional,
        "framing_distribution": dict(framings),
        "end_and_side_totals": end_totals,
        "near_far_matched_pairs": side_gap,
        "rare_cooccurrence": cooccur,
        "visible_count_distribution": dict(sorted(Counter(len(r["visible"]) for r in recs).items())),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "landmark_coverage.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "docs" / "dataset_audit" / "calibration_coverage_v1")
    args = ap.parse_args()
    s = build(args.out)

    print(f"images={s['images']}  rare(<{RARE_THRESHOLD})={s['rare_indices']}")
    print("\nframing distribution:", s["framing_distribution"])
    print("end/side totals:", s["end_and_side_totals"])
    print("\nnear vs far at matched pitch positions:")
    for p in s["near_far_matched_pairs"]:
        print(f"  {p['far_name']:<28} {p['far_visible_in']:>4}   vs   "
              f"{p['near_name']:<28} {p['near_visible_in']:>4}   ratio={p['ratio_far_over_near']}")
    c = s["clips"]
    print(f"\nclips: {c['distinct_clips']} distinct, "
          f"{c['clip_size_min']}-{c['clip_size_max']} images each, "
          f"{c['clips_spanning_registry_splits']} spanning registry splits")
    print("\nexisting supply for the prioritised buckets:")
    print(f"  {'bucket':<20} {'images':>7} {'clips':>7} {'max/clip':>9}")
    for b, d in s["existing_bucket_supply"].items():
        print(f"  {b:<20} {d['images_available']:>7} {d['clips_supplying']:>7} "
              f"{d['max_from_one_clip']:>9}")

    m = s["mirror_ambiguity"]
    print("\nmirror ambiguity (flip-invariance of the visible set):")
    print("  bands:", m["bands"])
    print("  mean symmetry by framing:", m["mean_symmetry_by_framing"])
    print("  images at/above 0.75:", m["images_at_or_above_0.75"])
    print("\nrare landmarks, conditioned on their own end being in frame:")
    for i in s["rare_indices"]:
        d = s["index_detail"][str(i) if str(i) in s["index_detail"] else i]
        print(f"  {i:>2} {d['name']:<28} region={d['region']:<9} side={d['side']:<6} "
              f"visible={d['visible_in']:>4}  region_in_frame={d['region_in_frame_images']:>4}  "
              f"conditional={d['conditional_rate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
