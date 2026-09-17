"""Propose train/val/test splits in which no duplicate group is ever split.

WHY THIS EXISTS
    ``scripts/dataset_dedupe.py`` removes EXACT (SHA-1) duplicates and keeps
    one copy per digest, resolving byte-identical leakage.  It does not see
    NEAR-duplicates: two adjacent frames from the same broadcast clip, or the
    same source frame re-encoded at a different quality by two Roboflow
    exports, differ in bytes and therefore survive as separate images that
    can land in different splits.  Evaluating on those reports memorisation
    just as surely as an exact duplicate does.

    Measured on ball: 500 exact duplicate groups, 114 of them cross-split
    under the registry-defined splits (docs/dataset_audit/ball.md).

WHAT IT DOES
    1. Hashes every image twice: SHA-1 of the bytes (exact) and a 64-bit
       difference hash (perceptual, near-duplicate).
    2. Unions images into groups: same SHA-1, or dHash Hamming distance
       <= --hamming.  Grouping is transitive (connected components), because
       "A is a near-duplicate of B and B of C" still means C must not be
       evaluated against a model trained on A.
    3. Assigns every GROUP -- never an image -- to exactly one split, using
       a deterministic greedy fill against target proportions, stratified so
       that labelled images distribute proportionally too.
    4. Writes proposed manifests and verifies zero cross-split groups.

    It NEVER writes into runs/_data (the directory the trainers read) and it
    never touches a source image or label.  Output is a proposal under
    datasets/derived/ for human review.

Usage
    python -m scripts.build_zero_leakage_splits ball calibration
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
from configs import registry as R  # noqa: E402

MAX_PATH = 260

# Filename-derived clip ids, per dataset. Roboflow exports as
# "<original_stem>.rf.<content_hash>.<ext>", so <original_stem> survives.
#
#   calibration: all 317 original stems are distinct and follow
#       "<clip>_<segment>_<frame>" -- 18 clips over 317 images. Frames of one
#       broadcast clip must never be split; 17 of the 18 clips currently are.
#
#   ball: deliberately absent. Its original stems collide across the six
#       independent exports (generic names like "-2-_jpeg" and "4_jpg" appear
#       in all six), and 58.6% of same-stem pairs are >20 dHash bits apart --
#       i.e. genuinely different images. Grouping ball by stem would merge
#       unrelated photos and overstate the leakage fix.
CLIP_PATTERNS = {"calibration": r"^([0-9a-fA-F]+)_"}

# Clip-diversity policy for calibration, derived in
# docs/dataset_audit/cv_dataset_preparation_report.md.
#
# The 317-image set spans 18 clips and one clip supplies 77 images (24.3% of
# the data). Image count was never the binding constraint on generalisation --
# independent scene count was. These thresholds exist so a future collection
# batch that satisfies every framing bucket out of three or four new clips is
# caught here rather than after another training run.
#
# `min_distinct_clips` is 42 = the 18 that exist plus the 24 new ones the
# collection spec requires. The per-clip cap is expressed as a share so it
# scales with the dataset instead of needing a rewrite at 509 images.
CLIP_POLICY = {
    "calibration": {
        "min_distinct_clips": 42,
        "max_share_from_one_clip": 0.05,
        "min_clips_per_split": 6,
    }
}


def _targets_from_registry(records: list[dict]) -> dict:
    """Split proportions measured from the dataset's OWN registry splits.

    Deliberately not a fixed constant: ball currently sits near 59/27/14
    while calibration sits near 80/11/9, and forcing one dataset's ratio
    onto the other would quietly redesign the split while claiming only to
    fix leakage.  This tool changes WHICH images are in each split, not how
    big the splits are -- anything else is a separate decision for a human.
    """
    # Same test > valid > train priority scripts/dataset_dedupe.py uses, so
    # the baseline is the 824/380/200 the current manifests actually contain.
    priority = {"test": 0, "valid": 1, "train": 2}
    chosen: dict[str, str] = {}
    for r in records:
        prev = chosen.get(r["sha1"])
        if prev is None or priority[r["registry_split"]] < priority[prev]:
            chosen[r["sha1"]] = r["registry_split"]
    counts = Counter("val" if s == "valid" else s for s in chosen.values())
    total = sum(counts.values()) or 1
    return {k: counts.get(k, 0) / total for k in ("train", "val", "test")}


def _readable(p: Path) -> str:
    s = str(p)
    if sys.platform == "win32" and len(s) > MAX_PATH and not s.startswith("\\\\?\\"):
        return "\\\\?\\" + s
    return s


# ----------------------------------------------------------------------
# Hashing
# ----------------------------------------------------------------------
def _dhash(path: Path, size: int = 8) -> int | None:
    """64-bit difference hash: grayscale, resize to 9x8, compare adjacent
    pixels along each row. Robust to re-encoding and small compression
    differences, which is exactly what separates two exports of one frame."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
            px = list(im.tobytes())  # mode "L" -> one byte per pixel, row-major
        bits = 0
        for row in range(size):
            base = row * (size + 1)
            for col in range(size):
                bits = (bits << 1) | int(px[base + col] > px[base + col + 1])
        return bits
    except Exception:
        return None


def _inventory(name: str) -> list[dict]:
    records = []
    for source in R.dataset_source_dirs(name):
        for split in ("train", "valid", "test"):
            image_dir = source / split / "images"
            label_dir = source / split / "labels"
            if not image_dir.is_dir():
                continue
            for image in sorted(image_dir.iterdir()):
                if image.suffix.lower() not in R.IMAGE_SUFFIXES:
                    continue
                label = label_dir / f"{image.stem}.txt"
                rows = 0
                if label.exists():
                    rows = len([x for x in label.read_text(encoding="utf-8").splitlines()
                                if x.strip()])
                records.append({
                    "source": source.name, "registry_split": split, "image": image,
                    "sha1": hashlib.sha1(Path(_readable(image)).read_bytes()).hexdigest(),
                    "dhash": _dhash(image), "labeled": rows > 0, "label_rows": rows,
                })
    return records


# ----------------------------------------------------------------------
# Grouping
# ----------------------------------------------------------------------
class _Union:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _clip_id(record: dict, pattern: str) -> str | None:
    m = re.match(pattern, record["image"].name.split(".rf.")[0])
    return m.group(1) if m else None


def _clip_stats(records: list[dict], dataset: str) -> dict:
    pattern = CLIP_PATTERNS.get(dataset)
    if not pattern:
        return {"pattern": None,
                "reason": "filename stems collide across exports; see CLIP_PATTERNS"}
    splits = defaultdict(set)
    sizes: Counter = Counter()
    for r in records:
        cid = _clip_id(r, pattern)
        if cid is None:
            continue
        splits[cid].add(r["registry_split"])
        sizes[cid] += 1
    return {
        "pattern": pattern,
        "clips": len(splits),
        "images_matched": sum(sizes.values()),
        "clips_crossing_registry_splits": sum(1 for v in splits.values() if len(v) > 1),
        "clip_size_min": min(sizes.values(), default=0),
        "clip_size_max": max(sizes.values(), default=0),
    }


def _check_clip_policy(records: list[dict], dataset: str,
                       by_split: dict[str, list[dict]]) -> dict:
    """Score the dataset against CLIP_POLICY and return findings.

    Reports rather than raises by default: the existing 317 images predate
    the policy and would fail it, and failing the audit tool on the data it
    is meant to audit helps nobody. `--enforce-clip-policy` turns a FAIL into
    a non-zero exit for use on a collection batch or in CI.
    """
    policy = CLIP_POLICY.get(dataset)
    pattern = CLIP_PATTERNS.get(dataset)
    if not policy or not pattern:
        return {"policy": None, "reason": "no clip policy declared for this dataset"}

    total = sum(len(v) for v in by_split.values()) or 1
    sizes: Counter = Counter()
    per_split: dict[str, set] = defaultdict(set)
    for split, rows in by_split.items():
        for r in rows:
            cid = _clip_id(r, pattern)
            if cid is None:
                continue
            sizes[cid] += 1
            per_split[split].add(cid)

    biggest, biggest_n = (sizes.most_common(1) or [(None, 0)])[0]
    min_split_clips = min((len(v) for v in per_split.values()), default=0)
    checks = [
        {"check": "min_distinct_clips", "required": policy["min_distinct_clips"],
         "actual": len(sizes), "pass": len(sizes) >= policy["min_distinct_clips"]},
        {"check": "max_share_from_one_clip", "required": policy["max_share_from_one_clip"],
         "actual": round(biggest_n / total, 4), "worst_clip": biggest,
         "pass": (biggest_n / total) <= policy["max_share_from_one_clip"]},
        {"check": "min_clips_per_split", "required": policy["min_clips_per_split"],
         "actual": min_split_clips, "pass": min_split_clips >= policy["min_clips_per_split"]},
    ]
    return {
        "policy": policy,
        "checks": checks,
        "status": "PASS" if all(c["pass"] for c in checks) else "FAIL",
        "clips_per_split": {k: len(v) for k, v in sorted(per_split.items())},
        "note": "The current 317-image set predates this policy and is expected "
                "to FAIL until the collection batch lands.",
    }


def _group(records: list[dict], hamming: int, dataset: str | None = None) -> list[list[int]]:
    """Connected components over clip-id, exact-SHA1 and near-dHash edges."""
    uf = _Union(len(records))

    by_sha = defaultdict(list)
    for i, r in enumerate(records):
        by_sha[r["sha1"]].append(i)
    for idxs in by_sha.values():
        for j in idxs[1:]:
            uf.union(idxs[0], j)

    # Clip grouping, where the export's filenames actually encode one.
    # Adjacent frames of one broadcast clip are the strongest near-duplicate
    # signal there is, and they are not always close in dHash (the camera
    # pans). This is applied ONLY where the naming has been validated --
    # see CLIP_PATTERNS.
    pattern = CLIP_PATTERNS.get(dataset)
    if pattern:
        clips = defaultdict(list)
        for i, r in enumerate(records):
            cid = _clip_id(r, pattern)
            if cid is not None:
                clips[cid].append(i)
        for idxs in clips.values():
            for j in idxs[1:]:
                uf.union(idxs[0], j)

    if hamming >= 0:
        # Exact pairwise comparison. An earlier version bucketed by 16-bit
        # bands to prune candidates; that is only sound for hamming <= 3
        # (pigeonhole over four bands), and at hamming 5 it silently missed
        # 11 cross-split pairs in ball. These datasets are ~2k images, so
        # the O(n^2) form costs a couple of seconds and is exactly correct.
        hashes = [r["dhash"] for r in records]
        for a in range(len(hashes)):
            ha = hashes[a]
            if ha is None:
                continue
            for b in range(a + 1, len(hashes)):
                hb = hashes[b]
                if hb is not None and (ha ^ hb).bit_count() <= hamming:
                    uf.union(a, b)

    comps = defaultdict(list)
    for i in range(len(records)):
        comps[uf.find(i)].append(i)
    return [sorted(v) for v in comps.values()]


# ----------------------------------------------------------------------
# Assignment
# ----------------------------------------------------------------------
def _assign(groups: list[list[int]], records: list[dict], targets: dict) -> dict[int, str]:
    """Greedy deterministic fill.

    Groups carrying labelled images are placed FIRST, so the scarce positives
    (551 of 1,404 for ball) are distributed against the target proportions
    before the unlabelled bulk is used to top each split up.  Within each
    pass the largest group goes to whichever split is furthest below its
    target -- placing large groups last is what strands them and forces a
    split.  Ties break on SHA-1, so the result is reproducible without a
    random seed.
    """
    total = sum(len(g) for g in groups)
    total_labeled = sum(1 for r in records if r["labeled"])
    assignment: dict[int, str] = {}
    have = dict.fromkeys(targets, 0)
    have_labeled = dict.fromkeys(targets, 0)

    def key(g: list[int]) -> tuple:
        return (-len(g), records[g[0]]["sha1"])

    labeled_groups = sorted([g for g in groups if any(records[i]["labeled"] for i in g)], key=key)
    plain_groups = sorted([g for g in groups if not any(records[i]["labeled"] for i in g)], key=key)

    for g in labeled_groups:
        n_lab = sum(1 for i in g if records[i]["labeled"])
        split = min(targets, key=lambda s: (
            (have_labeled[s] + n_lab) / max(targets[s] * total_labeled, 1e-9),
            (have[s] + len(g)) / max(targets[s] * total, 1e-9), s))
        assignment[id(g)] = split
        have[split] += len(g)
        have_labeled[split] += n_lab

    for g in plain_groups:
        split = min(targets, key=lambda s: ((have[s] + len(g)) / max(targets[s] * total, 1e-9), s))
        assignment[id(g)] = split
        have[split] += len(g)

    return {id(g): assignment[id(g)] for g in groups}


def build(name: str, out_root: Path, hamming: int, targets: dict | None = None) -> dict:
    records = _inventory(name)
    targets = targets or _targets_from_registry(records)
    groups = _group(records, hamming, dataset=name)

    # Where each group's images sit under the CURRENT registry splits --
    # this is the leakage being fixed, measured before any reassignment.
    old_cross = 0
    for g in groups:
        if len({records[i]["registry_split"] for i in g}) > 1:
            old_cross += 1

    assignment = _assign(groups, records, targets)

    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    by_split: dict[str, list[dict]] = defaultdict(list)
    group_rows = []
    for gi, g in enumerate(sorted(groups, key=lambda g: records[g[0]]["sha1"])):
        split = assignment[id(g)]
        old_splits = sorted({records[i]["registry_split"] for i in g})
        # One representative image per exact-duplicate digest: near-duplicates
        # are kept (they are distinct frames), byte-identical copies are not.
        seen: set[str] = set()
        for i in g:
            r = records[i]
            if r["sha1"] in seen:
                continue
            seen.add(r["sha1"])
            by_split[split].append(r)
        group_rows.append({
            "group_id": gi, "split": split, "n_images": len(g),
            "n_unique_sha1": len(seen),
            "n_labeled": sum(1 for i in g if records[i]["labeled"]),
            "registry_splits": "|".join(old_splits),
            "was_cross_split": len(old_splits) > 1,
            "reassigned": len(old_splits) > 1 or old_splits[0].replace("valid", "val") != split,
            "representative": str(records[g[0]]["image"]),
        })

    written = {}
    for split in ("train", "val", "test"):
        paths = sorted(by_split.get(split, []), key=lambda r: str(r["image"]))
        manifest = out_dir / f"{split}.txt"
        manifest.write_text("\n".join(_readable(r["image"]) for r in paths) + "\n",
                            encoding="utf-8")
        written[split] = {
            "manifest": str(manifest), "images": len(paths),
            "labeled": sum(1 for r in paths if r["labeled"]),
            "unlabeled": sum(1 for r in paths if not r["labeled"]),
            "instances": sum(r["label_rows"] for r in paths),
        }

    with (out_dir / "groups.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(group_rows[0]))
        w.writeheader()
        w.writerows(group_rows)

    # Verification: recompute cross-split groups against the NEW assignment.
    where: dict[str, set[str]] = defaultdict(set)
    for split in ("train", "val", "test"):
        for line in (out_dir / f"{split}.txt").read_text(encoding="utf-8").splitlines():
            if line.strip():
                where[line.strip()].add(split)
    path_to_group = {}
    for gi, g in enumerate(sorted(groups, key=lambda g: records[g[0]]["sha1"])):
        for i in g:
            path_to_group[_readable(records[i]["image"])] = gi
    group_splits = defaultdict(set)
    for path, splits in where.items():
        group_splits[path_to_group[path]] |= splits
    new_cross = sum(1 for s in group_splits.values() if len(s) > 1)

    summary = {
        "dataset": name,
        "hamming_threshold": hamming,
        "split_targets_from_registry": {k: round(v, 4) for k, v in targets.items()},
        "raw_images": len(records),
        "groups_total": len(groups),
        "groups_multi_image": sum(1 for g in groups if len(g) > 1),
        "exact_duplicate_groups": sum(
            1 for g in groups
            if len(g) > len({records[i]["sha1"] for i in g})),
        "near_duplicate_only_groups": sum(
            1 for g in groups
            if len(g) > 1 and len(g) == len({records[i]["sha1"] for i in g})),
        "clip_grouping": _clip_stats(records, name),
        "clip_policy": _check_clip_policy(records, name, by_split),
        "cross_split_groups_before": old_cross,
        "cross_split_groups_after": new_cross,
        "groups_reassigned": sum(1 for r in group_rows if r["reassigned"]),
        "splits": written,
        "unique_images_placed": sum(v["images"] for v in written.values()),
        "no_source_mutation": True,
        "written_outside_runs_data": True,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("datasets", nargs="*", default=["ball", "calibration"])
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "datasets" / "derived" / "splits_zero_leakage_v1")
    ap.add_argument("--hamming", type=int, default=5,
                    help="dHash Hamming distance treated as a near-duplicate; "
                         "-1 disables near-duplicate grouping (exact only)")
    ap.add_argument("--enforce-clip-policy", action="store_true",
                    help="exit non-zero if a dataset fails CLIP_POLICY; use on a "
                         "collection batch. Off by default because the existing "
                         "calibration set predates the policy.")
    args = ap.parse_args()
    failed = []
    for name in (args.datasets or ["ball", "calibration"]):
        summary = build(name, args.out, args.hamming)
        print(json.dumps(summary, indent=2))
        cp = summary.get("clip_policy") or {}
        if cp.get("status") == "FAIL":
            failed.append(name)
            for c in cp["checks"]:
                if not c["pass"]:
                    print(f"  CLIP POLICY FAIL [{name}] {c['check']}: "
                          f"required {c['required']}, got {c['actual']}", file=sys.stderr)
    if failed and args.enforce_clip_policy:
        print(f"\nclip policy failed for: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
