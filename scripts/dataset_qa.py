"""
Dataset QA -- content-level audit for every registered dataset.

configs/registry.py::verify_dataset() is the fast STRUCTURAL gate (do the
splits exist, are they non-empty). This script is the slow CONTENT gate: it
opens every label file and every image header and reports what is actually
wrong with the data.

Deliberate policy: this script REPORTS, it does not delete or rewrite
anything. Bad samples are listed with their paths so a human decides. A QA
pass that silently drops samples hides exactly the problems it exists to
find.

Checks performed
    orphans        image with no label file, label file with no image
    empty          zero-byte / whitespace-only label files (these are legal
                   in YOLO -- they mean "background, no objects" -- so they
                   are counted and reported, never treated as an error)
    malformed      wrong field count for the dataset's task, non-numeric
                   fields, or a class id outside [0, nc)
    degenerate     zero/negative width or height boxes
    out_of_range   normalised coordinates outside [0, 1]
    extreme_ar     bbox aspect ratio outside [1/20, 20]
    tiny           bbox area < 0.0001 of the image (reported, not an error:
                   for the ball dataset these are the whole point)
    dup_images     byte-identical images (sha1), including across splits --
                   a train/val duplicate is train/test leakage and is
                   escalated
    class_balance  instance counts per class per split
    corrupt        images PIL cannot open or that fail verify()

Usage
    python -m scripts.dataset_qa                 # all datasets
    python -m scripts.dataset_qa ball player     # named datasets
    python -m scripts.dataset_qa --out docs/dataset_audit
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

# Allow `python scripts/dataset_qa.py` as well as `-m scripts.dataset_qa`.
from configs import registry as R  # noqa: E402

EXTREME_AR = 20.0
TINY_AREA = 1e-4

# Windows MAX_PATH. Roboflow exports embed the original (often absurdly
# long) source filename into every export, and several goalpost samples
# blow past 260 characters, at which point the normal Win32 API refuses to
# open them -- os.scandir still LISTS them, so they look present right up
# until something tries to read one. Training hits this too, not just QA.
MAX_PATH = 260


def _ext(p: Path) -> str:
    r"""Extended-length ('\\?\') form of an absolute path on Windows.

    Lets us read files whose path exceeds MAX_PATH instead of crashing.
    No-op on POSIX."""
    if sys.platform != "win32":
        return str(p)
    s = str(p.resolve())
    return s if s.startswith("\\\\?\\") else "\\\\?\\" + s


def _expected_fields(task: str, kpt_shape) -> int | None:
    """Field count one label row must have. Segmentation is variable-length
    (a polygon of any vertex count), so it returns None and is checked by a
    different rule."""
    if task == "detect":
        return 5                                    # cls cx cy w h
    if task == "pose":
        return 5 + int(kpt_shape[0]) * int(kpt_shape[1])
    return None                                     # segment


def audit_dataset(name: str) -> dict:
    spec = R.dataset_spec(name)
    task = spec.get("task", "detect")
    nc = int(spec["nc"])
    kpt_shape = spec.get("kpt_shape")
    expected = _expected_fields(task, kpt_shape)

    report: dict = {
        "name": name, "task": task, "nc": nc,
        "root": str(R.dataset_root()),
        "sources": [str(p) for p in R.dataset_source_dirs(name)],
        "splits": {},
        "issues": defaultdict(list),
        "class_counts": defaultdict(Counter),
        "hash_index": defaultdict(list),
    }

    for src in R.dataset_source_dirs(name):
        for split in ("train", "valid", "test"):
            img_dir, lbl_dir = src / split / "images", src / split / "labels"
            if not img_dir.is_dir():
                continue
            key = f"{src.name}/{split}"
            stats = {"images": 0, "labels": 0, "instances": 0, "empty_labels": 0}

            images = [p for p in img_dir.iterdir()
                      if p.suffix.lower() in R.IMAGE_SUFFIXES]
            stats["images"] = len(images)
            label_stems = {p.stem for p in lbl_dir.glob("*.txt")} if lbl_dir.is_dir() else set()
            stats["labels"] = len(label_stems)

            for img in images:
                # -- duplicate / corrupt image ------------------------
                if len(str(img)) > MAX_PATH:
                    report["issues"]["long_path"].append(
                        f"{len(str(img))} chars: {img}")
                try:
                    raw = Path(_ext(img)).read_bytes()
                    digest = hashlib.sha1(raw).hexdigest()
                    report["hash_index"][digest].append(f"{split}:{img.name}")
                except OSError as exc:
                    report["issues"]["corrupt"].append(f"{img} ({exc})")
                    continue
                try:
                    from PIL import Image
                    with Image.open(_ext(img)) as im:
                        im.verify()
                except Exception as exc:                    # noqa: BLE001
                    report["issues"]["corrupt"].append(f"{img} ({exc})")
                    continue

                if img.stem not in label_stems:
                    report["issues"]["orphans"].append(f"image without label: {img}")

            for lbl in (lbl_dir.glob("*.txt") if lbl_dir.is_dir() else []):
                if len(str(lbl)) > MAX_PATH:
                    report["issues"]["long_path"].append(
                        f"{len(str(lbl))} chars: {lbl}")
                try:
                    text = Path(_ext(lbl)).read_text(
                        encoding="utf-8", errors="replace").strip()
                except OSError as exc:
                    report["issues"]["corrupt"].append(f"{lbl} ({exc})")
                    continue
                if not text:
                    stats["empty_labels"] += 1
                    continue
                for lineno, line in enumerate(text.splitlines(), 1):
                    parts = line.split()
                    if not parts:
                        continue
                    where = f"{lbl}:{lineno}"
                    try:
                        vals = [float(x) for x in parts]
                    except ValueError:
                        report["issues"]["malformed"].append(f"{where} non-numeric field")
                        continue

                    cls = int(vals[0])
                    if not (0 <= cls < nc):
                        report["issues"]["malformed"].append(
                            f"{where} class id {cls} outside [0,{nc})")
                        continue
                    stats["instances"] += 1
                    report["class_counts"][split][cls] += 1

                    if expected is not None and len(vals) != expected:
                        # A detect-task row carrying more than 5 fields is a
                        # POLYGON, not a broken box. Ultralytics' own
                        # verify_image_label() converts polygon labels to
                        # their bounding box (segments2boxes) when the task
                        # is detect, so these train correctly -- they are a
                        # mixed-export inconsistency, not corruption.
                        # Verified against the goalpost set: 2,750 rows are
                        # 5-field boxes and 252 are 11/13/15-field polygons.
                        # Pose is different: a wrong field count there means
                        # the keypoint count disagrees with kpt_shape, which
                        # genuinely cannot be loaded.
                        odd = len(vals) > 5 and (len(vals) - 1) % 2 == 0
                        if task == "detect" and odd:
                            report["issues"]["mixed_format"].append(
                                f"{where} polygon with {(len(vals)-1)//2} vertices "
                                f"(auto-converted to bbox by ultralytics)")
                        else:
                            report["issues"]["malformed"].append(
                                f"{where} has {len(vals)} fields, expected {expected}")
                        continue
                    if expected is None:                     # segment
                        coords = vals[1:]
                        if len(coords) < 6 or len(coords) % 2 != 0:
                            report["issues"]["malformed"].append(
                                f"{where} polygon has {len(coords)} coords "
                                f"(need an even count >= 6)")
                            continue
                        # Ultralytics does NOT clip out-of-range segments at
                        # load time -- it REJECTS the label and drops the image
                        # as corrupt:
                        #
                        #   ultralytics/data/utils.py:271-272
                        #     points = lb[:, 1:]   # xywh via segments2boxes
                        #     assert points.max() <= 1.01
                        #     assert lb.min()   >= -0.01
                        #
                        # Treating these as an informational "off_frame" note
                        # once produced a false PASS for the field dataset
                        # while 53% of its training images were being silently
                        # discarded.
                        #
                        # Note the check applies to the box DERIVED from
                        # the polygon, not the vertices, so we emulate
                        # segments2boxes here rather than testing vertices
                        # directly -- a polygon spanning -0.011..1.007 has
                        # width 1.018 and fails even though no single
                        # vertex is far out of range.
                        xs, ys = coords[0::2], coords[1::2]
                        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                        box = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
                        if max(box) > 1.01 or min(box) < -0.01:
                            report["issues"]["out_of_range"].append(
                                f"{where} derived box {['%.3f' % v for v in box]} "
                                f"outside ultralytics tolerance -- ultralytics "
                                f"WILL DROP this image as corrupt")
                        continue

                    # -- detect / pose share the cx cy w h prefix ------
                    cx, cy, w, h = vals[1:5]
                    if w <= 0 or h <= 0:
                        report["issues"]["degenerate"].append(
                            f"{where} w={w} h={h}")
                        continue
                    # Same ultralytics tolerance as above, applied to the
                    # detect/pose coordinate block. Anything outside it
                    # means the image is DROPPED at load, not clipped.
                    checked = vals[1:5] if task == "detect" else vals[5:]
                    if checked and (max(checked) > 1.01 or min(checked) < -0.01):
                        report["issues"]["out_of_range"].append(
                            f"{where} coordinate outside ultralytics tolerance "
                            f"(max={max(checked):.3f} min={min(checked):.3f}) -- "
                            f"ultralytics WILL DROP this image as corrupt")
                    ar = w / h
                    if ar > EXTREME_AR or ar < 1 / EXTREME_AR:
                        report["issues"]["extreme_ar"].append(f"{where} aspect={ar:.2f}")
                    if w * h < TINY_AREA:
                        report["issues"]["tiny"].append(f"{where} area={w*h:.6f}")

                stem_missing = lbl.stem not in {p.stem for p in images}
                if stem_missing:
                    report["issues"]["orphans"].append(f"label without image: {lbl}")

            report["splits"][key] = stats

    # -- duplicates, with train/val leakage escalated ------------------
    for _digest, occurrences in report["hash_index"].items():
        if len(occurrences) > 1:
            splits = {o.split(":")[0] for o in occurrences}
            label = "LEAKAGE across splits" if len(splits) > 1 else "within split"
            report["issues"]["dup_images"].append(
                f"[{label}] {len(occurrences)}x identical: {', '.join(occurrences[:6])}"
                + (" ..." if len(occurrences) > 6 else ""))
    # Severity-sort so the truncated sample in the rendered report shows
    # LEAKAGE first. Without this, cross-split duplicates can sit past the
    # 15-item display cut-off and the report reads as clean while the
    # summary count says otherwise -- which is exactly what happened on the
    # ball set (114 leakage groups, none visible in the printed sample).
    report["issues"]["dup_images"].sort(key=lambda s: (0 if "LEAKAGE" in s else 1, s))
    del report["hash_index"]
    return report


def render_markdown(rep: dict) -> str:
    name = rep["name"]
    total_imgs = sum(s["images"] for s in rep["splits"].values())
    total_inst = sum(s["instances"] for s in rep["splits"].values())
    issues = rep["issues"]

    # A dataset "blocks training" only on problems that corrupt learning.
    # tiny/extreme_ar/empty are informational -- for the ball dataset,
    # tiny boxes ARE the signal.
    blocking = ["corrupt", "malformed", "degenerate", "out_of_range", "long_path"]
    n_block = sum(len(issues.get(k, [])) for k in blocking)
    leakage = [d for d in issues.get("dup_images", []) if "LEAKAGE" in d]

    lines = [
        f"# Dataset audit -- `{name}`",
        "",
        f"_Generated {date.today().isoformat()} by `scripts/dataset_qa.py`._",
        "",
        f"- **Task**: `{rep['task']}`  |  **classes**: {rep['nc']}",
        f"- **Resolved root**: `{rep['root']}`",
        "- **Source dirs**:",
    ]
    lines += [f"  - `{s}`" for s in rep["sources"]]
    lines += [
        "",
        f"- **Images**: {total_imgs:,}  |  **labelled instances**: {total_inst:,}",
        "",
        f"**Verdict: {'PASS' if n_block == 0 and not leakage else 'REVIEW REQUIRED'}** "
        f"({n_block} blocking issue(s), {len(leakage)} cross-split duplicate group(s))",
        "",
        "## Splits",
        "",
        "| split | images | label files | instances | empty labels |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, s in rep["splits"].items():
        lines.append(
            f"| `{key}` | {s['images']:,} | {s['labels']:,} | "
            f"{s['instances']:,} | {s['empty_labels']:,} |")

    lines += ["", "## Class balance", "",
              "| split | class | instances |", "|---|---|---:|"]
    names = R.dataset_spec(name)["names"]
    for split, counter in sorted(rep["class_counts"].items()):
        for cls, n in sorted(counter.items()):
            lines.append(f"| {split} | `{names.get(cls, cls)}` | {n:,} |")

    lines += ["", "## Issues", ""]
    order = ["corrupt", "malformed", "degenerate", "out_of_range",
             "long_path", "dup_images", "orphans", "mixed_format",
             "extreme_ar", "tiny"]
    severity = {
        "corrupt": "BLOCKING", "malformed": "BLOCKING",
        "degenerate": "BLOCKING", "out_of_range": "BLOCKING",
        "long_path": "BLOCKING", "dup_images": "REVIEW", "orphans": "REVIEW",
        "mixed_format": "INFO",
        "extreme_ar": "INFO", "tiny": "INFO",
    }
    if not any(issues.get(k) for k in order):
        lines.append("None found.")
    for k in order:
        items = issues.get(k, [])
        if not items:
            continue
        lines += [f"### `{k}` -- {len(items)} ({severity[k]})", ""]
        for item in items[:15]:
            lines.append(f"- `{item}`")
        if len(items) > 15:
            lines.append(f"- _... and {len(items) - 15} more_")
        lines.append("")

    lines += [
        "## Policy",
        "",
        "Nothing here was auto-deleted or auto-corrected. `INFO` rows are "
        "expected characteristics, not defects -- in particular `tiny` boxes "
        "in the ball dataset are the intended signal, not noise. Only "
        "`BLOCKING` rows and cross-split duplicates should gate training.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("datasets", nargs="*", default=None)
    ap.add_argument("--out", default="docs/dataset_audit")
    args = ap.parse_args()

    names = args.datasets or list(R.datasets_config()["datasets"].keys())
    out_dir = R.REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    worst = 0
    for name in names:
        print(f"[qa] auditing {name} ...", flush=True)
        structural = R.verify_dataset(name, strict=False)
        if not structural.ok:
            print(R.format_dataset_report(structural))
            worst = 1
            continue
        rep = audit_dataset(name)
        path = out_dir / f"{name}.md"
        path.write_text(render_markdown(rep), encoding="utf-8")
        blocking = sum(len(rep["issues"].get(k, []))
                       for k in ("corrupt", "malformed", "degenerate", "out_of_range", "long_path"))
        dups = len(rep["issues"].get("dup_images", []))
        print(f"[qa] {name}: {sum(s['images'] for s in rep['splits'].values()):,} images, "
              f"{blocking} blocking, {dups} duplicate group(s) -> {path}")
        worst = max(worst, 1 if blocking else 0)
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
