"""
Build datasets/derived/field_clipped/ -- the field dataset with polygon
coordinates clipped into [0, 1].

WHY
    The field polygons are pitch quadrilaterals whose vertices legitimately
    extend past the frame edge (the touchline continues beyond what the
    camera sees) by up to 0.192 normalised units. Ultralytics rejects those
    labels outright rather than clipping them:

        ultralytics/data/utils.py:271-272
            points = lb[:, 1:]                      # xywh from segments2boxes
            assert points.max() <= 1.01, "non-normalized or out of bounds ..."

    Note it checks the box DERIVED from the polygon, not the vertices: a
    polygon spanning -0.011..1.007 has width 1.018, which trips the 1.01
    tolerance. The whole image is then dropped as "corrupt".

    Measured cost on the raw dataset: 448 of 841 train images (53%), 21 of
    35 valid and 8 of 15 test were silently discarded, so the field model
    trained on 46.7% of its data and was evaluated on 7 test images.

WHY CLIPPING IS CORRECT, NOT A FUDGE
    A segmentation mask describes what is visible IN THE IMAGE. Vertices
    outside the frame are an annotation-export artifact; the pitch region
    genuinely stops at the frame boundary. Clipping preserves the visible
    region exactly and changes nothing a model could have learned.

SOURCE IS NEVER MODIFIED
    Reads datasets/processed/calibration_dataset/ (yes -- that directory
    holds the FIELD data, see configs/datasets.yaml's naming warning) and
    writes a separate tree. Images are hard-linked where the filesystem
    allows it, so no image bytes are duplicated; labels are rewritten.

    python -m scripts.build_field_clipped
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from configs import registry as R  # noqa: E402

SPLITS = ("train", "valid", "test")


def clip_label_text(text: str) -> tuple[str, bool]:
    """Returns (clipped_text, was_modified). Class id is left untouched;
    only coordinate fields are clamped."""
    out_lines, modified = [], False
    for line in text.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls, coords = parts[0], [float(v) for v in parts[1:]]
        clipped = [min(1.0, max(0.0, v)) for v in coords]
        if any(abs(a - b) > 1e-9 for a, b in zip(coords, clipped)):
            modified = True
        # %g keeps the files compact and avoids trailing float noise
        out_lines.append(cls + " " + " ".join(f"{v:.6g}" for v in clipped))
    return "\n".join(out_lines) + "\n", modified


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)          # hard link: no duplicated image bytes
    except OSError:
        shutil.copy2(src, dst)     # different volume / no link support


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None,
                    help="output root (default: <datasets>/derived/field_clipped)")
    args = ap.parse_args()

    # The field dataset's single source dir (see configs/datasets.yaml).
    src_root = R.dataset_source_dirs("field")[0]
    out_root = Path(args.out) if args.out else (
        R.dataset_root().parent / "derived" / "field_clipped")

    print(f"source : {src_root}")
    print(f"output : {out_root}\n")
    if not src_root.exists():
        print(f"ERROR: source dataset not found: {src_root}")
        return 1

    totals = {"images": 0, "labels": 0, "clipped": 0, "unchanged": 0, "missing_label": 0}
    for split in SPLITS:
        s_img, s_lbl = src_root / split / "images", src_root / split / "labels"
        d_img, d_lbl = out_root / split / "images", out_root / split / "labels"
        if not s_img.is_dir():
            print(f"  {split:6s} SKIPPED (no images dir)")
            continue
        d_img.mkdir(parents=True, exist_ok=True)
        d_lbl.mkdir(parents=True, exist_ok=True)

        n_img = n_lbl = n_clip = n_same = n_miss = 0
        for img in sorted(s_img.iterdir()):
            if img.suffix.lower() not in R.IMAGE_SUFFIXES:
                continue
            link_or_copy(img, d_img / img.name)
            n_img += 1

            lbl = s_lbl / f"{img.stem}.txt"
            if not lbl.exists():
                n_miss += 1
                continue
            text, modified = clip_label_text(
                lbl.read_text(encoding="utf-8", errors="replace"))
            (d_lbl / lbl.name).write_text(text, encoding="utf-8")
            n_lbl += 1
            n_clip += int(modified)
            n_same += int(not modified)

        print(f"  {split:6s} images={n_img:5d}  labels={n_lbl:5d}  "
              f"clipped={n_clip:5d}  unchanged={n_same:5d}  no-label={n_miss}")
        totals["images"] += n_img
        totals["labels"] += n_lbl
        totals["clipped"] += n_clip
        totals["unchanged"] += n_same
        totals["missing_label"] += n_miss

    print(f"\n  TOTAL images={totals['images']:,}  labels={totals['labels']:,}  "
          f"clipped={totals['clipped']:,}  unchanged={totals['unchanged']:,}")
    print(f"  source left untouched: {src_root}")

    # ---- verify against ultralytics' own rule -------------------------
    import numpy as np
    bad = []
    for split in SPLITS:
        for lbl in sorted((out_root / split / "labels").glob("*.txt")):
            rows = [x.split() for x in lbl.read_text(encoding="utf-8").strip().splitlines() if x]
            if not rows:
                continue
            if any(len(x) > 6 for x in rows):        # segment rows
                segs = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in rows]
                boxes = []
                for s in segs:                        # emulate segments2boxes
                    x, y = s.T
                    x1, y1, x2, y2 = x.min(), y.min(), x.max(), y.max()
                    boxes.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
                pts = np.array(boxes, dtype=np.float32)
            else:
                pts = np.array([[float(v) for v in x[1:]] for x in rows], dtype=np.float32)
            if pts.max() > 1.01 or pts.min() < -0.01:
                bad.append((lbl.name, float(pts.max()), float(pts.min())))

    print(f"\n  ultralytics-rule check: "
          f"{'ALL LABELS PASS' if not bad else f'{len(bad)} STILL FAILING'}")
    for name, mx, mn in bad[:10]:
        print(f"    {name}: max={mx:.4f} min={mn:.4f}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
