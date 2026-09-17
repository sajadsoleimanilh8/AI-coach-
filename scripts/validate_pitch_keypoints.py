"""
Validates PITCH_KEYPOINTS_32 (ai/computer_vision/tactical_analysis/pitch_keypoints.py)
against the calibration dataset's GROUND-TRUTH keypoint labels.

WHY THIS EXISTS
    pitch_keypoints.py's docstring cited "Reproduce with:
    python -m scripts.validate_pitch_keypoints" and quoted a median
    reprojection error of 0.247 m -- but the script did not exist in the
    repo, so that number was unreproducible. This script is that missing
    check, written so the claim can be confirmed or refuted rather than
    taken on faith.

WHAT IT ACTUALLY TESTS
    The index -> pitch-metre TABLE, not the trained model. It reads the
    dataset's own labelled keypoint pixel positions (flag=2 "visible"
    only), fits a homography from those pixels to the table's pitch
    metres, and reports how well every point reprojects.

    A correct table produces sub-metre agreement across many independent
    camera poses; a mis-assigned index scheme cannot, because each image
    is a different projective view and a wrong correspondence set has no
    single homography that satisfies it.

    Model keypoint QUALITY is a separate question -- see
    auto_calibration.py, which runs the trained model and is gated on
    HOMOGRAPHY_CONFIDENCE_MIN precisely because the model can be wrong on
    footage unlike its training domain even when this table is right.

Usage:
    python -m scripts.validate_pitch_keypoints
    python -m scripts.validate_pitch_keypoints --split train --min-kpts 6
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

from ai.computer_vision.tactical_analysis.homography import compute_homography  # noqa: E402
from ai.computer_vision.tactical_analysis.pitch_keypoints import (  # noqa: E402
    PITCH_KEYPOINTS_32,
    keypoints_to_correspondences,
    verify_table,
)
from configs import registry  # noqa: E402

# Ultralytics pose visibility flag: 2 = labelled AND visible, 1 = labelled
# but occluded, 0 = not labelled. Only flag-2 points are trusted here --
# an occluded landmark's pixel position is the annotator's guess, and
# feeding guesses into a fit that is meant to VALIDATE the table would
# make a wrong table look better than it is.
VISIBLE_FLAG = 2


def _iter_labels(split: str):
    """Yields (label_path, image_size, {idx: (px, py)}) per labelled image."""
    for src in registry.dataset_source_dirs("calibration"):
        labels_dir = src / split / "labels"
        images_dir = src / split / "images"
        if not labels_dir.is_dir():
            continue
        for lab in sorted(labels_dir.glob("*.txt")):
            parts = lab.read_text(encoding="utf-8").split()
            if len(parts) < 5 + 32 * 3:
                continue

            img = None
            for suffix in (".jpg", ".jpeg", ".png"):
                cand = images_dir / (lab.stem + suffix)
                if cand.exists():
                    img = cand
                    break
            if img is None:
                continue
            im = cv2.imread(str(img))
            if im is None:
                continue
            h, w = im.shape[:2]

            kpts: dict[int, tuple[float, float]] = {}
            body = parts[5:]
            for i in range(32):
                x, y, v = body[i * 3], body[i * 3 + 1], body[i * 3 + 2]
                if int(float(v)) != VISIBLE_FLAG:
                    continue
                kpts[i] = (float(x) * w, float(y) * h)
            yield lab, (w, h), kpts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", choices=["train", "valid", "test"])
    ap.add_argument("--min-kpts", type=int, default=4,
                    help="minimum visible keypoints required to attempt a fit")
    args = ap.parse_args()

    problems = verify_table()
    print("PITCH_KEYPOINTS_32 self-consistency (indices + flip symmetry):")
    if problems:
        for p in problems:
            print(f"  PROBLEM: {p}")
    else:
        print(f"  OK -- all 32 indices present, flip symmetry consistent "
              f"({len(PITCH_KEYPOINTS_32)} entries)")

    errors: list[float] = []
    per_image: list[tuple[str, int, float]] = []
    skipped = 0

    for lab, _size, kpts in _iter_labels(args.split):
        if len(kpts) < args.min_kpts:
            skipped += 1
            continue
        pixel_pts, pitch_pts, _used = keypoints_to_correspondences(kpts)
        if len(pixel_pts) < 4:
            skipped += 1
            continue
        # RANSAC once there are more than the minimal 4: a single
        # mislabelled landmark otherwise drags the whole fit and would be
        # reported as evidence against the table rather than against that
        # one annotation.
        method = cv2.RANSAC if len(pixel_pts) > 4 else 0
        try:
            res = compute_homography(np.array(pixel_pts), np.array(pitch_pts),
                                     method=method, ransac_reproj_threshold=3.0)
        except ValueError:
            skipped += 1
            continue
        errors.append(res.reprojection_error_m)
        per_image.append((lab.stem, res.n_points, res.reprojection_error_m))

    print(f"\nsplit={args.split}  images fitted={len(errors)}  skipped(<{args.min_kpts} kpts)={skipped}")
    if not errors:
        print("NO IMAGES FITTED -- cannot validate the table.")
        return 1

    arr = np.array(errors)
    print("\nReprojection error, GROUND-TRUTH keypoints -> PITCH_KEYPOINTS_32 (metres):")
    print(f"  median      = {statistics.median(errors):.3f}")
    print(f"  mean        = {arr.mean():.3f}")
    print(f"  90th pct    = {np.percentile(arr, 90):.3f}")
    print(f"  max         = {arr.max():.3f}")
    print(f"  under 1.0 m = {int((arr < 1.0).sum())}/{len(arr)} "
          f"({100 * (arr < 1.0).mean():.1f}%)")
    print(f"  under 2.0 m = {int((arr < 2.0).sum())}/{len(arr)} "
          f"({100 * (arr < 2.0).mean():.1f}%)")

    worst = sorted(per_image, key=lambda t: -t[2])[:5]
    print("\nWorst 5 images:")
    for name, n, err in worst:
        print(f"  {err:8.3f} m  n={n:2d}  {name[:60]}")

    # A correct table is the only way independent camera poses agree at
    # this scale; treat >1 m median as a failure worth a non-zero exit.
    return 0 if statistics.median(errors) < 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
