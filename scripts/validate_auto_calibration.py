"""
Metre-level validation of AUTOMATIC calibration end to end.

WHAT QUESTION THIS ANSWERS
    The calibration model's published metrics are mAP50 0.9950 / pose
    mAP50-95 0.4319. Neither says what actually matters downstream: *if we
    calibrate from this model's keypoints, how many metres wrong is a
    player's pitch position?* mAP measures whether keypoints land near
    their targets in PIXELS on the detected instance; a homography turns
    small pixel errors on badly-chosen landmarks into large metre errors,
    and vice versa. This script measures the metres.

METHOD
    For each labelled image:
      1. Run the calibration model, keep keypoints above the visibility
         floor, fit a homography through the existing compute_homography().
      2. Take the GROUND-TRUTH keypoints (flag=2 only) as independent
         check points. Their true pitch coordinates are known from
         PITCH_KEYPOINTS_32.
      3. Project each GT pixel through the MODEL's homography and measure
         the distance, in metres, to where that landmark really is.

    Step 3 is the honest test. Reporting the fit's own reprojection error
    would only say the model's keypoints agree with each other -- and they
    can agree perfectly while describing the wrong end of the pitch, which
    is precisely the failure this script was written to catch.

RESULT ON THIS CHECKPOINT (2026-08-13, in-domain test split)
    See docs/pipeline_architecture.md. Summary: the model frequently fits
    a self-consistent homography to a MIRRORED reading of the pitch --
    sub-metre internal agreement, tens of metres wrong against ground
    truth. HOMOGRAPHY_CONFIDENCE_MIN catches most but not all of these,
    which is why the field-region cross-check exists and why
    manual_calibration.py is retained.

Usage:
    python -m scripts.validate_auto_calibration --split test
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

from ai.computer_vision.tactical_analysis.auto_calibration import AutoCalibrator  # noqa: E402
from ai.computer_vision.tactical_analysis.homography import pixels_to_pitch  # noqa: E402
from ai.computer_vision.tactical_analysis.pitch_keypoints import (  # noqa: E402
    PITCH_KEYPOINTS_32,
)
from configs import registry  # noqa: E402

VISIBLE_FLAG = 2


def _gt_keypoints(label_path: Path, w: int, h: int) -> dict[int, tuple[float, float]]:
    parts = label_path.read_text(encoding="utf-8").split()
    if len(parts) < 5 + 32 * 3:
        return {}
    body = parts[5:]
    out: dict[int, tuple[float, float]] = {}
    for i in range(32):
        x, y, v = body[i * 3], body[i * 3 + 1], body[i * 3 + 2]
        if int(float(v)) != VISIBLE_FLAG:
            continue
        out[i] = (float(x) * w, float(y) * h)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    cal = AutoCalibrator()
    print(f"checkpoint: {registry.get_model('calibration').checkpoint}")
    print(f"kpt_conf_min={cal.kpt_conf_min}  imgsz={cal.imgsz}  conf={cal.conf}\n")

    rows = []
    for src in registry.dataset_source_dirs("calibration"):
        images = sorted((src / args.split / "images").glob("*.jpg"))
        if args.limit:
            images = images[:args.limit]
        for img_path in images:
            lab = src / args.split / "labels" / (img_path.stem + ".txt")
            if not lab.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            gt = _gt_keypoints(lab, w, h)
            if len(gt) < 4:
                continue

            res = cal.calibrate_frame(img)
            if not res.ok:
                rows.append((img_path.stem, None, None, None, res.reason))
                continue

            # Project GROUND-TRUTH pixels through the MODEL's homography.
            gt_idx = sorted(gt)
            gt_px = np.array([gt[i] for i in gt_idx])
            true_m = np.array([PITCH_KEYPOINTS_32[i] for i in gt_idx])
            proj = pixels_to_pitch(gt_px, res.homography.H)
            errs = np.linalg.norm(proj - true_m, axis=1)

            rows.append((img_path.stem, res.homography.confidence,
                         res.homography.reprojection_error_m,
                         float(np.median(errs)), None))

    ok = [r for r in rows if r[3] is not None]
    print(f"images: {len(rows)}   calibrated: {len(ok)}   no-fit: {len(rows) - len(ok)}\n")
    if not ok:
        print("NO IMAGE PRODUCED A HOMOGRAPHY.")
        return 1

    print(f"{'image':42s} {'selfconf':>8s} {'selferr_m':>10s} {'TRUE_err_m':>11s}")
    for name, conf, self_err, true_err, _ in ok[:30]:
        print(f"{name[:42]:42s} {conf:8.3f} {self_err:10.3f} {true_err:11.3f}")

    true_errs = [r[3] for r in ok]
    confs = [r[1] for r in ok]
    passed_gate = [r for r in ok if r[1] >= 0.6]

    print("\n" + "=" * 78)
    print("TRUE positional error (GT landmarks projected through the model's H)")
    print("=" * 78)
    print(f"  median = {statistics.median(true_errs):8.3f} m")
    print(f"  mean   = {statistics.mean(true_errs):8.3f} m")
    print(f"  min    = {min(true_errs):8.3f} m")
    print(f"  max    = {max(true_errs):8.3f} m")
    print(f"  under 1 m : {sum(1 for e in true_errs if e < 1.0)}/{len(true_errs)}")
    print(f"  under 3 m : {sum(1 for e in true_errs if e < 3.0)}/{len(true_errs)}")
    print(f"  over 10 m : {sum(1 for e in true_errs if e > 10.0)}/{len(true_errs)}")

    print(f"\n  fits clearing HOMOGRAPHY_CONFIDENCE_MIN (0.6): {len(passed_gate)}/{len(ok)}")
    if passed_gate:
        pg = [r[3] for r in passed_gate]
        print(f"    of those, TRUE error median = {statistics.median(pg):.3f} m, "
              f"max = {max(pg):.3f} m")
        bad = [r for r in passed_gate if r[3] > 5.0]
        print(f"    CLEARED THE GATE BUT >5 m WRONG: {len(bad)}/{len(passed_gate)}")
        for name, conf, self_err, true_err, _ in bad[:10]:
            print(f"      {name[:46]:46s} conf={conf:.3f} selferr={self_err:.2f}m "
                  f"TRUE={true_err:.1f}m")
    print(f"\n  self-reported confidence: median = {statistics.median(confs):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
