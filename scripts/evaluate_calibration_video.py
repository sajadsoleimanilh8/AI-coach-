"""
Per-frame calibration diagnostics over a REAL VIDEO, not a labelled split.

WHY THIS EXISTS ALONGSIDE validate_auto_calibration.py
    validate_auto_calibration.py answers "how many metres wrong is the
    homography" and needs GROUND-TRUTH keypoints, so it can only run on the
    labelled dataset split -- which is exactly the in-domain footage the
    model already handles. It cannot say anything about a broadcast clip.

    This script answers the different question the pipeline actually cares
    about: *over a real video, how many frames end up with
    calibration.valid, and for the ones that do not, WHY NOT.* There is no
    ground truth here, so it deliberately reports no positional accuracy --
    only detection yield, fit geometry, and the exact gate that rejected
    each frame.

WHAT IT MEASURES (per frame)
    - box confidence of the pitch instance, and all 32 keypoint confidences
    - usable keypoints at EVERY threshold in the sweep, from one inference
    - homography fit: reprojection error, RANSAC inlier count/ratio
    - which gate rejected the frame (the exact invalid_reason)
    - preprocessing path comparison (--compare-preprocess)

ONE INFERENCE PER FRAME PER PATH
    The model is run at a near-zero box-confidence floor and the configured
    floor is applied afterwards in software. With max_det=1 that is exactly
    equivalent to running at the configured floor -- the same single best
    box either survives the gate or does not -- and it makes the whole
    threshold sweep free instead of one full pass per threshold.

Usage:
    python -m scripts.evaluate_calibration_video test.mp4 --frames 3300
    python -m scripts.evaluate_calibration_video test.mp4 --frames 3300 \
        --compare-preprocess --json runs/calib_eval.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

from ai.computer_vision.frame_data import FieldRegion  # noqa: E402
from ai.computer_vision.tactical_analysis.auto_calibration import (  # noqa: E402
    calibrate_from_keypoints,
    to_calibration_state,
)
from ai.computer_vision.tactical_analysis.constants import (  # noqa: E402
    MIN_CALIBRATION_POINTS,
)
from configs import registry  # noqa: E402

#: Thresholds swept for the keypoint visibility floor. Every one of these is
#: evaluated from the SAME inference, so the sweep costs nothing extra.
DEFAULT_SWEEP = (0.20, 0.25, 0.30, 0.35, 0.40, 0.50)

#: Box-confidence floor used for the raw inference. Deliberately near zero:
#: the configured floor is re-applied in software so that "the model found
#: nothing at all" and "the model found something the gate rejected" are
#: distinguishable, which a run at the configured floor cannot show.
RAW_BOX_CONF = 0.001

FIELD_DETECT_STRIDE = 25


def _predict(model, frame, imgsz, stretch_square):
    """One inference. Returns (box_conf, kpts_xy [32,2], kpt_conf [32]) in
    ORIGINAL frame pixel coordinates, or None when nothing was detected."""
    h, w = frame.shape[:2]
    if stretch_square:
        img = cv2.resize(frame, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        sx, sy = w / float(imgsz), h / float(imgsz)
    else:
        img = frame
        sx = sy = 1.0

    res = model.predict(img, imgsz=imgsz, conf=RAW_BOX_CONF, verbose=False,
                        max_det=1)[0]
    if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
        return None
    conf = res.boxes.conf
    conf = conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf)
    best = int(np.argmax(conf))
    data = res.keypoints.data[best]
    data = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
    xy = data[:, :2].astype(np.float64).copy()
    xy[:, 0] *= sx
    xy[:, 1] *= sy
    kc = data[:, 2].astype(np.float64) if data.shape[1] >= 3 else np.ones(len(data))
    return float(conf[best]), xy, kc


def _fit_at(xy, kc, threshold, frame_size, field_region):
    """Fit + gate at one keypoint threshold. Returns a diagnostics dict."""
    keypoints = {}
    confidences = {}
    for idx in range(len(kc)):
        confidences[idx] = float(kc[idx])
        x, y = float(xy[idx][0]), float(xy[idx][1])
        if kc[idx] < threshold or (x == 0.0 and y == 0.0):
            continue
        keypoints[idx] = (x, y)

    out = {"n_above": len(keypoints), "valid": False, "reason": None,
           "reproj_m": None, "confidence": None, "inliers": None,
           "inlier_ratio": None, "n_points": 0}

    if len(keypoints) < MIN_CALIBRATION_POINTS:
        out["reason"] = f"only {len(keypoints)} keypoints >= {threshold}"
        return out

    res = calibrate_from_keypoints(keypoints, confidences=confidences,
                                   min_confidence=threshold)
    if not res.ok:
        out["reason"] = res.reason
        return out

    h = res.homography
    out["reproj_m"] = h.reprojection_error_m
    out["confidence"] = h.confidence
    out["n_points"] = h.n_points
    out["inliers"] = getattr(h, "n_inliers", None)
    out["inlier_ratio"] = getattr(h, "inlier_ratio", None)

    state = to_calibration_state(res, field_region=field_region,
                                 frame_size=frame_size)
    out["valid"] = state.valid
    out["reason"] = state.invalid_reason
    return out


def _summarise(rows, key, label, sweep, configured_thr, box_conf_floor):
    """Prints the BEFORE/AFTER block for one preprocessing path."""
    n = len(rows)
    print(f"\n{'=' * 76}\n{label}   ({n} frames)\n{'=' * 76}")

    detected = [r for r in rows if r[key] is not None]
    passing_box = [r for r in detected if r[key]["box_conf"] >= box_conf_floor]
    print(f"  model produced a box at all (conf >= {RAW_BOX_CONF}): "
          f"{len(detected)}/{n} ({100 * len(detected) / max(n, 1):.1f}%)")
    print(f"  box clears the configured floor (conf >= {box_conf_floor}): "
          f"{len(passing_box)}/{n} ({100 * len(passing_box) / max(n, 1):.1f}%)")
    if detected:
        bcs = [r[key]["box_conf"] for r in detected]
        print(f"  box confidence  min={min(bcs):.4f}  median={statistics.median(bcs):.4f}"
              f"  max={max(bcs):.4f}")

    if passing_box:
        allk = np.concatenate([r[key]["kpt_conf"] for r in passing_box])
        print(f"  keypoint confidence over detected frames: "
              f"min={allk.min():.3f} mean={allk.mean():.3f} max={allk.max():.3f}")

    print(f"\n  {'kpt_thr':>8s} {'kpts/frm':>9s} {'>=4kpts':>9s} {'fitted':>8s} "
          f"{'VALID':>8s} {'valid%':>8s} {'reproj_m':>9s} {'inlier%':>8s}")
    for thr in sweep:
        fits = [r[key]["sweep"][str(thr)] for r in passing_box]
        if not fits:
            print(f"  {thr:8.2f} {'-':>9s} {'-':>9s} {'-':>8s} {0:8d} {0.0:7.1f}%")
            continue
        n_above = [f["n_above"] for f in fits]
        enough = [f for f in fits if f["n_above"] >= MIN_CALIBRATION_POINTS]
        fitted = [f for f in fits if f["reproj_m"] is not None]
        valid = [f for f in fits if f["valid"]]
        rep = [f["reproj_m"] for f in fitted]
        ratios = [f["inlier_ratio"] for f in fitted if f["inlier_ratio"] is not None]
        mark = "  <- configured" if abs(thr - configured_thr) < 1e-9 else ""
        print(f"  {thr:8.2f} {statistics.mean(n_above):9.2f} {len(enough):9d} "
              f"{len(fitted):8d} {len(valid):8d} {100 * len(valid) / n:7.1f}% "
              f"{(statistics.median(rep) if rep else float('nan')):9.3f} "
              f"{(100 * statistics.mean(ratios) if ratios else float('nan')):7.1f}%{mark}")

    # Rejection reasons at the configured threshold, over EVERY frame.
    reasons = Counter()
    for r in rows:
        d = r[key]
        if d is None:
            reasons["model returned no instance at all"] += 1
            continue
        if d["box_conf"] < box_conf_floor:
            reasons[f"box confidence {box_conf_floor} not cleared"] += 1
            continue
        f = d["sweep"][str(configured_thr)]
        if f["valid"]:
            reasons["VALID"] += 1
        else:
            reasons[_bucket(f["reason"])] += 1

    print(f"\n  per-frame outcome at the configured floor "
          f"(box>={box_conf_floor}, kpt>={configured_thr}):")
    for reason, count in reasons.most_common():
        print(f"    {count:6d}  ({100 * count / max(n, 1):5.1f}%)  {reason}")


def _bucket(reason: str | None) -> str:
    """Collapses per-frame reasons (which embed numbers) into stable buckets."""
    if not reason:
        return "invalid, no reason recorded"
    for probe, bucket in (
        ("usable keypoints", "too few keypoints above the visibility floor"),
        ("keypoints >=", "too few keypoints above the visibility floor"),
        ("confidence", "reprojection confidence below HOMOGRAPHY_CONFIDENCE_MIN"),
        ("outside", "keypoints outside the detected pitch region"),
        ("span", "keypoints too clustered (point-spread gate)"),
        ("inlier", "too few RANSAC inliers"),
        ("degenerate", "degenerate / non-invertible homography"),
        ("mirror", "mirrored pitch geometry"),
        ("scale", "implausible projected scale"),
        ("bounds", "projected pitch outside plausible bounds"),
        ("fit failed", "cv2.findHomography failed"),
    ):
        if probe in reason:
            return bucket
    return reason[:70]


def run_pipeline_mode(video: str, n_want: int, stride: int,
                      use_field: bool) -> int:
    """
    Runs the REAL AutoCalibrator.calibrate() over the video, exactly as
    backend/pipeline/runner.py does, and reports the end-to-end outcome.

    WHY THIS IS SEPARATE FROM THE PER-FRAME MODE
        The per-frame mode above deliberately has no memory: it measures
        what the DETECTOR and the GEOMETRY can do on each frame standing
        alone. This mode adds the temporal machinery -- carry-forward,
        confidence decay, expiry, jump rejection, smoothing.

        Both numbers are needed, and reporting only the second would be
        misleading. "Valid frames went up" means something very different
        when the increase came from one good solve being reused for twenty
        frames than when it came from twenty frames solving independently.
        The source breakdown below is printed precisely so that question
        cannot be dodged.
    """
    from ai.computer_vision.frame_data import CalibrationSource
    from ai.computer_vision.tactical_analysis.auto_calibration import AutoCalibrator

    cal = AutoCalibrator()
    field_detector = None
    if use_field:
        try:
            from ai.computer_vision.detectors import FieldDetector
            field_detector = FieldDetector()
        except Exception as exc:  # noqa: BLE001
            print(f"field model unavailable ({exc}); pitch-region gate disabled")

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"could not open {video}")
        return 1

    print(f"preprocess={cal.preprocess}  kpt_conf_min={cal.kpt_conf_min}  "
          f"kpt_conf_relaxed={cal.kpt_conf_relaxed}")
    print(f"max_fallback_frames={cal.max_fallback_frames}  "
          f"fallback_decay={cal.fallback_decay}  smoothing={cal.smoothing}\n")

    sources = Counter()
    reasons = Counter()
    confidences = []
    reproj = []
    inlier_ratios = []
    carried_runs = []
    last_field = None
    n = 0
    t0 = time.time()
    for i in range(0, n_want, stride):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if stride > 1:
            for _ in range(stride - 1):
                if not cap.grab():
                    break
        if field_detector is not None and i % FIELD_DETECT_STRIDE == 0:
            try:
                detected = field_detector.detect(frame)
                last_field = detected if detected is not None else last_field
            except Exception:  # noqa: BLE001
                pass

        state, _camera = cal.calibrate(frame, i, field_region=last_field)
        n += 1
        if state.valid:
            sources[state.source.value] += 1
            confidences.append(state.confidence)
            if state.reprojection_error_m is not None:
                reproj.append(state.reprojection_error_m)
            if state.inlier_ratio is not None:
                inlier_ratios.append(state.inlier_ratio)
            if state.source is CalibrationSource.carried:
                carried_runs.append(state.carried_frames)
        else:
            sources["INVALID"] += 1
            reasons[_bucket(state.invalid_reason)] += 1
        if n % 250 == 0:
            print(f"  ... {n} frames ({n / (time.time() - t0):.1f} fps)")
    cap.release()
    elapsed = time.time() - t0

    n_valid = n - sources["INVALID"]
    print(f"\n{'=' * 76}\nEND-TO-END (AutoCalibrator.calibrate, as the pipeline "
          f"runs it)\n{'=' * 76}")
    print(f"  frames                 : {n}")
    print(f"  VALID calibrations     : {n_valid} ({100 * n_valid / max(n, 1):.1f}%)")
    print(f"  invalid                : {sources['INVALID']} "
          f"({100 * sources['INVALID'] / max(n, 1):.1f}%)")
    print("\n  where the valid ones came from:")
    solved = sources.get("model", 0)
    carried = sources.get("carried", 0)
    print(f"    solved on this frame : {solved} "
          f"({100 * solved / max(n_valid, 1):.1f}% of valid)")
    print(f"    carried from earlier : {carried} "
          f"({100 * carried / max(n_valid, 1):.1f}% of valid)")
    if carried_runs:
        print(f"      carried-age: mean {statistics.mean(carried_runs):.1f} frames, "
              f"max {max(carried_runs)}")

    stats = cal.stats()
    print(f"\n  calibrator counters   : {stats}")
    print(f"    strict rung          : {stats['accepted_strict']}")
    print(f"    relaxed rung         : {stats['accepted_relaxed']}")
    print(f"    fallback expiries    : {stats['fallback_expired']}")
    print(f"    smoothed solves      : {stats['smoothed']}")
    print(f"    rejected (invalid)   : {stats['rejected_invalid']}")
    print(f"    rejected (jump)      : {stats['rejected_jump']}")

    if confidences:
        print(f"\n  confidence  median={statistics.median(confidences):.3f}  "
              f"min={min(confidences):.3f}  max={max(confidences):.3f}")
    if reproj:
        print(f"  reproj (all pts, m)  median={statistics.median(reproj):.3f}  "
              f"mean={statistics.mean(reproj):.3f}  max={max(reproj):.3f}")
    if inlier_ratios:
        print(f"  inlier ratio  mean={100 * statistics.mean(inlier_ratios):.1f}%  "
              f"min={100 * min(inlier_ratios):.1f}%")
    if reasons:
        print("\n  invalid reasons:")
        for reason, count in reasons.most_common():
            print(f"    {count:6d}  ({100 * count / max(n, 1):5.1f}%)  {reason}")

    print(f"\nruntime: {elapsed:.1f}s for {n} frames "
          f"({n / max(elapsed, 1e-9):.2f} frames/s)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="path to the video to evaluate")
    ap.add_argument("--frames", type=int, default=3300,
                    help="number of consecutive frames from the start (0 = all)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--compare-preprocess", action="store_true",
                    help="also run the stretch-to-square path and print both")
    ap.add_argument("--no-field", action="store_true",
                    help="skip the field model (drops the pitch-region gate)")
    ap.add_argument("--json", default=None, help="write per-frame rows here")
    ap.add_argument("--pipeline", action="store_true",
                    help="run the real AutoCalibrator with its temporal logic "
                         "and report how much of the result came from reuse")
    args = ap.parse_args()

    if args.pipeline:
        probe = cv2.VideoCapture(args.video)
        total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
        probe.release()
        n_want = total if args.frames <= 0 else min(args.frames, total)
        print(f"video       : {args.video}  ({total} frames)")
        print(f"evaluating  : {n_want} frames, stride {args.stride}\n")
        return run_pipeline_mode(args.video, n_want, args.stride,
                                 not args.no_field)

    from ultralytics import YOLO

    spec = registry.get_model("calibration")
    inference = spec.inference
    imgsz = int(inference.get("imgsz", 960))
    box_conf_floor = float(inference.get("conf", 0.30))
    kpt_thr = float(inference.get("kpt_conf_min", 0.50))
    preprocess = str(inference.get("preprocess", "native"))
    model = YOLO(str(spec.require_checkpoint()))

    sweep = sorted(set(DEFAULT_SWEEP) | {kpt_thr})

    field_detector = None
    if not args.no_field:
        try:
            from ai.computer_vision.detectors import FieldDetector
            field_detector = FieldDetector()
        except Exception as exc:  # noqa: BLE001
            print(f"field model unavailable ({exc}); pitch-region gate disabled")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"could not open {args.video}")
        return 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_want = total if args.frames <= 0 else min(args.frames, total)

    print(f"video       : {args.video}  ({total} frames)")
    print(f"checkpoint  : {spec.checkpoint}")
    print(f"configured  : imgsz={imgsz} conf={box_conf_floor} "
          f"kpt_conf_min={kpt_thr} preprocess={preprocess}")
    print(f"evaluating  : {n_want} frames, stride {args.stride}\n")

    paths = [("configured", preprocess == "stretch_square")]
    if args.compare_preprocess:
        paths = [("native", False), ("stretch_square", True)]

    rows = []
    last_field: FieldRegion | None = None
    t0 = time.time()
    for i in range(0, n_want, args.stride):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if args.stride > 1:
            for _ in range(args.stride - 1):
                if not cap.grab():
                    break
        h, w = frame.shape[:2]

        if field_detector is not None and i % FIELD_DETECT_STRIDE == 0:
            try:
                detected = field_detector.detect(frame)
                last_field = detected if detected is not None else last_field
            except Exception:  # noqa: BLE001
                pass

        row = {"frame": i}
        for name, stretch in paths:
            pred = _predict(model, frame, imgsz, stretch)
            if pred is None:
                row[name] = None
                continue
            box_conf, xy, kc = pred
            row[name] = {
                "box_conf": box_conf,
                "kpt_conf": kc,
                "sweep": {str(t): _fit_at(xy, kc, t, (w, h), last_field)
                          for t in sweep},
            }
        rows.append(row)

        if len(rows) % 250 == 0:
            rate = len(rows) / (time.time() - t0)
            print(f"  ... {len(rows)}/{n_want // args.stride} frames "
                  f"({rate:.1f} fps)")
    cap.release()

    elapsed = time.time() - t0
    for name, _ in paths:
        _summarise(rows, name, f"PREPROCESSING PATH: {name}", sweep,
                   kpt_thr, box_conf_floor)

    print(f"\nruntime: {elapsed:.1f}s for {len(rows)} frames "
          f"({len(rows) / max(elapsed, 1e-9):.2f} frames/s, "
          f"{len(paths)} inference path(s) per frame)")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        serialisable = []
        for r in rows:
            item = {"frame": r["frame"]}
            for name, _ in paths:
                d = r[name]
                item[name] = None if d is None else {
                    "box_conf": d["box_conf"],
                    "kpt_conf": [round(float(c), 4) for c in d["kpt_conf"]],
                    "sweep": d["sweep"],
                }
            serialisable.append(item)
        out.write_text(json.dumps(serialisable), encoding="utf-8")
        print(f"per-frame rows written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
