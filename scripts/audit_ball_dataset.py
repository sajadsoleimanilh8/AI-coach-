"""Auditable ball-dataset triage and derived-manifest builder."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from configs import registry as R  # noqa: E402

IMAGE_SUFFIXES = R.IMAGE_SUFFIXES
SPLIT_PRIORITY = {"test": 0, "valid": 1, "train": 2}
_DETECTOR_CACHE: dict[str, object] = {}

COCO_SPORTS_BALL = 32

REPORT_THRESHOLDS = (0.10, 0.25, 0.50)
BATCH = 8


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _source_records() -> list[dict]:
    records = []
    for source in R.dataset_source_dirs("ball"):
        for split in ("train", "valid", "test"):
            image_dir = source / split / "images"
            label_dir = source / split / "labels"
            if not image_dir.is_dir():
                continue
            for image in sorted(image_dir.iterdir()):
                if image.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                label = label_dir / f"{image.stem}.txt"
                rows = []
                if label.exists():
                    rows = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
                records.append({
                    "source": source.name,
                    "source_split": split,
                    "image": image,
                    "label": label if label.exists() else None,
                    "sha1": _sha1(image),
                    "labeled": bool(rows),
                    "label_rows": len(rows),
                    "label_text": "\\n".join(rows),
                })
    return records


def _dedupe(records: list[dict]) -> tuple[list[dict], dict]:
    chosen: dict[str, dict] = {}
    duplicate_groups = 0
    cross_split_groups = 0
    for row in records:
        old = chosen.get(row["sha1"])
        if old is None:
            chosen[row["sha1"]] = row
        elif SPLIT_PRIORITY[row["source_split"]] < SPLIT_PRIORITY[old["source_split"]]:
            chosen[row["sha1"]] = row
    by_hash = defaultdict(list)
    for row in records:
        by_hash[row["sha1"]].append(row)
    for group in by_hash.values():
        if len(group) > 1:
            duplicate_groups += 1
            if len({r["source_split"] for r in group}) > 1:
                cross_split_groups += 1
    return list(chosen.values()), {
        "raw_images": len(records), "unique_images": len(chosen),
        "duplicate_groups": duplicate_groups, "cross_split_duplicate_groups": cross_split_groups,
    }


def _heuristic(path: Path) -> tuple[str, str]:
    """Cheap scene signal; deliberately not a verdict."""
    try:
        from PIL import Image, ImageStat
        with Image.open(path).convert("RGB") as im:
            stat = ImageStat.Stat(im.resize((64, 64)))
            r, g, b = stat.mean
            green = g - (r + b) / 2
            if green > 8:
                return "football_context_candidate", f"mean_rgb={r:.1f},{g:.1f},{b:.1f}; green_excess={green:.1f}"
            return "non_football_or_unknown_candidate", f"mean_rgb={r:.1f},{g:.1f},{b:.1f}; green_excess={green:.1f}"
    except Exception as exc:  # pragma: no cover - optional dependency/runtime issue
        return "unavailable", f"heuristic_error={exc}"


def _detector_signals(model_path: str | None, images: list[Path], imgsz: int,
                      classes: list[int] | None = None) -> list[tuple[str, float, str]]:
    """Max-confidence per image for one detector, in ``images`` order."""
    if not model_path:
        return [("not_run", 0.0, "model_not_configured")] * len(images)
    try:
        from ultralytics import YOLO
        model = _DETECTOR_CACHE.get(model_path)
        if model is None:
            model = YOLO(model_path)
            _DETECTOR_CACHE[model_path] = model
    except Exception as exc:  # pragma: no cover - optional dependency issue
        return [("unavailable", 0.0, f"detector_load_error={exc}")] * len(images)

    kwargs: dict = {"conf": 0.01, "imgsz": imgsz, "verbose": False}
    scope = "all_classes"
    if classes is not None:
        kwargs["classes"] = list(classes)
        scope = "classes=" + "|".join(str(c) for c in classes)

    out: list[tuple[str, float, str]] = []
    for start in range(0, len(images), BATCH):
        chunk = images[start:start + BATCH]
        try:
            results = model.predict([str(p) for p in chunk], **kwargs)
        except Exception as exc:  # pragma: no cover - runtime/decode issue
            out.extend([("unavailable", 0.0, f"detector_error={exc}")] * len(chunk))
            continue
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                out.append(("no_candidate", 0.0,
                            f"no_boxes_at_conf_0.01;imgsz={imgsz};{scope}"))
                continue
            confs = [float(c) for c in boxes.conf.cpu().numpy().tolist()]
            out.append(("candidate", max(confs),
                        f"n_boxes={len(confs)};imgsz={imgsz};{scope};threshold=0.01"))
    return out


def build(out_dir: Path, trained_model: str | None = None, coco_model: str | None = None,
          trained_imgsz: int = 1280, coco_imgsz: int = 640,
          flag_threshold: float = 0.25) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    unique, counts = _dedupe(_source_records())
    ordered = sorted(unique, key=lambda x: str(x["image"]))
    images = [rec["image"] for rec in ordered]

    trained_signals = _detector_signals(trained_model, images, trained_imgsz)
    coco_signals = _detector_signals(coco_model, images, coco_imgsz,
                                     classes=[COCO_SPORTS_BALL])

    rows = []
    for rec, (trained_status, trained_conf, trained_evidence), \
            (coco_status, coco_conf, coco_evidence) in zip(ordered, trained_signals, coco_signals):
        bucket = "labeled_positive" if rec["labeled"] else "ambiguous"
        reason = "existing_label_requires_human_box_QA" if rec["labeled"] else "unlabeled_source_image;human_signoff_required"
        heuristic, heuristic_evidence = _heuristic(rec["image"])
        if not rec["labeled"]:
            flags = []
            if trained_status == "candidate" and trained_conf >= flag_threshold:
                flags.append("trained_ball_candidate")
            if coco_status == "candidate" and coco_conf >= flag_threshold:
                flags.append("coco_sports_ball_candidate")
            if heuristic == "non_football_or_unknown_candidate":
                flags.append("scene_not_green_candidate")
            reason += f";flag_threshold={flag_threshold:.2f}"
            reason += ";flags=" + ("|".join(flags) if flags else "none")
        rows.append({
            "bucket": bucket, "human_decision": "REVIEW_REQUIRED", "reason": reason,
            "source": rec["source"], "source_split": rec["source_split"],
            "image": str(rec["image"].resolve()), "sha1": rec["sha1"],
            "label": str(rec["label"].resolve()) if rec["label"] else "",
            "label_rows": rec["label_rows"], "heuristic": heuristic,
            "heuristic_evidence": heuristic_evidence, "trained_status": trained_status,
            "trained_conf": f"{trained_conf:.6f}", "trained_evidence": trained_evidence,
            "coco_status": coco_status, "coco_conf": f"{coco_conf:.6f}",
            "coco_evidence": coco_evidence, "review_notes": "",
        })
    fields = list(rows[0]) if rows else []
    with (out_dir / "triage.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    unlabeled = [r for r in rows if r["bucket"] == "ambiguous"]

    def _signal_stats(prefix: str, subset: list[dict]) -> dict:
        statuses = Counter(r[f"{prefix}_status"] for r in subset)
        scored = [float(r[f"{prefix}_conf"]) for r in subset
                  if r[f"{prefix}_status"] == "candidate"]
        return {
            "status_counts": dict(statuses),
            "rows_with_signal": sum(v for k, v in statuses.items()
                                    if k in ("candidate", "no_candidate")),
            "at_threshold": {f"{t:.2f}": sum(1 for c in scored if c >= t)
                             for t in REPORT_THRESHOLDS},
        }

    summary = {"counts": counts, "bucket_counts": dict(Counter(r["bucket"] for r in rows)),
               "automated_signals_are_candidates_only": True,
               "models": {"trained": trained_model, "coco": coco_model},
               "detector_config": {"trained_imgsz": trained_imgsz, "coco_imgsz": coco_imgsz,
                                   "coco_classes": [COCO_SPORTS_BALL],
                                   "flag_threshold": flag_threshold},
               "signals_all_rows": {"trained": _signal_stats("trained", rows),
                                    "coco": _signal_stats("coco", rows)},
               "signals_unlabeled_rows": {"n_rows": len(unlabeled),
                                          "trained": _signal_stats("trained", unlabeled),
                                          "coco": _signal_stats("coco", unlabeled)}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Ball_dataset_cleaned_v1 (review staging)\n\n"
        "This is a non-mutating derived review set. `triage.csv` is the source-of-truth\n"
        "for every deduplicated image. Existing positives are `labeled_positive` and\n"
        "still require box/duplicate QA. Every source negative is intentionally\n"
        "`ambiguous` until a human records `human_decision` as true_negative,\n"
        "true_positive, non_football, or ambiguous with a reason. Detector and\n"
        "scene signals are candidate evidence only; no threshold auto-labels rows.\n",
        encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "datasets" / "derived" / "Ball_dataset_cleaned_v1")
    ap.add_argument("--trained-model")
    ap.add_argument("--coco-model")
    ap.add_argument("--trained-imgsz", type=int, default=1280,
                    help="must match the trained checkpoint's training imgsz")
    ap.add_argument("--coco-imgsz", type=int, default=640)
    ap.add_argument("--flag-threshold", type=float, default=0.25,
                    help="review-queue hint only; never assigns a bucket")
    args = ap.parse_args()
    result = build(args.out, args.trained_model, args.coco_model,
                   trained_imgsz=args.trained_imgsz, coco_imgsz=args.coco_imgsz,
                   flag_threshold=args.flag_threshold)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
