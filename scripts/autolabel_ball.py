"""
Auto-label the ball in broadcast videos, for retraining the 'ball' model on
the domain it actually fails on.

Pipeline per video:
  1. tiled detection  -- ball_v1 at low conf on the full frame + an overlapping
     tile grid, so a 10 px ball at 1080p is seen at an effective imgsz well
     above what one whole-frame pass gives it.
  2. trajectory linking -- start from high-confidence anchors, extend frame by
     frame to the candidate nearest the constant-velocity prediction inside a
     gate radius. Detections that never connect to an anchor are dropped.
  3. gap interpolation  -- straight-line fill across <= MAX_GAP missing frames.
  4. sampling           -- keep accepted frames on a stride, split into
     train/val/test by TIME BLOCK (first 80% / next 10% / last 10% of the
     video) so frames seconds apart never straddle the split.

Outputs, under datasets/processed/ball_datasets/<--source-name>/ :
  {train,val,test}/images/*.jpg   {train,val,test}/labels/*.txt   (YOLO, class 0)
  review/<video>_review.mp4       accepted ball drawn + frame index + tag
  manifest.csv                    one row per emitted frame

NOTHING is auto-added to a training run. Review the MP4, delete or fix bad
frames, then add "<source-name>" to configs/datasets.yaml::datasets.ball.sources.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

from configs import registry as R  # noqa: E402

MAX_GAP = 5                    # frames of interpolation across a detection gap
GATE_BASE_PX = 45.0            # trajectory gate radius at velocity 0
GATE_VEL_MULT = 2.5           # gate grows with recent speed
ANCHOR_CONF = 0.20            # a detection this strong can start a track
LINK_CONF = 0.03             # a detection this strong can extend a track
TILE_COLS, TILE_ROWS = 3, 2
TILE_OVERLAP = 0.20


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _nms(cands: list[dict], iou_thr: float = 0.4) -> list[dict]:
    cands = sorted(cands, key=lambda c: -c["conf"])
    kept: list[dict] = []
    for c in cands:
        if all(_iou(c["xyxy"], k["xyxy"]) < iou_thr for k in kept):
            kept.append(c)
    return kept


def _tiles(w: int, h: int):
    tw = int(w / (TILE_COLS - (TILE_COLS - 1) * TILE_OVERLAP))
    th = int(h / (TILE_ROWS - (TILE_ROWS - 1) * TILE_OVERLAP))
    xs = np.linspace(0, w - tw, TILE_COLS).astype(int)
    ys = np.linspace(0, h - th, TILE_ROWS).astype(int)
    return [(int(x), int(y), tw, th) for y in ys for x in xs]


def _detect_frame(model, frame, imgsz: int, device) -> list[dict]:
    h, w = frame.shape[:2]
    regions = [(0, 0, w, h)] + _tiles(w, h)
    cands: list[dict] = []
    for (rx, ry, rw, rh) in regions:
        crop = frame[ry:ry + rh, rx:rx + rw]
        res = model.predict(crop, imgsz=imgsz, conf=LINK_CONF, iou=0.4,
                            max_det=10, verbose=False,
                            **({"device": device} if device is not None else {}))[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), cf in zip(xyxy, confs):
            cands.append({
                "xyxy": (float(x1 + rx), float(y1 + ry),
                         float(x2 + rx), float(y2 + ry)),
                "conf": float(cf),
            })
    for c in cands:
        x1, y1, x2, y2 = c["xyxy"]
        c["cx"], c["cy"] = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        c["w"], c["h"] = x2 - x1, y2 - y1
    return _nms(cands)


def _link(per_frame: list[list[dict]]) -> dict[int, dict]:
    """Frame -> chosen ball dict ({cx,cy,w,h,conf,tag}). Greedy grow from anchors."""
    n = len(per_frame)
    chosen: dict[int, dict] = {}

    anchors = sorted(
        ((i, max(per_frame[i], key=lambda c: c["conf"]))
         for i in range(n) if per_frame[i]
         and max(c["conf"] for c in per_frame[i]) >= ANCHOR_CONF),
        key=lambda t: -t[1]["conf"],
    )

    for start, cand in anchors:
        if start in chosen:
            continue
        chosen[start] = {**cand, "tag": "auto"}
        for direction in (1, -1):
            vx = vy = 0.0
            px, py = cand["cx"], cand["cy"]
            i = start + direction
            miss = 0
            while 0 <= i < n:
                if i in chosen:
                    break
                pred_x, pred_y = px + vx * direction, py + vy * direction
                gate = GATE_BASE_PX + GATE_VEL_MULT * float(np.hypot(vx, vy))
                pool = [c for c in per_frame[i]
                        if np.hypot(c["cx"] - pred_x, c["cy"] - pred_y) <= gate]
                if not pool:
                    miss += 1
                    if miss > MAX_GAP:
                        break
                    px, py = pred_x, pred_y
                    i += direction
                    continue
                pick = min(pool, key=lambda c: np.hypot(c["cx"] - pred_x, c["cy"] - pred_y))
                chosen[i] = {**pick, "tag": "auto"}
                nvx, nvy = (pick["cx"] - px) / max(miss + 1, 1), (pick["cy"] - py) / max(miss + 1, 1)
                vx, vy = 0.6 * vx + 0.4 * nvx, 0.6 * vy + 0.4 * nvy
                px, py = pick["cx"], pick["cy"]
                miss = 0
                i += direction
    return chosen


def _interpolate(chosen: dict[int, dict], n: int) -> dict[int, dict]:
    keys = sorted(chosen)
    for a, b in zip(keys, keys[1:]):
        gap = b - a - 1
        if gap <= 0 or gap > MAX_GAP:
            continue
        ca, cb = chosen[a], chosen[b]
        for s in range(1, gap + 1):
            t = s / (b - a)
            chosen[a + s] = {
                "cx": ca["cx"] + (cb["cx"] - ca["cx"]) * t,
                "cy": ca["cy"] + (cb["cy"] - ca["cy"]) * t,
                "w": (ca["w"] + cb["w"]) / 2.0, "h": (ca["h"] + cb["h"]) / 2.0,
                "conf": None, "tag": "interp",
            }
    return chosen


def _split_for(frame_idx: int, total: int) -> str:
    r = frame_idx / max(total - 1, 1)
    return "train" if r < 0.8 else ("val" if r < 0.9 else "test")


def process_video(video: Path, model, out_root: Path, device, *, imgsz: int,
                  stride: int, cap_frames: int, writer_rows: list[dict]) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    per_frame: list[list[dict]] = []
    frames_cache: dict[int, np.ndarray] = {}
    idx = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            per_frame.append(_detect_frame(model, fr, imgsz, device))
            frames_cache[len(per_frame) - 1] = fr
        idx += 1
    cap.release()

    chosen = _interpolate(_link(per_frame), len(per_frame))
    n_sampled = len(per_frame)

    emitted = {"train": 0, "val": 0, "test": 0}
    stem = video.stem
    review_frames: list[np.ndarray] = []
    accepted = sorted(chosen)
    if cap_frames and len(accepted) > cap_frames:
        step = len(accepted) / cap_frames
        accepted = [accepted[int(i * step)] for i in range(cap_frames)]

    for si in range(n_sampled):
        fr = frames_cache[si]
        ball = chosen.get(si)
        if si in accepted and ball is not None:
            h, w = fr.shape[:2]
            split = _split_for(si, n_sampled)
            name = f"{stem}_{si:06d}"
            (out_root / split / "images").mkdir(parents=True, exist_ok=True)
            (out_root / split / "labels").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_root / split / "images" / f"{name}.jpg"), fr)
            cx, cy = ball["cx"] / w, ball["cy"] / h
            bw, bh = max(ball["w"], 6) / w, max(ball["h"], 6) / h
            (out_root / split / "labels" / f"{name}.txt").write_text(
                f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")
            emitted[split] += 1
            writer_rows.append({
                "video": video.name, "sampled_frame": si,
                "source_frame": si * stride, "split": split,
                "tag": ball["tag"], "conf": ball.get("conf"),
                "cx_px": round(ball["cx"], 1), "cy_px": round(ball["cy"], 1),
            })

        if ball is not None:
            vis = fr.copy()
            c = (0, 255, 255) if ball["tag"] == "auto" else (255, 128, 0)
            cv2.circle(vis, (int(ball["cx"]), int(ball["cy"])), 9, c, 2)
            cv2.putText(vis, f"{si} {ball['tag']} "
                        f"{'' if ball.get('conf') is None else round(ball['conf'], 2)}",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
            review_frames.append(vis)

    if review_frames:
        rev_dir = out_root / "review"
        rev_dir.mkdir(parents=True, exist_ok=True)
        h, w = review_frames[0].shape[:2]
        vw = cv2.VideoWriter(str(rev_dir / f"{stem}_review.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), min(fps, 12.0), (w, h))
        for v in review_frames:
            vw.write(v)
        vw.release()

    return {"video": video.name, "sampled": n_sampled,
            "linked": len(chosen), "emitted": emitted}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--source-name", default="auto_broadcast_v1")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--stride", type=int, default=6,
                    help="detect every Nth video frame")
    ap.add_argument("--cap-per-video", type=int, default=400)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO

    ckpt = R.checkpoint_path("ball")
    model = YOLO(str(ckpt))
    out_root = R.dataset_root() / "ball_datasets" / args.source_name
    print(f"[autolabel] model   : {ckpt}")
    print(f"[autolabel] out     : {out_root}")

    rows: list[dict] = []
    summary = []
    for v in args.videos:
        print(f"[autolabel] {v} ...", flush=True)
        summary.append(process_video(v, model, out_root, args.device,
                                     imgsz=args.imgsz, stride=args.stride,
                                     cap_frames=args.cap_per_video, writer_rows=rows))

    if rows:
        with (out_root / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    print("\n[autolabel] done")
    for s in summary:
        print(f"  {s['video']:24s} sampled={s['sampled']:5d} linked={s['linked']:5d} "
              f"emitted={s['emitted']}")
    print(f"\nReview: {out_root / 'review'}")
    print(f"Then add '{args.source_name}' to configs/datasets.yaml::datasets.ball.sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
