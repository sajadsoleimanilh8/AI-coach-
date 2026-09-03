"""
Tracking-quality evaluation harness.

Measures identity stability of the player tracker on a video clip and
writes an ID-labelled overlay so the numbers can be eyeballed.

    venv/Scripts/python.exe scripts/eval_tracking.py --video videos/test1.mp4
        --frames 300 --tracker bytetrack.yaml --label baseline

THERE IS NO TRACKING GROUND TRUTH IN THIS REPO, so "ID switches" here is a
PROXY, computed by an offline identity linker that is deliberately more
permissive than anything running in production:

  * A track that dies at frame f and a *different* track that is born
    within GAP_MAX_S seconds is treated as the same real player when the
    newborn's first box sits near where the dead track was heading
    (constant-velocity extrapolation, distance scaled by box height), the
    two boxes are of comparable size, and their jersey-colour features
    match. Every such link is one identity handover -- i.e. one proxy ID
    switch.
  * Chaining those links gives identity clusters. A cluster is one real
    player's on-screen episode, so "on-screen duration" is the cluster's
    span and "track length" is the individual track's span.

Two honesty notes, because this proxy is not free of circularity:

  1. reid_merge.py links tracks by the same *kind* of evidence, so a merger
     tuned to exactly this linker's thresholds would score well by
     construction. The linker therefore runs looser gates than the merger
     (wider gap, wider radius, offline rather than causal) and the report
     also carries metrics the merger cannot flatter: unique track count,
     ghost-track count, and IDs-per-concurrent-player.
  2. A hard camera cut legitimately resets identity -- no appearance-only
     method can re-pair 11 identically-dressed players across one. Links
     whose gap spans a detected cut are excluded from the switch count and
     reported separately as `cut_resets`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.computer_vision.player_tracking.tracker import TrackedDetection, track_video
from ai.computer_vision.tactical_analysis.team_assignment import crop_to_feature
from configs import registry

PLAYER_CLASSES = ("player", "goalkeeper")

# --- offline identity-linker gates (evaluation only; see module docstring) ---
GAP_MAX_S = 2.0          # a newborn track this long after a death can still be the same player
DIST_MAX_HEIGHTS = 3.5   # prediction-to-birth distance, in multiples of the dead box's height
HEIGHT_RATIO_RANGE = (0.55, 1.80)
APPEARANCE_MAX = 0.40    # euclidean distance in crop_to_feature()'s 3-D jersey space
VELOCITY_SAMPLE = 5      # frames averaged for the constant-velocity prediction
VELOCITY_MAX_PX = 40.0   # per-frame velocity clamp, so a jittery last box can't fling the prediction
APPEARANCE_CROPS = 12    # crops sampled per track for its colour signature

GHOST_MAX_FRAMES = 3     # a track shorter than this is a ghost


@dataclass
class TrackSummary:
    track_id: int
    frames: list[int] = field(default_factory=list)
    boxes: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)

    @property
    def first(self) -> int:
        return self.frames[0]

    @property
    def last(self) -> int:
        return self.frames[-1]

    @property
    def span(self) -> int:
        return self.last - self.first + 1

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def centre(self, frame: int) -> tuple[float, float]:
        x, y, w, h = self.boxes[frame]
        return (x + w / 2.0, y + h / 2.0)

    def height(self, frame: int) -> float:
        return self.boxes[frame][3]


def summarise_tracks(frames: list[list[TrackedDetection]]) -> dict[int, TrackSummary]:
    """One TrackSummary per player track id, frames in ascending order."""
    tracks: dict[int, TrackSummary] = {}
    for frame_number, detections in enumerate(frames):
        for det in detections:
            if det.class_name not in PLAYER_CLASSES:
                continue
            summary = tracks.setdefault(det.player_id, TrackSummary(det.player_id))
            if frame_number not in summary.boxes:
                summary.frames.append(frame_number)
            summary.boxes[frame_number] = (det.x, det.y, det.width, det.height)
    for summary in tracks.values():
        summary.frames.sort()
    return tracks


def detect_cut_frames(video_path: str, n_frames: int) -> set[int]:
    """Shot boundaries, from the same detector the production merger uses.

    Deliberately NOT a second implementation: if the evaluator disagreed
    with reid_merge.py about where the cuts are, the cut_resets column
    would be measuring the disagreement rather than the tracker.
    """
    from ai.computer_vision.player_tracking.reid_merge import hard_cut_frames

    return hard_cut_frames(video_path, n_frames)


def track_appearances(video_path: str,
                      frames: list[list[TrackedDetection]],
                      tracks: dict[int, TrackSummary]) -> dict[int, np.ndarray]:
    """Median jersey-colour feature per track.

    Crops are taken from frames spread evenly across each track's life, so
    one badly-occluded moment cannot define a track's colour. The median
    (not the mean) makes a handful of grass/limb-heavy crops harmless.
    """
    wanted: dict[int, list[int]] = {}
    for track_id, summary in tracks.items():
        picks = summary.frames
        if len(picks) > APPEARANCE_CROPS:
            idx = np.linspace(0, len(picks) - 1, APPEARANCE_CROPS).round().astype(int)
            picks = [picks[i] for i in sorted(set(idx.tolist()))]
        for frame_number in picks:
            wanted.setdefault(frame_number, []).append(track_id)

    features: dict[int, list[np.ndarray]] = {}
    cap = cv2.VideoCapture(video_path)
    try:
        for frame_number in range(len(frames)):
            ok, raw = cap.read()
            if not ok or raw is None:
                break
            if frame_number not in wanted:
                continue
            frame_h, frame_w = raw.shape[:2]
            for track_id in wanted[frame_number]:
                x, y, w, h = tracks[track_id].boxes[frame_number]
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(frame_w, int(x + w)), min(frame_h, int(y + h))
                if x2 <= x1 or y2 <= y1:
                    continue
                feat = crop_to_feature(raw[y1:y2, x1:x2])
                if feat is not None:
                    features.setdefault(track_id, []).append(feat)
    finally:
        cap.release()

    return {tid: np.median(np.stack(f), axis=0) for tid, f in features.items() if f}


def _velocity(summary: TrackSummary) -> tuple[float, float]:
    """Mean per-frame centre velocity over the track's last VELOCITY_SAMPLE frames."""
    tail = summary.frames[-VELOCITY_SAMPLE:]
    if len(tail) < 2:
        return (0.0, 0.0)
    (x0, y0), (x1, y1) = summary.centre(tail[0]), summary.centre(tail[-1])
    dt = tail[-1] - tail[0]
    vx, vy = (x1 - x0) / dt, (y1 - y0) / dt
    speed = float(np.hypot(vx, vy))
    if speed > VELOCITY_MAX_PX:
        vx, vy = vx * VELOCITY_MAX_PX / speed, vy * VELOCITY_MAX_PX / speed
    return (vx, vy)


def link_identities(tracks: dict[int, TrackSummary],
                    appearances: dict[int, np.ndarray],
                    fps: float,
                    cut_frames: set[int]) -> tuple[list[dict], list[dict]]:
    """Offline one-to-one linking of died->born track pairs.

    Returns (links, cut_resets): links are proxy ID switches, cut_resets are
    the pairs rejected only because a hard camera cut sits in the gap.
    """
    gap_max = max(1, int(round(GAP_MAX_S * fps)))
    sorted_cuts = sorted(cut_frames)

    def cut_between(a: int, b: int) -> bool:
        return any(a < c <= b for c in sorted_cuts)

    candidates: list[tuple[float, int, int, bool]] = []
    for a_id, a in tracks.items():
        vx, vy = _velocity(a)
        ax, ay = a.centre(a.last)
        a_h = a.height(a.last)
        if a_h <= 0:
            continue
        for b_id, b in tracks.items():
            if b_id == a_id:
                continue
            gap = b.first - a.last
            if gap < 1 or gap > gap_max:
                continue
            bx, by = b.centre(b.first)
            b_h = b.height(b.first)
            if b_h <= 0:
                continue
            ratio = b_h / a_h
            if not (HEIGHT_RATIO_RANGE[0] <= ratio <= HEIGHT_RATIO_RANGE[1]):
                continue
            px, py = ax + vx * gap, ay + vy * gap
            dist = float(np.hypot(bx - px, by - py)) / a_h
            if dist > DIST_MAX_HEIGHTS:
                continue
            fa, fb = appearances.get(a_id), appearances.get(b_id)
            if fa is None or fb is None:
                continue
            app = float(np.linalg.norm(fa - fb))
            if app > APPEARANCE_MAX:
                continue
            cost = dist / DIST_MAX_HEIGHTS + app / APPEARANCE_MAX
            candidates.append((cost, a_id, b_id, cut_between(a.last, b.first)))

    # Greedy one-to-one, ascending cost; ids break ties so the result is
    # identical for identical input.
    candidates.sort(key=lambda c: (round(c[0], 9), c[1], c[2]))
    used_a: set[int] = set()
    used_b: set[int] = set()
    links: list[dict] = []
    cut_resets: list[dict] = []
    for cost, a_id, b_id, spans_cut in candidates:
        if a_id in used_a or b_id in used_b:
            continue
        used_a.add(a_id)
        used_b.add(b_id)
        record = {"from": a_id, "to": b_id, "cost": round(cost, 4),
                  "gap": tracks[b_id].first - tracks[a_id].last}
        (cut_resets if spans_cut else links).append(record)
    return links, cut_resets


def cluster_identities(tracks: dict[int, TrackSummary],
                       links: list[dict]) -> list[list[int]]:
    """Chain links into identity clusters (one real on-screen episode each)."""
    parent = {tid: tid for tid in tracks}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for link in links:
        ra, rb = find(link["from"]), find(link["to"])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    clusters: dict[int, list[int]] = {}
    for tid in sorted(tracks):
        clusters.setdefault(find(tid), []).append(tid)
    return [sorted(v) for _, v in sorted(clusters.items())]


def evaluate(frames: list[list[TrackedDetection]],
             video_path: str,
             fps: float) -> dict:
    tracks = summarise_tracks(frames)
    player_frames = sum(1 for dets in frames for d in dets if d.class_name in PLAYER_CLASSES)

    if not tracks:
        return {"error": "no player tracks produced", "player_frames": player_frames,
                "frames": len(frames), "fps": fps}

    cut_frames = detect_cut_frames(video_path, len(frames))
    appearances = track_appearances(video_path, frames, tracks)
    links, cut_resets = link_identities(tracks, appearances, fps, cut_frames)
    clusters = cluster_identities(tracks, links)

    track_spans = [t.span for t in tracks.values()]
    cluster_spans = []
    for cluster in clusters:
        first = min(tracks[t].first for t in cluster)
        last = max(tracks[t].last for t in cluster)
        cluster_spans.append(last - first + 1)

    ghosts = [t for t in tracks.values() if t.n_frames < GHOST_MAX_FRAMES]
    per_frame_counts = [sum(1 for d in dets if d.class_name in PLAYER_CLASSES) for dets in frames]
    visible = [c for c in per_frame_counts if c > 0]
    concurrent = float(np.median(visible)) if visible else 0.0

    mean_track = float(np.mean(track_spans))
    mean_onscreen = float(np.mean(cluster_spans))

    return {
        "frames": len(frames),
        "fps": fps,
        "player_frames": player_frames,
        "unique_tracks": len(tracks),
        "identity_clusters": len(clusters),
        "id_switches": len(links),
        "id_switches_per_1000_player_frames": round(1000.0 * len(links) / player_frames, 3)
        if player_frames else None,
        "cut_frames": len(cut_frames),
        "cut_resets": len(cut_resets),
        "mean_track_length_frames": round(mean_track, 2),
        "mean_onscreen_duration_frames": round(mean_onscreen, 2),
        "fragmentation_ratio": round(mean_track / mean_onscreen, 4) if mean_onscreen else None,
        "ghost_tracks": len(ghosts),
        "ghost_track_fraction": round(len(ghosts) / len(tracks), 4),
        "median_concurrent_players": concurrent,
        "ids_per_concurrent_player": round(len(tracks) / concurrent, 2) if concurrent else None,
        "links": links,
        "cut_reset_pairs": cut_resets,
        "cut_frame_numbers": sorted(cut_frames),
    }


def _colour(track_id: int) -> tuple[int, int, int]:
    """Stable, well-spread BGR colour per id."""
    hue = int((track_id * 47) % 180)
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def render_id_overlay(video_path: str,
                      frames: list[list[TrackedDetection]],
                      out_path: Path,
                      fps: float,
                      cut_frames: set[int]) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    ok, first = cap.read()
    if not ok:
        cap.release()
        return False
    height, width = first.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    try:
        for frame_number, detections in enumerate(frames):
            ok, raw = cap.read()
            if not ok or raw is None:
                break
            for det in detections:
                if det.class_name not in PLAYER_CLASSES:
                    continue
                colour = _colour(det.player_id)
                x1, y1 = int(det.x), int(det.y)
                x2, y2 = int(det.x + det.width), int(det.y + det.height)
                cv2.rectangle(raw, (x1, y1), (x2, y2), colour, 2)
                label = f"#{det.player_id}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(raw, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
                cv2.putText(raw, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 2, cv2.LINE_AA)
            banner = f"frame {frame_number}"
            if frame_number in cut_frames:
                banner += "  CUT"
                cv2.rectangle(raw, (0, 0), (width - 1, height - 1), (0, 0, 255), 6)
            cv2.putText(raw, banner, (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(raw)
    finally:
        writer.release()
        cap.release()
    return out_path.exists() and out_path.stat().st_size > 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default="videos/test1.mp4")
    parser.add_argument("--frames", type=int, default=300,
                        help="frames to evaluate (>=250 to be meaningful)")
    parser.add_argument("--tracker", default="bytetrack.yaml",
                        help="ultralytics tracker yaml (stock name or repo path)")
    parser.add_argument("--reid", action="store_true",
                        help="apply the reid_merge post-pass before measuring")
    parser.add_argument("--label", default=None, help="name for the output files")
    parser.add_argument("--out", default="runs/_eval/tracking")
    parser.add_argument("--device", default="0")
    parser.add_argument("--no-overlay", action="store_true")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        print(f"video not found: {video_path}", file=sys.stderr)
        return 2

    label = args.label or (Path(args.tracker).stem + ("_reid" if args.reid else ""))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    checkpoint = str(registry.checkpoint_path("player"))
    print(f"[eval] video={video_path} frames={args.frames} tracker={args.tracker} "
          f"reid={args.reid} fps={fps:.2f}")

    frames: list[list[TrackedDetection]] = []
    for detections in track_video(checkpoint, video_path,
                                  tracker_config=args.tracker, device=args.device):
        frames.append(detections)
        if len(frames) >= args.frames:
            break

    merge_report = None
    if args.reid:
        from ai.computer_vision.player_tracking.reid_merge import merge_reidentified_tracks

        result = merge_reidentified_tracks(video_path, frames, fps=fps)
        merge_report = result.as_dict()
        print(f"[eval] reid_merge: {merge_report}")

    metrics = evaluate(frames, video_path, fps)
    metrics["label"] = label
    metrics["video"] = video_path
    metrics["tracker_config"] = args.tracker
    metrics["reid_merge"] = merge_report

    if not args.no_overlay:
        overlay_path = out_dir / f"overlay_{Path(video_path).stem}_{label}.mp4"
        written = render_id_overlay(video_path, frames, overlay_path, fps,
                                    set(metrics.get("cut_frame_numbers", [])))
        metrics["overlay"] = str(overlay_path) if written else None
        print(f"[eval] overlay: {metrics['overlay']}")

    json_path = out_dir / f"metrics_{Path(video_path).stem}_{label}.json"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\n--- {label} on {video_path} ({metrics['frames']} frames) ---")
    for key in ("player_frames", "unique_tracks", "identity_clusters",
                "id_switches", "id_switches_per_1000_player_frames",
                "cut_frames", "cut_resets",
                "mean_track_length_frames", "mean_onscreen_duration_frames",
                "fragmentation_ratio", "ghost_tracks", "ghost_track_fraction",
                "median_concurrent_players", "ids_per_concurrent_player"):
        print(f"  {key:38s} {metrics.get(key)}")
    print(f"[eval] metrics written to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
