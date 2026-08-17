"""
One-time manual pitch calibration tool.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_TACTICAL_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TACTICAL_ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _TACTICAL_ANALYSIS_DIR)

from constants import CALIBRATION_POINT_ORDER, MIN_CALIBRATION_POINTS, REFERENCE_POINTS
from homography import compute_homography


def _load_first_frame(video_path: str, frame_number: int = 0):
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    if frame_number > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_number} from {video_path}")
    return frame


def run_interactive(video_path: str, frame_number: int) -> dict[str, tuple[float, float]]:
    """
    Opens a window over the frame and lets the operator click each
    reference point in CALIBRATION_POINT_ORDER. Returns {name: (x_px, y_px)}.
    Requires a display (cv2.imshow) -- not available in headless sandboxes.
    """
    import cv2

    frame = _load_first_frame(video_path, frame_number)
    display = frame.copy()
    clicked: dict[str, tuple[float, float]] = {}
    idx = {"i": 0}

    window = "Pitch Calibration - click the highlighted landmark"

    def redraw():
        nonlocal display
        display = frame.copy()
        for name, (x, y) in clicked.items():
            cv2.circle(display, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(display, name, (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        if idx["i"] < len(CALIBRATION_POINT_ORDER):
            current = CALIBRATION_POINT_ORDER[idx["i"]]
            cv2.putText(display, f"Click: {current}  (s=skip, u=undo, n=preview, q=finish)",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(display, "All points done. Press q to save.",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(window, display)

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and idx["i"] < len(CALIBRATION_POINT_ORDER):
            name = CALIBRATION_POINT_ORDER[idx["i"]]
            clicked[name] = (float(x), float(y))
            idx["i"] += 1
            redraw()

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s") and idx["i"] < len(CALIBRATION_POINT_ORDER):
            idx["i"] += 1
            redraw()
        elif key == ord("u") and clicked:
            last_name = CALIBRATION_POINT_ORDER[idx["i"] - 1] if idx["i"] > 0 else None
            if last_name in clicked:
                del clicked[last_name]
                idx["i"] -= 1
                redraw()
        elif key == ord("n"):
            _preview(clicked)

    cv2.destroyAllWindows()
    return clicked


def _preview(clicked: dict[str, tuple[float, float]]) -> None:
    if len(clicked) < MIN_CALIBRATION_POINTS:
        print(f"  [preview] need >= {MIN_CALIBRATION_POINTS} points, have {len(clicked)}")
        return
    pixel_pts = np.array(list(clicked.values()))
    pitch_pts = np.array([REFERENCE_POINTS[n] for n in clicked])
    result = compute_homography(pixel_pts, pitch_pts)
    print(f"  [preview] n={result.n_points}  reproj_error={result.reprojection_error_m:.3f}m  "
          f"confidence={result.confidence:.3f}")


def build_calibration_record(
    clicked: dict[str, tuple[float, float]],
    video_path: str | None,
    frame_number: int,
) -> dict:
    if len(clicked) < MIN_CALIBRATION_POINTS:
        raise ValueError(
            f"Need at least {MIN_CALIBRATION_POINTS} calibration points, got {len(clicked)}"
        )

    names = list(clicked.keys())
    pixel_pts = np.array([clicked[n] for n in names])
    pitch_pts = np.array([REFERENCE_POINTS[n] for n in names])

    method = 0
    result = compute_homography(pixel_pts, pitch_pts, method=method)

    return {
        "video_path": video_path,
        "frame_number": frame_number,
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
        "points": [
            {"name": n, "pixel": list(clicked[n]), "pitch_m": list(REFERENCE_POINTS[n])}
            for n in names
        ],
        "n_points": result.n_points,
        "homography_matrix": result.H.tolist(),
        "reprojection_error_m": result.reprojection_error_m,
        "max_reprojection_error_m": result.max_reprojection_error_m,
        "confidence": result.confidence,
        "per_point_errors_m": result.per_point_errors_m,
    }


def load_calibration(path: str) -> tuple[np.ndarray, dict]:
    """Load a saved calibration JSON. Returns (H, full_record)."""
    with open(path) as f:
        record = json.load(f)
    H = np.array(record["homography_matrix"])
    return H, record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", help="Path to match video (interactive mode)")
    parser.add_argument("--frame", type=int, default=0, help="Frame number to calibrate on")
    parser.add_argument("--points-file", help="JSON {name: [x_px, y_px]} for headless calibration")
    parser.add_argument("--out", required=True, help="Output path for calibration JSON")
    args = parser.parse_args()

    if args.points_file:
        with open(args.points_file) as f:
            raw = json.load(f)
        clicked = {}
        for name, xy in raw.items():
            if name not in REFERENCE_POINTS:
                print(f"WARNING: '{name}' is not a known reference point, skipping. "
                      f"Valid names: {sorted(REFERENCE_POINTS)}", file=sys.stderr)
                continue
            clicked[name] = (float(xy[0]), float(xy[1]))
        video_path = args.video
    elif args.video:
        clicked = run_interactive(args.video, args.frame)
        video_path = args.video
    else:
        parser.error("Provide either --video (interactive) or --points-file (headless)")
        return

    record = build_calibration_record(clicked, video_path, args.frame)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"Saved calibration: {out_path}")
    print(f"  n_points            = {record['n_points']}")
    print(f"  reprojection_error  = {record['reprojection_error_m']:.3f} m")
    print(f"  confidence          = {record['confidence']:.3f}")
    if record["confidence"] < 0.6:
        print("  WARNING: confidence below HOMOGRAPHY_CONFIDENCE_MIN (0.6) -- "
              "re-click points or pick a clearer frame before using this for analysis.")


if __name__ == "__main__":
    main()
