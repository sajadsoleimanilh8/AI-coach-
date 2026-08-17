"""Render the calibration debugger without starting the API server."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.api.calibration_debug import _read_frame, render_calibration_debug  # noqa: E402


@dataclass
class _DebugPlayer:
    """The three fields render_calibration_debug() reads off a detection."""
    player_id: int
    pixel_x: float
    pixel_y: float


def _detect_players(frame, conf: float) -> list[_DebugPlayer]:
    from ultralytics import YOLO

    from configs import registry as R

    model = YOLO(str(R.checkpoint_path("player")))
    result = model.predict(frame, conf=conf, verbose=False)[0]
    players = []
    for i, box in enumerate(result.boxes.xyxy.cpu().numpy().tolist()):
        x1, y1, x2, y2 = box
        players.append(_DebugPlayer(player_id=i, pixel_x=(x1 + x2) / 2.0, pixel_y=y2))
    return players


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image")
    source.add_argument("--video")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--with-players", action="store_true",
                        help="run the player detector and overlay each detection's "
                             "foot point, plus its projected pitch position when "
                             "the calibration is valid")
    parser.add_argument("--player-conf", type=float, default=0.25)
    args = parser.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"could not read image: {args.image}")
    else:
        frame = _read_frame(args.video, args.frame)

    players = _detect_players(frame, args.player_conf) if args.with_players else []
    png, metadata = render_calibration_debug(frame, players)
    metadata["n_players_detected"] = len(players)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(png)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"overlay": str(output), **metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
