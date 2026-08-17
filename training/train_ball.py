"""CLI trainer for the 'ball' model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.computer_vision.train_common import train_model  # noqa: E402

MODEL = "ball"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, help="override configs/models.yaml")
    ap.add_argument("--batch", type=int, help="override configs/models.yaml")
    ap.add_argument("--imgsz", type=int, help="override configs/models.yaml")
    ap.add_argument("--device", help="e.g. 0, 0,1, cpu")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    overrides = {k: v for k, v in vars(args).items()
                 if k != "resume" and v is not None}
    train_model(MODEL, resume=args.resume, overrides=overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
