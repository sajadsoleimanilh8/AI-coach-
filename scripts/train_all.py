"""
Trains every registered model sequentially, using each model's own config.

Order is deliberate: smallest/most pipeline-critical first, so usable
checkpoints and real metrics exist early instead of after the longest run.
`player` is last because it is by far the largest (12,248 training images at
imgsz 960).

A failure in one model does not abort the rest -- the failure is recorded
and reported in the summary at the end, so an overnight run does not lose
four good models to one bad one.

    python -m scripts.train_all
    python -m scripts.train_all --only ball calibration
    python -m scripts.train_all --epochs 5        # smoke-test everything
"""

from __future__ import annotations

import argparse
import traceback
from datetime import UTC, datetime

from ai.computer_vision.train_common import train_model  # noqa: E402

# `player` is deliberately NOT in this list. It is the ~7-hour run and is
# trained in operator-controlled parts instead, one at a time with a
# cooldown between them -- see training/train_player.py's --part flag.
# Auto-starting it here would violate that workflow. Pass it explicitly
# (--only player) only if you really want one uninterrupted 7-hour run.
ORDER = ["ball", "calibration", "field", "goalpost"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", help="subset of models to train")
    ap.add_argument("--epochs", type=int, help="override every model's epochs")
    args = ap.parse_args()

    models = args.only or ORDER
    overrides = {"epochs": args.epochs} if args.epochs else {}

    results: list[tuple[str, str, str]] = []
    for name in models:
        started = datetime.now(UTC)
        print(f"\n{'=' * 70}\n=== TRAINING {name}  ({started.isoformat()})\n{'=' * 70}", flush=True)
        try:
            outcome = train_model(name, overrides=overrides)
            box = outcome.metrics.get("box", {})
            summary = (f"mAP50={box.get('map50', float('nan')):.4f} "
                       f"mAP50-95={box.get('map50_95', float('nan')):.4f} "
                       f"P={box.get('precision', float('nan')):.4f} "
                       f"R={box.get('recall', float('nan')):.4f}")
            results.append((name, "OK", summary))
        except Exception as exc:                              # noqa: BLE001
            traceback.print_exc()
            results.append((name, "FAILED", f"{type(exc).__name__}: {exc}"))
        elapsed = datetime.now(UTC) - started
        print(f"=== {name} finished in {elapsed}", flush=True)

    print(f"\n{'=' * 70}\n=== SUMMARY\n{'=' * 70}")
    for name, status, summary in results:
        print(f"{name:14s} {status:7s} {summary}")
    return 0 if all(s == "OK" for _, s, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
