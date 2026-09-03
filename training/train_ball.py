"""CLI trainer for the 'ball' model.

Supports part-wise ("phased") training: split the full schedule into N
contiguous epoch ranges and run one at a time, letting the laptop cool
between them. Each Part auto-resumes from the shared checkpoint.

    python -m training.train_ball --status
    python -m training.train_ball --part 1 --total-parts 4
    python -m training.train_ball --part 2 --total-parts 4
    ...
    python -m training.train_ball --part 4 --total-parts 4   # auto-finalizes

Run with no --part for the old uninterrupted single-run behaviour.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.computer_vision.train_common import train_model          # noqa: E402
from ai.computer_vision.train_parts import (                     # noqa: E402
    PartsState, all_parts_complete, checkpoint_epoch, finalize, plan_parts, run_part,
)
from configs import registry as R                                # noqa: E402

MODEL = "ball"
DEFAULT_TOTAL_PARTS = 4


def _print_status(total_parts: int | None) -> None:
    spec = R.get_model(MODEL)
    run_dir = R.runs_root() / spec.run_name
    state = PartsState(run_dir / "parts_state.json")
    total_epochs = state.data.get("total_epochs") or spec.train["epochs"]
    total_parts = total_parts or state.data.get("total_parts") or DEFAULT_TOTAL_PARTS
    ckpt = checkpoint_epoch(run_dir / "weights" / "last.pt")

    print(f"\n  {MODEL} -- part status")
    print(f"  total epochs   : {total_epochs}")
    print(f"  checkpoint at  : {ckpt if ckpt is not None else 'not started'}")
    print(f"  {'part':<6}{'epochs':<14}{'status':<14}{'ran':<6}")
    done = state.completed()
    for p in plan_parts(total_epochs, total_parts):
        rec = state.data.get("parts", {}).get(str(p.part), {})
        st = rec.get("status", "pending")
        print(f"  {p.part:<6}{f'{p.start_epoch}-{p.end_epoch}':<14}{st:<14}"
              f"{rec.get('epochs_run', 0):<6}")
    ok, missing = all_parts_complete(MODEL, total_parts)
    print(f"\n  completed: {sorted(done) or 'none'}   remaining: {missing or 'none'}")
    if ok:
        print("  All parts done -- run with --finalize to publish the model.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", type=int, help="which Part to run (1-based)")
    ap.add_argument("--total-parts", type=int, default=DEFAULT_TOTAL_PARTS,
                    help=f"how many Parts the run is split into (default {DEFAULT_TOTAL_PARTS})")
    ap.add_argument("--status", action="store_true", help="show part progress and exit")
    ap.add_argument("--finalize", action="store_true",
                    help="evaluate + publish after all Parts are complete")
    ap.add_argument("--force", action="store_true",
                    help="re-run a completed Part / finalize despite gaps")
    ap.add_argument("--epochs", type=int, help="override configs/models.yaml")
    ap.add_argument("--batch", type=int, help="override configs/models.yaml")
    ap.add_argument("--imgsz", type=int, help="override configs/models.yaml")
    ap.add_argument("--device", help="e.g. 0, cpu")
    ap.add_argument("--resume", action="store_true",
                    help="uninterrupted mode only; Parts always auto-resume")
    args = ap.parse_args()

    overrides = {k: v for k, v in vars(args).items()
                 if k in ("epochs", "batch", "imgsz", "device") and v is not None}

    if args.status:
        _print_status(args.total_parts)
        return 0

    if args.finalize:
        finalize(MODEL, args.total_parts, force=args.force)
        return 0

    if args.part is not None:
        result = run_part(MODEL, args.part, args.total_parts,
                          force=args.force, overrides=overrides)

        if result["status"] not in ("completed", "already_completed"):
            print(f"\n  Part {args.part} did not complete ({result['status']}). "
                  f"Re-run the SAME command to continue from the checkpoint.")
            return 1

        if result["all_parts_done"]:
            print(f"\n  All {args.total_parts} parts complete. Finalizing "
                  f"automatically (evaluate + publish + integrity check)...")
            finalize(MODEL, args.total_parts)
            print("\n  DONE. The ball model is trained and published.")
        else:
            nxt = result["remaining_parts"][0]
            print(f"\n  STOPPING HERE. Part {args.part} is done and nothing else "
                  f"will start automatically.\n"
                  f"  Let the laptop cool, then run Part {nxt} when ready:\n\n"
                  f"      python -m training.train_ball "
                  f"--part {nxt} --total-parts {args.total_parts}\n")
        return 0

    train_model(MODEL, resume=args.resume, overrides=overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
