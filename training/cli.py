"""Shared command-line trainer for the YOLO models in configs/models.yaml.

    python -m training.train ball --status
    python -m training.train ball --part 1 --total-parts 4
    python -m training.train ball --part 4 --total-parts 4   # auto-finalizes
    python -m training.train field --epochs 50

Part-wise ("phased") training splits the full schedule into N contiguous epoch
ranges and runs one at a time, so the laptop can cool between them. Each Part
auto-resumes from the shared checkpoint. With no --part, the whole schedule
runs uninterrupted.

The per-model modules (training.train_ball, train_player, ...) are thin
wrappers around this one. They are kept because in-progress phased runs print
their exact command for the next Part.
"""

from __future__ import annotations

import argparse

from ai.computer_vision.train_common import train_model
from ai.computer_vision.train_parts import (
    PartsState,
    all_parts_complete,
    checkpoint_epoch,
    finalize,
    plan_parts,
    run_part,
)
from configs import registry as R

# Heavier models get more, shorter Parts. Anything not listed uses 4.
DEFAULT_TOTAL_PARTS = {"player": 6}
FALLBACK_TOTAL_PARTS = 4


def default_total_parts(model: str) -> int:
    return DEFAULT_TOTAL_PARTS.get(model, FALLBACK_TOTAL_PARTS)


def print_status(model: str, total_parts: int | None) -> None:
    spec = R.get_model(model)
    run_dir = R.runs_root() / spec.run_name
    state = PartsState(run_dir / "parts_state.json")
    total_epochs = state.data.get("total_epochs") or spec.train["epochs"]
    total_parts = total_parts or state.data.get("total_parts") or default_total_parts(model)
    ckpt = checkpoint_epoch(run_dir / "weights" / "last.pt")

    print(f"\n  {model} -- part status")
    print(f"  total epochs   : {total_epochs}")
    print(f"  checkpoint at  : {ckpt if ckpt is not None else 'not started'}")
    print(f"  {'part':<6}{'epochs':<14}{'status':<14}{'ran':<6}")
    done = state.completed()
    for p in plan_parts(total_epochs, total_parts):
        rec = state.data.get("parts", {}).get(str(p.part), {})
        st = rec.get("status", "pending")
        print(f"  {p.part:<6}{f'{p.start_epoch}-{p.end_epoch}':<14}{st:<14}"
              f"{rec.get('epochs_run', 0):<6}")
    ok, missing = all_parts_complete(model, total_parts)
    print(f"\n  completed: {sorted(done) or 'none'}   remaining: {missing or 'none'}")
    if ok:
        print("  All parts done -- run with --finalize to publish the model.")


def build_parser(model: str | None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    if model is None:
        ap.add_argument("model", help="model name from configs/models.yaml (player, ball, ...)")
    ap.add_argument("--part", type=int, help="which Part to run (1-based)")
    ap.add_argument("--total-parts", type=int, default=None,
                    help="how many Parts the run is split into "
                         f"(default {FALLBACK_TOTAL_PARTS}; player {DEFAULT_TOTAL_PARTS['player']})")
    ap.add_argument("--status", action="store_true", help="show part progress and exit")
    ap.add_argument("--finalize", action="store_true",
                    help="evaluate + publish after all Parts are complete")
    ap.add_argument("--force", action="store_true",
                    help="re-run a completed Part / finalize despite gaps")
    ap.add_argument("--stop-at", type=int, default=None,
                    help="pause inside a Part at this ABSOLUTE epoch, for "
                         "splitting a Part too long to run in one sitting; "
                         "re-run the same command to finish the rest")
    ap.add_argument("--epochs", type=int, help="override configs/models.yaml")
    ap.add_argument("--batch", type=int, help="override configs/models.yaml")
    ap.add_argument("--imgsz", type=int, help="override configs/models.yaml")
    ap.add_argument("--device", help="e.g. 0, 0,1, cpu")
    ap.add_argument("--resume", action="store_true",
                    help="uninterrupted mode only; Parts always auto-resume")
    return ap


def main(argv: list[str] | None = None, *, model: str | None = None) -> int:
    args = build_parser(model).parse_args(argv)
    model = model or args.model
    R.get_model(model)  # fail fast on an unknown name, before any training work
    total_parts = args.total_parts or default_total_parts(model)
    overrides = {k: v for k, v in vars(args).items()
                 if k in ("epochs", "batch", "imgsz", "device") and v is not None}

    if args.status:
        print_status(model, args.total_parts)
        return 0

    if args.finalize:
        finalize(model, total_parts, force=args.force)
        return 0

    if args.part is not None:
        result = run_part(model, args.part, total_parts, force=args.force,
                          overrides=overrides, stop_at=args.stop_at)

        if result["status"] == "paused":
            print(f"\n  PAUSED at epoch {result['checkpoint_epoch']}. Part "
                  f"{args.part} is NOT finished -- its remaining epochs still "
                  f"have to run.\n  Continue it with:\n\n"
                  f"      python -m training.train {model} "
                  f"--part {args.part} --total-parts {total_parts}\n")
            return 0

        if result["status"] not in ("completed", "already_completed"):
            print(f"\n  Part {args.part} did not complete ({result['status']}). "
                  f"Re-run the SAME command to continue from the checkpoint.")
            return 1

        if result["all_parts_done"]:
            print(f"\n  All {total_parts} parts complete. Finalizing "
                  f"automatically (evaluate + publish + integrity check)...")
            finalize(model, total_parts)
            print(f"\n  DONE. The {model} model is trained and published.")
        else:
            nxt = result["remaining_parts"][0]
            print(f"\n  STOPPING HERE. Part {args.part} is done and nothing else "
                  f"will start automatically.\n"
                  f"  Let the laptop cool, then run Part {nxt} when ready:\n\n"
                  f"      python -m training.train_{model} "
                  f"--part {nxt} --total-parts {total_parts}\n")
        return 0

    train_model(model, resume=args.resume, overrides=overrides)
    return 0
