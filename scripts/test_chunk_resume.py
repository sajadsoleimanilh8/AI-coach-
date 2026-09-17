"""
Live chunk-and-resume test for part-wise training.

scripts/verify_parts.py proves the ARITHMETIC (coverage, contiguity,
ledger). This script proves the RUNTIME behaviour that arithmetic cannot:
that training actually stops at an epoch boundary and that the next Part
genuinely resumes from last.pt instead of restarting from scratch.

Runs against the small ball dataset at low resolution so it takes minutes,
not hours. Both SSC_RUNS_ROOT and SSC_MODEL_ROOT must point at scratch
directories -- otherwise finalize() would publish this throwaway 6-epoch
model over a real trained checkpoint. The script refuses to run if they
are not set.

    python -m scripts.test_chunk_resume
"""

from __future__ import annotations

import os
from pathlib import Path

MODEL = "ball"
TOTAL_EPOCHS = 6
TOTAL_PARTS = 3
OVERRIDES = {"epochs": TOTAL_EPOCHS, "imgsz": 320, "batch": 8}

results: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    results.append((label, ok, detail))
    print(f"\n  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""),
          flush=True)


def main() -> int:
    if not os.getenv("SSC_RUNS_ROOT") or not os.getenv("SSC_MODEL_ROOT"):
        print("REFUSING TO RUN: set SSC_RUNS_ROOT and SSC_MODEL_ROOT to scratch "
              "paths first, or this test would overwrite real checkpoints.")
        return 2

    from ai.computer_vision.train_parts import PartsState, checkpoint_state, finalize, plan_parts, run_part
    from configs import registry as R

    spec = R.get_model(MODEL)
    run_dir = R.runs_root() / spec.run_name
    last_pt = run_dir / "weights" / "last.pt"
    plans = plan_parts(TOTAL_EPOCHS, TOTAL_PARTS)
    print(f"scratch runs : {R.runs_root()}")
    print(f"scratch model: {R.model_root()}")
    print("plan         : " + "  ".join(
        f"P{p.part}:{p.start_epoch}-{p.end_epoch}" for p in plans))

    # ---- 1. Part 1 stops exactly at its epoch boundary ---------------
    print("\n" + "=" * 62 + "\n  STEP 1: run Part 1 (epochs 1-2)\n" + "=" * 62)
    r1 = run_part(MODEL, 1, TOTAL_PARTS, overrides=OVERRIDES)
    e1 = checkpoint_state(last_pt)[0]
    check("Part 1 completes", r1["status"] == "completed", r1["status"])
    check("Part 1 stops exactly at epoch 2", e1 == 2, f"checkpoint epoch={e1}")
    check("Part 1 does not report all parts done",
          r1["all_parts_done"] is False, f"remaining={r1['remaining_parts']}")

    # ---- 2. ordering guard: cannot skip a Part -----------------------
    print("\n" + "=" * 62 + "\n  STEP 2: try Part 3 while Part 2 is missing\n" + "=" * 62)
    try:
        run_part(MODEL, 3, TOTAL_PARTS, overrides=OVERRIDES)
        check("Part 3 refused while Part 2 incomplete", False, "it ran anyway")
    except RuntimeError as exc:
        check("Part 3 refused while Part 2 incomplete", True, str(exc)[:90])

    # ---- 3. Part 2 RESUMES from the checkpoint -----------------------
    print("\n" + "=" * 62 + "\n  STEP 3: run Part 2 (epochs 3-4) -- must resume\n" + "=" * 62)
    before = checkpoint_state(last_pt)[0]
    r2 = run_part(MODEL, 2, TOTAL_PARTS, overrides=OVERRIDES)
    e2 = checkpoint_state(last_pt)[0]
    check("Part 2 completes", r2["status"] == "completed", r2["status"])
    check("Part 2 resumed (did not restart at epoch 1)",
          before == 2 and e2 == 4, f"{before} -> {e2}")
    check("Part 2 trained exactly 2 epochs", r2["epochs_run"] == 2,
          f"epochs_run={r2['epochs_run']}")

    # ---- 4. completed Part is skipped, not retrained -----------------
    print("\n" + "=" * 62 + "\n  STEP 4: re-run Part 2 (should skip)\n" + "=" * 62)
    r2b = run_part(MODEL, 2, TOTAL_PARTS, overrides=OVERRIDES)
    e2b = checkpoint_state(last_pt)[0]
    check("re-running a completed Part is skipped",
          r2b["status"] == "already_completed", r2b["status"])
    check("skip did not touch the checkpoint", e2b == 4, f"epoch still {e2b}")

    # ---- 5. final Part triggers finalize ------------------------------
    print("\n" + "=" * 62 + "\n  STEP 5: run Part 3 (epochs 5-6)\n" + "=" * 62)
    r3 = run_part(MODEL, 3, TOTAL_PARTS, overrides=OVERRIDES)
    e3 = r3["checkpoint_epoch"]      # last.pt is stripped after the final part
    _, finalized = checkpoint_state(last_pt)
    check("Part 3 completes", r3["status"] == "completed", r3["status"])
    check("all 6 epochs trained across 3 parts", e3 == TOTAL_EPOCHS,
          f"checkpoint epoch={e3}")
    check("final part finalized the checkpoint (optimizer stripped)",
          finalized is True, f"finalized={finalized}")
    check("all_parts_done now true", r3["all_parts_done"] is True,
          f"completed={r3['completed_parts']}")

    # ---- 6. ledger survives a reload (reboot simulation) -------------
    reloaded = PartsState(run_dir / "parts_state.json")
    check("ledger on disk records all 3 parts",
          reloaded.completed() == {1, 2, 3}, f"{sorted(reloaded.completed())}")

    # ---- 7. finalize publishes + integrity report ---------------------
    print("\n" + "=" * 62 + "\n  STEP 6: finalize\n" + "=" * 62)
    payload = finalize(MODEL, TOTAL_PARTS)
    check("finalize published the checkpoint",
          Path(payload["weights"]).exists(), payload["weights"])
    check("finalize recorded full epoch count",
          payload["final_checkpoint_epoch"] == TOTAL_EPOCHS,
          f"{payload['final_checkpoint_epoch']}/{TOTAL_EPOCHS}")
    check("finalize evaluated on the test split",
          "box" in payload.get("metrics", {}),
          str(payload.get("metrics", {}).get("box", {}))[:80])

    print("\n" + "=" * 62 + "\n  RESULT\n" + "=" * 62)
    failed = [r for r in results if not r[1]]
    for label, ok, _detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"\n  {len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
