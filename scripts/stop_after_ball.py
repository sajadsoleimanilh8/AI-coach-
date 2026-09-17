"""
One-shot watcher: let the in-flight ball training finish, then stop the
train_all process before it auto-starts the next model.

WHY THIS IS NEEDED
    scripts/train_all.py was launched with `player` already removed from
    its ORDER list, but the process had ALREADY imported that list into
    memory as ["ball", "calibration", "field", "goalpost"]. Editing the
    file now cannot change the behaviour of the running process, so the
    only way to honour "stop after ball" without killing ball mid-epoch is
    to watch its log and terminate the process tree at the right moment.

WHEN IT IS SAFE TO KILL
    train_all prints "=== <model> finished in ..." only AFTER
    train_common.train_model() has returned, and that function does, in
    order: train -> validate on the test split -> copy best.pt to the
    registry checkpoint path -> write metrics.json. So by the time that
    line appears, every ball artefact is already durably on disk and the
    process can be terminated with nothing lost.

    As a backstop it also triggers on "=== TRAINING calibration", in case
    the finish line is missed; at that point calibration has done at most a
    few seconds of setup and no meaningful work is discarded.

    python -m scripts.stop_after_ball --pid 12320
"""

from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "runs" / "train_all.log"
DONE_MARKERS = ("=== ball finished", "=== TRAINING calibration")


def alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                         capture_output=True, text=True)
    return str(pid) in out.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--poll", type=float, default=10.0)
    ap.add_argument("--timeout", type=float, default=7200.0)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[watch] waiting for ball to finish; will then stop PID {args.pid}",
          flush=True)

    while time.time() - t0 < args.timeout:
        if not alive(args.pid):
            print("[watch] process already exited; nothing to stop.", flush=True)
            return 0
        try:
            text = LOG.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        hit = next((m for m in DONE_MARKERS if m in text), None)
        if hit:
            print(f"[watch] marker seen: {hit!r}", flush=True)
            # brief grace so any final flush completes
            time.sleep(5)
            subprocess.run(["taskkill", "/PID", str(args.pid), "/T", "/F"],
                           capture_output=True, text=True)
            print(f"[watch] stopped train_all at "
                  f"{datetime.now().isoformat(timespec='seconds')}", flush=True)
            print("[watch] calibration/field/goalpost/player NOT started.",
                  flush=True)
            return 0
        time.sleep(args.poll)

    print("[watch] timed out; left the process running untouched.", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
