"""
Part-wise ("chunked") training with reboot-safe resume.

PURPOSE
    The player model is a ~7-hour run on a laptop. Running it in one go
    cooks the machine. This module splits it into N operator-controlled
    Parts that are run ONE AT A TIME, with an arbitrary cooldown between
    them, and no automatic progression to the next Part.

WHY PARTS ARE EPOCH RANGES, NOT FILE RANGES
    This is the load-bearing design decision, so it is stated plainly.

    For a file-wise preprocessing job, "Part 3 of 6" means "the third
    sixth of the files", and the parts are independent. Training is not
    like that. Splitting 12,248 images into 6 disjoint sets and training
    on each separately produces six models that have each seen one sixth
    of the data -- or, run sequentially against one checkpoint, a model
    that progressively forgets the earlier fifths. Either way the result
    is materially worse than training on the whole set, and its metrics
    still look plausible, which is the dangerous part.

    So a Part here is a contiguous EPOCH range. Every Part trains on
    100% of the dataset; Part N simply continues where Part N-1 stopped.
    The finished model is identical to what an uninterrupted run would
    have produced (same data, same schedule, same total epochs).

HOW RESUME WORKS (survives a reboot)
    Two independent mechanisms, neither of which is an in-memory counter:

      1. Ultralytics writes `last.pt` after EVERY epoch, and that
         checkpoint stores the epoch number, optimizer state, LR schedule
         and EMA. `resume=True` continues from it exactly.
      2. This module writes `parts_state.json` next to the weights after
         every Part, recording which Parts are complete and the epoch each
         one ended on. That file is the operator-facing ledger and is what
         the ordering/duplicate checks read.

    On restart the two are cross-checked against each other. If they
    disagree (e.g. a Part was killed mid-epoch), `last.pt` wins, because
    it is the thing that actually holds the weights, and the discrepancy
    is reported rather than silently reconciled.

SAFETY RULES
    - A Part refuses to run if the previous Part is not complete: no gaps.
    - A completed Part refuses to re-run without --force: no wasted hours,
      no accidental overwrite of good weights.
    - Nothing here touches the dataset. No file is renamed, moved,
      relabelled or rewritten -- the dataset is read-only input to
      training, so the "don't break the dataset structure" requirement is
      satisfied by construction.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from configs import registry as R


@dataclass
class PartPlan:
    part: int
    total_parts: int
    start_epoch: int          # 1-based, inclusive
    end_epoch: int            # 1-based, inclusive
    total_epochs: int

    @property
    def n_epochs(self) -> int:
        return self.end_epoch - self.start_epoch + 1


def plan_parts(total_epochs: int, total_parts: int) -> list[PartPlan]:
    """
    Splits `total_epochs` into `total_parts` contiguous ranges.

    Guarantees, which are asserted by scripts/verify_parts.py:
      - every epoch belongs to exactly one Part (no overlap, no gaps);
      - the union of all Parts is exactly 1..total_epochs;
      - sizes differ by at most 1 (the remainder is spread over the
        earliest Parts rather than dumped on the last one).
    """
    if total_parts < 1 or total_parts > total_epochs:
        raise ValueError(
            f"total_parts must be in 1..{total_epochs} (got {total_parts})")
    base, extra = divmod(total_epochs, total_parts)
    plans, cursor = [], 1
    for i in range(total_parts):
        n = base + (1 if i < extra else 0)
        plans.append(PartPlan(i + 1, total_parts, cursor, cursor + n - 1, total_epochs))
        cursor += n
    return plans


# ----------------------------------------------------------------------
# Persisted state
# ----------------------------------------------------------------------

class PartsState:
    """The on-disk ledger of completed Parts (`parts_state.json`)."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {"model": None, "total_epochs": None,
                           "total_parts": None, "parts": {}}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # A truncated ledger (power loss mid-write) must not brick
                # the run: last.pt is the real source of truth, so start a
                # fresh ledger and let the cross-check report the gap.
                backup = path.with_suffix(".corrupt.json")
                path.replace(backup)
                print(f"[parts] WARNING: unreadable state file, moved to {backup}")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        tmp.replace(self.path)          # atomic: never a half-written ledger

    def completed(self) -> set[int]:
        return {int(k) for k, v in self.data.get("parts", {}).items()
                if v.get("status") == "completed"}

    def record(self, plan: PartPlan, *, status: str, started: str,
               finished: str, epochs_run: int, note: str = "") -> None:
        self.data["parts"][str(plan.part)] = {
            "status": status,
            "start_epoch": plan.start_epoch,
            "end_epoch": plan.end_epoch,
            "epochs_run": epochs_run,
            "started_at": started,
            "finished_at": finished,
            "note": note,
        }
        self.save()


class ChunkComplete(Exception):
    """Raised from the epoch callback to end a Part without letting
    ultralytics treat the run as FINISHED. See run_part() for why."""

    def __init__(self, epoch: int):
        super().__init__(f"chunk complete at epoch {epoch}")
        self.epoch = epoch


def checkpoint_state(last_pt: Path) -> tuple[int | None, bool]:
    """
    Reads `last.pt` and returns (epochs_completed, is_finalized).

    This is the reboot-proof source of truth -- it reads the weights file
    itself, not any counter this program kept.

    `is_finalized` is True when ultralytics has stripped the optimizer from
    the checkpoint, which it does in final_eval() once training reaches its
    natural end. A stripped checkpoint stores epoch = -1 and has no
    optimizer state, so it CANNOT be resumed -- distinguishing that from
    "0 epochs done" matters, because the two look identical if you only
    read the epoch field.
    """
    if not last_pt.exists():
        return None, False
    try:
        import torch
        ckpt = torch.load(last_pt, map_location="cpu", weights_only=False)
        epoch = int(ckpt.get("epoch", -1))
        if epoch < 0:
            return None, True          # stripped: training ran to completion
        return epoch + 1, False        # 0-based index -> epochs completed
    except Exception as exc:                                  # noqa: BLE001
        print(f"[parts] could not read epoch from {last_pt}: {exc}")
        return None, False


def checkpoint_epoch(last_pt: Path) -> int | None:
    """Epochs completed, or None if absent//finalized. Thin wrapper kept for
    callers that do not care about the finalized distinction."""
    return checkpoint_state(last_pt)[0]


# ----------------------------------------------------------------------
# Progress reporting
# ----------------------------------------------------------------------

class PartProgress:
    """Prints the per-epoch progress block for the current Part."""

    def __init__(self, plan: PartPlan):
        self.plan = plan
        self.t0 = time.time()
        self.done_this_part = 0

    def update(self, absolute_epoch: int) -> None:
        self.done_this_part = absolute_epoch - self.plan.start_epoch + 1
        elapsed = time.time() - self.t0
        frac_part = self.done_this_part / self.plan.n_epochs
        eta_part = (elapsed / frac_part - elapsed) if frac_part > 0 else 0
        overall = absolute_epoch / self.plan.total_epochs
        print(
            f"\n  Part {self.plan.part}/{self.plan.total_parts}"
            f"  |  epoch {absolute_epoch}/{self.plan.total_epochs}"
            f"  (part {self.done_this_part}/{self.plan.n_epochs})\n"
            f"  Part progress   : {frac_part * 100:5.1f}%"
            f"   Overall: {overall * 100:5.1f}%\n"
            f"  Elapsed (part)  : {timedelta(seconds=int(elapsed))}"
            f"   ETA (part): {timedelta(seconds=int(eta_part))}",
            flush=True,
        )


# ----------------------------------------------------------------------
# Running one Part
# ----------------------------------------------------------------------

def run_part(model_name: str, part: int, total_parts: int, *,
             force: bool = False, overrides: dict | None = None) -> dict:
    """
    Trains exactly ONE Part and returns a summary. Never starts the next
    Part -- that is the operator's decision, by design.
    """
    from ai.computer_vision.train_common import preflight

    spec, data_yaml = preflight(model_name)
    total_epochs = int((overrides or {}).get("epochs", spec.train["epochs"]))
    plans = plan_parts(total_epochs, total_parts)
    if not 1 <= part <= total_parts:
        raise ValueError(f"--part must be 1..{total_parts} (got {part})")
    plan = plans[part - 1]

    run_dir = R.runs_root() / spec.run_name
    state = PartsState(run_dir / "parts_state.json")
    state.data.update({"model": spec.name, "total_epochs": total_epochs,
                       "total_parts": total_parts})
    last_pt = run_dir / "weights" / "last.pt"
    ckpt_epoch, ckpt_finalized = checkpoint_state(last_pt)
    done = state.completed()

    print(f"\n{'=' * 62}\n  {spec.name.upper()}  --  PART {part}/{total_parts}\n{'=' * 62}")
    print(f"  epochs this part : {plan.start_epoch}-{plan.end_epoch} "
          f"({plan.n_epochs} of {total_epochs})")
    print(f"  parts completed  : {sorted(done) or 'none'}")
    if ckpt_finalized:
        print("  last.pt epoch    : FINALIZED (optimizer stripped -- training "
              "already reached its natural end)")
    else:
        print(f"  last.pt epoch    : "
              f"{ckpt_epoch if ckpt_epoch is not None else 'no checkpoint yet'}")

    # ---- ordering / duplicate guards --------------------------------
    if part in done and not force:
        msg = (f"Part {part} is already marked completed "
               f"(epochs {plan.start_epoch}-{plan.end_epoch}). "
               f"Nothing to do. Re-run with --force to train it again.")
        print(f"\n  SKIPPED: {msg}")
        return {"status": "already_completed", "part": part, "message": msg}

    missing = [p for p in range(1, part) if p not in done]
    if missing and not force:
        raise RuntimeError(
            f"Cannot start Part {part}: Part(s) {missing} have not completed. "
            f"Parts are contiguous epoch ranges and must run in order, or the "
            f"skipped epochs would never be trained. Run Part {missing[0]} first."
        )

    # ---- cross-check ledger against the real checkpoint --------------
    expected_start = plan.start_epoch - 1        # epochs completed before this part
    if ckpt_epoch is not None and ckpt_epoch != expected_start and not force:
        if ckpt_epoch > expected_start:
            print(f"\n  NOTE: last.pt holds {ckpt_epoch} epochs but Part {part} "
                  f"expects to start after {expected_start}. Training will "
                  f"continue from the checkpoint ({ckpt_epoch}) -- the weights "
                  f"file is authoritative, not the ledger.")
        else:
            print(f"\n  WARNING: last.pt holds only {ckpt_epoch} epochs but Part "
                  f"{part} expects {expected_start} done. A previous Part likely "
                  f"died mid-epoch; training resumes from {ckpt_epoch}, so no "
                  f"epoch is skipped.")

    started = datetime.now(UTC)
    progress = PartProgress(plan)

    from ultralytics import YOLO

    resume = ckpt_epoch is not None and ckpt_epoch > 0
    args = dict(spec.train)
    args.update(overrides or {})
    args.update({
        "data": str(data_yaml),
        "project": str(R.runs_root()),
        "name": spec.run_name,
        "epochs": total_epochs,     # ALWAYS the full schedule -- the chunk
                                    # stops early via callback, so the LR
                                    # schedule matches an uninterrupted run
        "exist_ok": True,
        "resume": resume,
    })

    model = YOLO(str(last_pt) if resume else spec.base_weights)

    is_final_part = plan.end_epoch >= total_epochs

    def _on_epoch_end(trainer):
        absolute = int(trainer.epoch) + 1
        progress.update(absolute)
        if absolute >= plan.end_epoch and not is_final_part:
            # Abort by RAISING, not by setting trainer.stop.
            #
            # Verified against ultralytics/engine/trainer.py (8.4.98):
            #   565  save_model()                 <- last.pt for THIS epoch
            #   578  run_callbacks(on_fit_epoch_end)   <- we are here
            #   587  if self.stop: break
            #   594  final_eval()
            #   896  strip_optimizer(self.last)
            #
            # Setting trainer.stop breaks the loop but still falls through
            # to final_eval(), which strips the optimizer and writes
            # epoch = -1. That checkpoint CANNOT be resumed, so every Part
            # after the first would silently restart from epoch 1 with a
            # fresh optimizer. Raising here escapes before line 594, so
            # last.pt keeps its optimizer state and epoch number and
            # `resume=True` picks up exactly where this Part stopped.
            #
            # The FINAL Part deliberately does not raise: it runs to its
            # natural end so final_eval() does run and best.pt is properly
            # finalized.
            print(f"\n  >> Part {plan.part} epoch budget reached "
                  f"({plan.end_epoch}). Stopping before final_eval to keep "
                  f"the checkpoint resumable.", flush=True)
            raise ChunkComplete(absolute)

    model.add_callback("on_fit_epoch_end", _on_epoch_end)

    status, note = "completed", ""
    reached_natural_end = False
    try:
        model.train(**args)
        reached_natural_end = True
    except ChunkComplete:
        pass                      # expected: this Part hit its epoch budget
    except Exception as exc:                                  # noqa: BLE001
        status, note = "failed", f"{type(exc).__name__}: {exc}"
        print(f"\n  PART FAILED: {note}")

    finished = datetime.now(UTC)
    final_epoch, finalized = checkpoint_state(last_pt)
    if finalized or (reached_natural_end and is_final_part):
        # Training reached its natural end; the stripped checkpoint no
        # longer carries an epoch number, but we know it completed.
        final_epoch = total_epochs
    epochs_run = (final_epoch or 0) - expected_start

    # A Part only counts as completed if the checkpoint really reached its
    # last epoch -- never on the basis of train() simply returning.
    if status == "completed" and (final_epoch or 0) < plan.end_epoch:
        status = "incomplete"
        note = (f"checkpoint reached epoch {final_epoch}, expected "
                f"{plan.end_epoch}; re-run this Part to finish it")

    state.record(plan, status=status, started=started.isoformat(),
                 finished=finished.isoformat(), epochs_run=max(epochs_run, 0),
                 note=note)

    elapsed = finished - started
    remaining_parts = [p for p in range(1, total_parts + 1)
                       if p not in state.completed()]
    print(f"\n{'=' * 62}\n  PART {part}/{total_parts} SUMMARY\n{'=' * 62}")
    print(f"  status            : {status.upper()}")
    print(f"  epochs this part  : {max(epochs_run, 0)} "
          f"(target {plan.n_epochs}: {plan.start_epoch}-{plan.end_epoch})")
    print(f"  checkpoint epoch  : {final_epoch}/{total_epochs}")
    print(f"  elapsed           : {elapsed}")
    print(f"  parts completed   : {sorted(state.completed())}")
    print(f"  parts remaining   : {remaining_parts or 'NONE -- ready to finalize'}")
    if note:
        print(f"  note              : {note}")
    print(f"  weights           : {last_pt}")

    return {
        "status": status, "part": part, "total_parts": total_parts,
        "epochs_run": max(epochs_run, 0), "checkpoint_epoch": final_epoch,
        "elapsed_s": elapsed.total_seconds(),
        "completed_parts": sorted(state.completed()),
        "remaining_parts": remaining_parts,
        "all_parts_done": not remaining_parts,
        "note": note,
    }


def all_parts_complete(model_name: str, total_parts: int) -> tuple[bool, list[int]]:
    spec = R.get_model(model_name)
    state = PartsState(R.runs_root() / spec.run_name / "parts_state.json")
    done = state.completed()
    missing = [p for p in range(1, total_parts + 1) if p not in done]
    return (not missing), missing


# ----------------------------------------------------------------------
# Finalize (the "merge" step, run once after the last Part)
# ----------------------------------------------------------------------

def finalize(model_name: str, total_parts: int, *, force: bool = False) -> dict:
    """
    Runs ONCE, after every Part has completed. There is nothing to merge in
    the file sense -- each Part continued the same weights file, so the
    trained model already exists. Finalizing therefore means:

      1. verify all Parts completed and the checkpoint reached the full
         epoch count (refuses to proceed otherwise);
      2. verify the dataset is intact and unchanged by training;
      3. evaluate best.pt on the held-out test split;
      4. publish the checkpoint to its registry path and write metrics.json.

    Deliberately never called automatically from run_part() -- the operator
    controls progression. train_player.py calls it after the final Part.
    """
    from ai.computer_vision.train_common import evaluate_model

    spec = R.get_model(model_name)
    run_dir = R.runs_root() / spec.run_name
    state = PartsState(run_dir / "parts_state.json")
    total_epochs = state.data.get("total_epochs") or spec.train["epochs"]
    ok, missing = all_parts_complete(model_name, total_parts)

    print(f"\n{'=' * 62}\n  FINALIZING {spec.name.upper()}\n{'=' * 62}")
    if not ok and not force:
        raise RuntimeError(
            f"Refusing to finalize: Part(s) {missing} are not completed. "
            f"Those epochs were never trained.")

    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    final_epoch, finalized = checkpoint_state(last_pt)
    if finalized:
        # Optimizer stripped => ultralytics ran final_eval() => the full
        # schedule completed. The epoch field is -1 by design, not zero.
        final_epoch = total_epochs
    if final_epoch is not None and final_epoch < total_epochs and not force:
        raise RuntimeError(
            f"Refusing to finalize: checkpoint reached epoch {final_epoch} of "
            f"{total_epochs}. Re-run the unfinished Part.")
    if not best_pt.exists():
        raise FileNotFoundError(f"No trained weights at {best_pt}")

    # ---- dataset integrity: training must not have altered it --------
    report = R.verify_dataset(spec.dataset, strict=False)
    n_images = sum(s.n_images for s in report.splits)
    n_labels = sum(s.n_labels for s in report.splits)
    print(f"  dataset          : {spec.dataset}  ({'OK' if report.ok else 'PROBLEMS'})")
    print(f"  images / labels  : {n_images:,} / {n_labels:,}")
    for s in report.splits:
        print(f"    {s.name:<22} {s.n_images:>7,} images  {s.n_labels:>7,} labels")

    manifest_dir = R.REPO_ROOT / "runs" / "_data" / spec.dataset
    manifest_counts = {}
    for key in ("train", "val", "test"):
        p = manifest_dir / f"{key}.txt"
        if p.exists():
            lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            manifest_counts[key] = {"count": len(lines), "unique": len(set(lines))}

    print("\n  evaluating best.pt on the held-out test split ...")
    # Evaluate the RUN's best.pt explicitly. Calling evaluate_model without
    # a model would go through spec.require_checkpoint(), which points at
    # the published registry path -- and that file is only written a few
    # lines below, so on a first run it does not exist yet and finalize
    # would fail at the last step of the last Part.
    from ultralytics import YOLO
    metrics = evaluate_model(spec, model=YOLO(str(best_pt)), split="test")

    spec.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(best_pt, spec.checkpoint)

    from ai.computer_vision.train_common import _environment
    payload = {
        "model": spec.name, "task": spec.task, "dataset": spec.dataset,
        "trained_in_parts": total_parts,
        "total_epochs": total_epochs, "final_checkpoint_epoch": final_epoch,
        "parts": state.data.get("parts", {}),
        "weights": str(spec.checkpoint),
        "dataset_images": n_images, "dataset_labels": n_labels,
        "split_manifests": manifest_counts,
        "metrics": metrics,
        "finalized_at": datetime.now(UTC).isoformat(),
        "environment": _environment(),
    }
    spec.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    spec.metrics_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    box = metrics.get("box", {})
    dup = {k: v["count"] - v["unique"] for k, v in manifest_counts.items()}
    failed = [p for p, v in state.data.get("parts", {}).items()
              if v.get("status") != "completed"]
    print(f"\n{'=' * 62}\n  FINAL INTEGRITY REPORT\n{'=' * 62}")
    print(f"  total images         : {n_images:,}")
    print(f"  total labels         : {n_labels:,}")
    print(f"  missing label files  : {max(n_images - n_labels, 0):,}")
    print(f"  duplicate entries    : {dup or 'none'}")
    print(f"  failed parts         : {failed or 'none'}")
    print(f"  epochs trained       : {final_epoch}/{total_epochs} across {total_parts} parts")
    if box:
        print(f"  precision / recall   : {box.get('precision', float('nan')):.4f}"
              f" / {box.get('recall', float('nan')):.4f}")
        print(f"  mAP50 / mAP50-95     : {box.get('map50', float('nan')):.4f}"
              f" / {box.get('map50_95', float('nan')):.4f}")
    print(f"  final model          : {spec.checkpoint}")
    print(f"  metrics              : {spec.metrics_file}")
    return payload
