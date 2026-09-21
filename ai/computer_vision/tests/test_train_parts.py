"""Phased ("Parts") training: the plan, the ledger, and checkpoint reading.

plan_parts() documents guarantees that were previously only checked by the
hand-run scripts/verify_parts.py. A long training run is resumed from these
pieces after every reboot, so they are pinned here. Nothing in this file
trains a model.
"""

from __future__ import annotations

import json

import pytest
import torch

from ai.computer_vision import train_parts
from ai.computer_vision.train_parts import (
    PartPlan,
    PartProgress,
    PartsState,
    checkpoint_epoch,
    checkpoint_state,
    plan_parts,
)

# --- plan_parts ------------------------------------------------------------------

@pytest.mark.parametrize("total_epochs", [1, 7, 50, 100, 150, 301])
@pytest.mark.parametrize("total_parts", [1, 2, 3, 4, 6, 7])
def test_parts_cover_every_epoch_exactly_once(total_epochs, total_parts):
    if total_parts > total_epochs:
        pytest.skip("more parts than epochs is rejected; tested separately")
    plans = plan_parts(total_epochs, total_parts)
    epochs = [e for p in plans for e in range(p.start_epoch, p.end_epoch + 1)]
    assert epochs == list(range(1, total_epochs + 1)), "gap or overlap between parts"
    assert [p.part for p in plans] == list(range(1, total_parts + 1))
    assert all(p.total_parts == total_parts and p.total_epochs == total_epochs for p in plans)


@pytest.mark.parametrize("total_epochs, total_parts", [(100, 6), (150, 4), (101, 4), (10, 3)])
def test_part_sizes_differ_by_at_most_one_with_the_remainder_first(total_epochs, total_parts):
    sizes = [p.n_epochs for p in plan_parts(total_epochs, total_parts)]
    assert max(sizes) - min(sizes) <= 1
    assert sizes == sorted(sizes, reverse=True), "remainder goes to the earliest parts"


def test_the_player_run_in_progress_keeps_its_boundaries():
    """The live `player` run (100 epochs, 6 parts) resumes against exactly
    these ranges; changing plan_parts would silently re-split it."""
    ranges = [(p.start_epoch, p.end_epoch) for p in plan_parts(100, 6)]
    assert ranges == [(1, 17), (18, 34), (35, 51), (52, 68), (69, 84), (85, 100)]


@pytest.mark.parametrize("total_epochs, total_parts", [(10, 0), (10, 11), (5, -1)])
def test_invalid_part_counts_are_rejected(total_epochs, total_parts):
    with pytest.raises(ValueError):
        plan_parts(total_epochs, total_parts)


# --- PartsState ledger ----------------------------------------------------------------

def _plan(part=1, start=1, end=10):
    return PartPlan(part, 4, start, end, 40)


def test_new_ledger_starts_empty(tmp_path):
    state = PartsState(tmp_path / "parts_state.json")
    assert state.completed() == set()


def test_recorded_parts_survive_a_reload(tmp_path):
    path = tmp_path / "run" / "parts_state.json"
    state = PartsState(path)
    state.record(_plan(1, 1, 10), status="completed", started="t0", finished="t1", epochs_run=10)
    state.record(_plan(2, 11, 20), status="interrupted", started="t2", finished="t3", epochs_run=4)

    reloaded = PartsState(path)
    assert reloaded.completed() == {1}, "only completed parts count"
    assert reloaded.data["parts"]["2"]["epochs_run"] == 4


def test_save_leaves_no_temporary_file_behind(tmp_path):
    path = tmp_path / "parts_state.json"
    PartsState(path).record(_plan(), status="completed", started="a", finished="b", epochs_run=10)
    assert path.exists()
    assert not path.with_suffix(".tmp").exists()


def test_a_corrupt_ledger_is_set_aside_not_fatal(tmp_path):
    """Power loss mid-write must not brick the run: the file is moved to
    .corrupt.json and a fresh ledger starts (last.pt stays the truth)."""
    path = tmp_path / "parts_state.json"
    path.write_text('{"parts": {"1": {"status": "compl', encoding="utf-8")
    state = PartsState(path)
    assert state.completed() == set()
    assert path.with_suffix(".corrupt.json").exists()
    assert not path.exists()


# --- checkpoint reading -----------------------------------------------------------------

def test_missing_checkpoint_means_not_started(tmp_path):
    assert checkpoint_state(tmp_path / "last.pt") == (None, False)
    assert checkpoint_epoch(tmp_path / "last.pt") is None


def test_checkpoint_epoch_is_converted_to_epochs_completed(tmp_path):
    path = tmp_path / "last.pt"
    torch.save({"epoch": 67, "optimizer": {}}, path)   # 0-based index 67 = 68 epochs done
    assert checkpoint_state(path) == (68, False)
    assert checkpoint_epoch(path) == 68


def test_a_stripped_checkpoint_is_finalized_not_zero_epochs(tmp_path):
    """ultralytics sets epoch=-1 when training ends naturally; that must not be
    mistaken for "0 epochs done", which would restart training."""
    path = tmp_path / "last.pt"
    torch.save({"epoch": -1}, path)
    assert checkpoint_state(path) == (None, True)


def test_an_unreadable_checkpoint_is_reported_not_raised(tmp_path, capsys):
    path = tmp_path / "last.pt"
    path.write_bytes(b"not a torch file")
    assert checkpoint_state(path) == (None, False)
    assert "could not read epoch" in capsys.readouterr().out


# --- progress + completeness ------------------------------------------------------------

def test_progress_reports_part_and_overall_position(capsys):
    progress = PartProgress(PartPlan(2, 4, 11, 20, 40))
    progress.update(15)
    out = capsys.readouterr().out
    assert progress.done_this_part == 5
    assert "Part 2/4" in out and "epoch 15/40" in out and "part 5/10" in out
    assert "50.0%" in out   # halfway through this part


def test_all_parts_complete_lists_what_is_missing(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs"
    monkeypatch.setattr(train_parts.R, "runs_root", lambda: run_dir)
    run_name = train_parts.R.get_model("ball").run_name
    state = PartsState(run_dir / run_name / "parts_state.json")
    for part in (1, 3):
        state.record(_plan(part), status="completed", started="a", finished="b", epochs_run=10)

    done, missing = train_parts.all_parts_complete("ball", 4)
    assert done is False
    assert missing == [2, 4]

    for part in (2, 4):
        state.record(_plan(part), status="completed", started="a", finished="b", epochs_run=10)
    assert train_parts.all_parts_complete("ball", 4) == (True, [])


def test_ledger_file_is_plain_json(tmp_path):
    """Humans read and occasionally repair this file; keep it plain JSON."""
    path = tmp_path / "parts_state.json"
    PartsState(path).record(_plan(), status="completed", started="a", finished="b", epochs_run=10)
    assert json.loads(path.read_text(encoding="utf-8"))["parts"]["1"]["status"] == "completed"

# --- pausing inside a Part (--stop-at) -------------------------------------------
#
# Splitting a Part must not change the training, only where it pauses. The
# epoch budget passed to ultralytics is always the FULL schedule so the
# learning-rate curve is identical either way; --stop-at only moves the epoch
# at which the callback raises. These pin the guard and the bookkeeping,
# which are the parts that can silently lose epochs.


def _stub_preflight(monkeypatch, tmp_path, total_epochs=100):
    """Enough of a model spec for run_part to reach its --stop-at guard."""
    from ai.computer_vision import train_common

    spec = train_parts.R.get_model("player")
    monkeypatch.setattr(spec.__class__, "train",
                        property(lambda self: {"epochs": total_epochs}), raising=False)
    monkeypatch.setattr(train_common, "preflight",
                        lambda name: (spec, tmp_path / "data.yaml"))
    monkeypatch.setattr(train_parts.R, "runs_root", lambda: tmp_path)
    return spec


@pytest.mark.parametrize("stop_at", [68, 85, 0, -1, 101])
def test_stop_at_outside_the_part_is_rejected(tmp_path, monkeypatch, stop_at):
    """Part 5 is epochs 69-84. A stop point outside it would either re-run
    finished epochs or skip unfinished ones, so it must not be accepted."""
    _stub_preflight(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="stop-at"):
        train_parts.run_part("player", 5, 6, stop_at=stop_at)


@pytest.mark.parametrize("stop_at", [69, 76, 84])
def test_stop_at_inside_the_part_is_accepted(tmp_path, monkeypatch, stop_at):
    """The boundaries themselves are legal: the first epoch of the Part, a
    midpoint, and its last epoch (which is simply the normal behaviour)."""
    _stub_preflight(monkeypatch, tmp_path)
    # Fails later, when it tries to train -- but past the guard, which is
    # what this asserts.
    with pytest.raises(Exception) as exc:
        train_parts.run_part("player", 5, 6, stop_at=stop_at)
    assert "stop-at" not in str(exc.value)


def test_a_paused_part_is_not_recorded_as_completed(tmp_path):
    """The ledger must never mark a Part done while some of its epochs are
    unrun: the next Part would start from the wrong epoch and the gap would
    never be trained."""
    state = PartsState(tmp_path / "parts_state.json")
    plan = plan_parts(100, 6)[4]          # Part 5: epochs 69-84
    state.record(plan, status="paused", started="s", finished="f", epochs_run=8)
    assert 5 not in state.completed()

    state.record(plan, status="completed", started="s", finished="f", epochs_run=16)
    assert 5 in state.completed()
