"""Dispatch tests for training/cli.py. No training runs: every entry point
into ultralytics is replaced with a recorder."""

from __future__ import annotations

import pytest

from configs.registry import RegistryError
from training import cli


@pytest.fixture
def calls(monkeypatch):
    log: list[tuple] = []
    monkeypatch.setattr(cli, "train_model", lambda m, resume, overrides: log.append(("train", m, resume, overrides)))
    monkeypatch.setattr(cli, "finalize", lambda m, n, force=False: log.append(("finalize", m, n, force)))
    monkeypatch.setattr(cli, "print_status", lambda m, n: log.append(("status", m, n)))

    def fake_run_part(m, part, total, force, overrides):
        log.append(("part", m, part, total, force, overrides))
        return {"status": "completed", "all_parts_done": part == total, "remaining_parts": [part + 1]}

    monkeypatch.setattr(cli, "run_part", fake_run_part)
    return log


def test_player_keeps_six_default_parts_others_four():
    assert cli.default_total_parts("player") == 6
    for model in ("ball", "calibration", "field", "goalpost"):
        assert cli.default_total_parts(model) == 4


def test_no_part_runs_uninterrupted_training_with_overrides(calls):
    assert cli.main(["--epochs", "3", "--device", "cpu"], model="field") == 0
    assert calls == [("train", "field", False, {"epochs": 3, "device": "cpu"})]


def test_part_uses_model_default_total(calls):
    cli.main(["--part", "2"], model="player")
    assert calls[0][:4] == ("part", "player", 2, 6)


def test_last_part_auto_finalizes(calls):
    cli.main(["--part", "4", "--total-parts", "4"], model="ball")
    assert [c[0] for c in calls] == ["part", "finalize"]


def test_next_part_command_names_the_wrapper_module(calls, capsys):
    """In-progress runs tell the user what to type next; that command must
    still exist, which is why the per-model wrappers were kept."""
    cli.main(["--part", "1"], model="calibration")
    assert "python -m training.train_calibration --part 2 --total-parts 4" in capsys.readouterr().out


def test_generic_entry_point_takes_model_positionally(calls):
    cli.main(["goalpost", "--resume"])
    assert calls == [("train", "goalpost", True, {})]


def test_unknown_model_fails_before_any_training(calls):
    with pytest.raises(RegistryError):
        cli.main(["no-such-model"])
    assert calls == []
