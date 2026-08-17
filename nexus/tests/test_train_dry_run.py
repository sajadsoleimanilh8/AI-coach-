from __future__ import annotations

import json
import sys

import pytest

from nexus.training.config import TrainingConfig
from nexus.training.train import (
    StepThroughputTracker,
    StratificationError,
    ThroughputMonitor,
    build_training_plan,
    find_latest_checkpoint,
    main,
    render_plan,
    stratified_split,
)

_ML_PACKAGES = ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "trl", "datasets")

_TASK_TYPES = {
    "tool_selection",
    "honest_uncertainty",
    "health_safe",
    "structured_output",
    "sports_interpretation",
}


def _mixed_task_type_rows() -> list[dict]:
    """12 of each of the five real task types — the shape of the actual
    template dataset, which is what the split has to cope with."""
    return [
        {"id": f"{task_type}-{i}", "task_type": task_type, "messages": []}
        for task_type in sorted(_TASK_TYPES)
        for i in range(12)
    ]


def _dataset(tmp_path, count: int = 40) -> str:
    path = tmp_path / "dataset.jsonl"
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"Question {i}?"},
                {"role": "assistant", "content": "Answer."},
            ],
            "task_type": "general",
        }
        for i in range(count)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(path)


def _config_file(tmp_path) -> str:
    path = tmp_path / "config.yaml"
    TrainingConfig(output_dir=str(tmp_path / "output")).to_yaml(path)
    return str(path)


def test_dry_run_imports_no_ml_package(tmp_path, capsys) -> None:
    """The whole point of the lazy-import discipline. Nothing in the ML
    stack may be imported by a dry run, so the plan and preflight stay
    usable on a machine with no CUDA toolchain at all."""
    for package in _ML_PACKAGES:
        assert package not in sys.modules, f"{package} was already imported before the test"

    main(["--dataset", _dataset(tmp_path), "--config", _config_file(tmp_path), "--dry-run"])

    for package in _ML_PACKAGES:
        assert package not in sys.modules, f"--dry-run imported {package}"


def test_dry_run_prints_the_plan(tmp_path, capsys) -> None:
    main(["--dataset", _dataset(tmp_path), "--config", _config_file(tmp_path), "--dry-run"])

    out = capsys.readouterr().out
    assert "TRAINING PLAN" in out
    assert "Estimated peak VRAM" in out
    assert "Total steps" in out
    assert "Checkpoints" in out
    assert "PREFLIGHT" in out


def test_dry_run_succeeds_on_a_valid_plan_and_dataset(tmp_path) -> None:
    exit_code = main(
        ["--dataset", _dataset(tmp_path), "--config", _config_file(tmp_path), "--dry-run"]
    )

    assert exit_code == 0


def test_dry_run_returns_nonzero_on_a_bad_dataset(tmp_path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"messages": [{"role": "user", "content": "x"}]}\n', encoding="utf-8")

    exit_code = main(["--dataset", str(bad), "--config", _config_file(tmp_path), "--dry-run"])

    assert exit_code == 1


def test_dry_run_says_plainly_that_it_did_not_probe_the_gpu(tmp_path, capsys) -> None:
    """A passing dry run must not read as 'this machine can train' — it
    never looked at the GPU."""
    main(["--dataset", _dataset(tmp_path), "--config", _config_file(tmp_path), "--dry-run"])

    out = capsys.readouterr().out
    assert "NOT PROBED" in out
    assert "python -m nexus.training.preflight" in out


def test_json_plan_output_is_machine_readable(tmp_path, capsys) -> None:
    main(
        [
            "--dataset", _dataset(tmp_path), "--config", _config_file(tmp_path),
            "--dry-run", "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["plan"]["example_count"] == 40
    assert "preflight_ok" in payload


def test_plan_arithmetic(tmp_path) -> None:
    config = TrainingConfig(num_epochs=3, batch_size=1, gradient_accumulation_steps=16)

    plan = build_training_plan(config, dataset_path=_dataset(tmp_path, count=320))

    assert plan["example_count"] == 320
    assert plan["effective_batch_size"] == 16
    assert plan["train_example_count"] == 288
    assert plan["val_example_count"] == 32
    assert plan["total_steps"] == 54
    assert plan["checkpoint_count"] == 54 // config.save_steps or 1


def test_split_is_deterministic_under_a_fixed_seed(tmp_path) -> None:
    rows = _mixed_task_type_rows()

    first = stratified_split(rows, val_split=0.2, seed=42)
    second = stratified_split(rows, val_split=0.2, seed=42)

    assert first == second


def test_a_different_seed_gives_a_different_split(tmp_path) -> None:
    rows = _mixed_task_type_rows()

    _, val_a = stratified_split(rows, val_split=0.2, seed=1)
    _, val_b = stratified_split(rows, val_split=0.2, seed=2)

    assert [r["id"] for r in val_a] != [r["id"] for r in val_b]


def test_split_does_not_depend_on_row_order_in_the_file(tmp_path) -> None:
    rows = _mixed_task_type_rows()
    shuffled = list(reversed(rows))

    train_a, val_a = stratified_split(rows, val_split=0.2, seed=42)
    train_b, val_b = stratified_split(shuffled, val_split=0.2, seed=42)

    assert {r["id"] for r in val_a} == {r["id"] for r in val_b}
    assert {r["id"] for r in train_a} == {r["id"] for r in train_b}


def test_split_is_stratified_across_every_task_type(tmp_path) -> None:
    rows = _mixed_task_type_rows()

    train, val = stratified_split(rows, val_split=0.2, seed=42)

    train_types = {r["task_type"] for r in train}
    val_types = {r["task_type"] for r in val}
    assert train_types == val_types == _TASK_TYPES
    for task_type in _TASK_TYPES:
        assert sum(1 for r in val if r["task_type"] == task_type) >= 1


def test_no_row_is_in_both_halves_and_none_is_lost(tmp_path) -> None:
    rows = _mixed_task_type_rows()

    train, val = stratified_split(rows, val_split=0.2, seed=42)

    train_ids = {r["id"] for r in train}
    val_ids = {r["id"] for r in val}
    assert not (train_ids & val_ids)
    assert train_ids | val_ids == {r["id"] for r in rows}


def test_a_task_type_too_small_to_split_raises_loudly() -> None:
    rows = _mixed_task_type_rows() + [{"id": "lonely", "task_type": "brand_new_type"}]

    with pytest.raises(StratificationError) as excinfo:
        stratified_split(rows, val_split=0.1, seed=42)

    assert "brand_new_type" in str(excinfo.value)


def test_a_nonsense_val_split_raises() -> None:
    rows = _mixed_task_type_rows()

    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(StratificationError):
            stratified_split(rows, val_split=bad, seed=42)


def test_dry_run_reports_an_unsplittable_dataset_without_a_traceback(tmp_path, capsys) -> None:
    path = tmp_path / "unsplittable.jsonl"
    path.write_text(
        json.dumps({"task_type": "only_one_of_these", "messages": []}) + "\n", encoding="utf-8"
    )

    exit_code = main(["--dataset", str(path), "--config", _config_file(tmp_path), "--dry-run"])

    assert exit_code == 1
    assert "cannot be split for validation" in capsys.readouterr().out


def test_plan_flags_an_over_budget_config(tmp_path) -> None:
    config = TrainingConfig(load_in_4bit=False)

    rendered = render_plan(build_training_plan(config, dataset_path=_dataset(tmp_path)))

    assert "DOES NOT FIT" in rendered
    assert "max_seq_length 1024 -> 512" in rendered


def test_find_latest_checkpoint_sorts_numerically(tmp_path) -> None:
    for step in (100, 900, 1000):
        (tmp_path / f"checkpoint-{step}").mkdir()
    (tmp_path / "not-a-checkpoint").mkdir()

    latest = find_latest_checkpoint(tmp_path)

    assert latest is not None and latest.name == "checkpoint-1000"


def test_find_latest_checkpoint_returns_none_on_a_fresh_output_dir(tmp_path) -> None:
    assert find_latest_checkpoint(tmp_path / "nothing-here") is None
    assert find_latest_checkpoint(tmp_path) is None


def test_throughput_monitor_stays_quiet_before_a_baseline_exists() -> None:
    monitor = ThroughputMonitor(baseline_window=10)

    for step in range(5):
        assert monitor.record(step, 1.0) is None


def test_throughput_monitor_stays_quiet_at_steady_speed() -> None:
    monitor = ThroughputMonitor(baseline_window=10)

    for step in range(30):
        assert monitor.record(step, 1.0) is None


def test_throughput_monitor_warns_on_a_sustained_drop() -> None:
    monitor = ThroughputMonitor(baseline_window=10, warn_ratio=0.75)
    for step in range(10):
        monitor.record(step, 1.0)

    warning = monitor.record(10, 0.5)

    assert warning is not None
    assert "THROUGHPUT DROP" in warning
    assert "thermal-throttling" in warning


def test_throughput_monitor_warns_only_once() -> None:
    monitor = ThroughputMonitor(baseline_window=10)
    for step in range(10):
        monitor.record(step, 1.0)

    assert monitor.record(10, 0.4) is not None
    assert monitor.record(11, 0.4) is None
    assert monitor.record(12, 0.3) is None


def test_throughput_monitor_ignores_a_small_dip() -> None:
    monitor = ThroughputMonitor(baseline_window=10, warn_ratio=0.75)
    for step in range(10):
        monitor.record(step, 1.0)

    assert monitor.record(10, 0.9) is None


def test_throughput_monitor_needs_a_sustained_drop_not_one_slow_step() -> None:
    monitor = ThroughputMonitor(baseline_window=4, rolling_window=3, warn_ratio=0.75)
    for step in range(4):
        monitor.record(step, 1.0)
    for step in range(4, 8):
        monitor.record(step, 1.0)

    assert monitor.record(8, 0.3) is None

    assert monitor.record(9, 0.3) is not None


def _drive_steps(
    tracker: StepThroughputTracker, *, count: int, seconds_per_step: float, clock: list[float]
) -> list[str]:
    """Runs `count` steps through the tracker the way the TrainerCallback
    does — on_step_begin, work, on_step_end — off a synthetic clock, so no
    GPU, no transformers and no real waiting are involved."""
    warnings: list[str] = []
    start_step = len(tracker.monitor.samples)
    for offset in range(count):
        tracker.on_step_begin(clock[0])
        clock[0] += seconds_per_step
        warning = tracker.on_step_end(start_step + offset, clock[0])
        if warning:
            warnings.append(warning)
        clock[0] += 0.01
    return warnings


def test_step_timing_tracker_fires_on_a_sustained_slowdown() -> None:
    clock = [1000.0]
    tracker = StepThroughputTracker(
        samples_per_step=16,
        monitor=ThroughputMonitor(baseline_window=4, rolling_window=3, warn_ratio=0.75),
        warmup_steps_skipped=1,
    )

    fast = _drive_steps(tracker, count=5, seconds_per_step=1.0, clock=clock)
    assert fast == []

    slow = _drive_steps(tracker, count=4, seconds_per_step=4.0, clock=clock)

    assert len(slow) == 1, "a sustained 4x slowdown must warn exactly once"
    assert "THROUGHPUT DROP" in slow[0]
    assert "thermal-throttling" in slow[0]
    assert "{save_steps}" in slow[0]


def test_step_timing_tracker_stays_quiet_at_a_steady_pace() -> None:
    clock = [1000.0]
    tracker = StepThroughputTracker(
        samples_per_step=16, monitor=ThroughputMonitor(baseline_window=4, rolling_window=3)
    )

    assert _drive_steps(tracker, count=30, seconds_per_step=1.0, clock=clock) == []


def test_step_timing_tracker_discards_the_warmup_step() -> None:
    clock = [1000.0]
    tracker = StepThroughputTracker(
        samples_per_step=16,
        monitor=ThroughputMonitor(baseline_window=2, rolling_window=2),
        warmup_steps_skipped=1,
    )

    _drive_steps(tracker, count=1, seconds_per_step=10.0, clock=clock)
    assert tracker.monitor.samples == []

    _drive_steps(tracker, count=2, seconds_per_step=1.0, clock=clock)
    assert [s.samples_per_second for s in tracker.monitor.samples] == [16.0, 16.0]


def test_step_timing_tracker_measures_the_step_not_the_gap_between_steps() -> None:
    clock = [1000.0]
    tracker = StepThroughputTracker(
        samples_per_step=16,
        monitor=ThroughputMonitor(baseline_window=2, rolling_window=2, warn_ratio=0.75),
        warmup_steps_skipped=0,
    )

    for step in range(4):
        tracker.on_step_begin(clock[0])
        clock[0] += 1.0
        warning = tracker.on_step_end(step, clock[0])
        clock[0] += 30.0
        assert warning is None

    assert [s.samples_per_second for s in tracker.monitor.samples] == [16.0] * 4


def test_step_timing_tracker_ignores_an_end_without_a_begin() -> None:
    tracker = StepThroughputTracker(samples_per_step=16)

    assert tracker.on_step_end(0, 1000.0) is None
    assert tracker.monitor.samples == []
