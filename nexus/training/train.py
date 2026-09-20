from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nexus.training.config import (
    TrainingConfig,
    estimate_peak_vram_gb,
    infer_param_count_b,
    total_training_steps,
)
from nexus.training.preflight import run_preflight

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")

# Below this fraction of the early-run average, throughput has dropped far
# enough that it is worth telling the user. On a laptop 5070 Ti the usual
# cause is thermal throttling, not a code problem.
_THROUGHPUT_WARN_RATIO = 0.75

# Steps to average over before drift comparisons mean anything — the first
# few steps include warmup and compilation and are not representative.
#
# Four, not ten. This warning is timed per optimizer step, and real runs
# here are short: the 7B smoke run was 12 steps total. A ten-step baseline
# left at most two steps to compare against and, combined with reading a
# key that only appears in the end-of-run summary, meant the check could
# not fire at all. Four leaves the majority of even a short run inside the
# comparison window.
_BASELINE_WINDOW_STEPS = 4

# How many recent steps are averaged before crying throttle. A single slow
# step is normal — an allocator stall, another process grabbing the GPU
# for a moment — and warning on one would train users to ignore this.
_ROLLING_WINDOW_STEPS = 3

# Measured intervals discarded before any are recorded. The very first
# timed step still carries autotuning and allocator growth, which makes it
# an unrepresentatively slow anchor for everything after it.
_WARMUP_STEPS_SKIPPED = 1

# Rough sustained throughput for a 7B QLoRA step at seq_len 1024 on this
# class of laptop GPU, used only to put a wall-time figure on the plan.
_BASELINE_SAMPLES_PER_SEC_7B = 0.45


@dataclass
class ThroughputSample:
    step: int
    samples_per_second: float
    timestamp: float


@dataclass
class ThroughputMonitor:
    """Tracks samples/sec across a run and flags sustained drops.

    Lives at module scope with no ML imports so it stays unit-testable
    without a GPU; train() wraps it in a transformers callback rather than
    implementing the logic inside one.
    """

    warn_ratio: float = _THROUGHPUT_WARN_RATIO
    baseline_window: int = _BASELINE_WINDOW_STEPS
    rolling_window: int = _ROLLING_WINDOW_STEPS
    samples: list[ThroughputSample] = field(default_factory=list)
    _warned: bool = False

    def record(self, step: int, samples_per_second: float) -> str | None:
        self.samples.append(
            ThroughputSample(step=step, samples_per_second=samples_per_second, timestamp=time.time())
        )
        return self.check_drift()

    @property
    def baseline(self) -> float | None:
        early = [s.samples_per_second for s in self.samples[: self.baseline_window]]
        if len(early) < self.baseline_window:
            return None
        return sum(early) / len(early)

    @property
    def recent(self) -> list[ThroughputSample]:
        """The steps compared against the baseline: the most recent ones
        that came AFTER the baseline was established. Baseline samples are
        excluded deliberately — leaving them in would drag the recent
        average back toward the very number it is being compared to, and
        blunt exactly the sustained drop this is looking for."""
        return self.samples[self.baseline_window :][-self.rolling_window :]

    def check_drift(self) -> str | None:
        """Returns a warning string the first time throughput falls
        sustainedly below the early-run average, then stays quiet — a
        throttling laptop would otherwise emit the same warning every
        step for hours.

        "Sustained" means the mean of up to `rolling_window` recent steps,
        not a single reading: once a run is long enough to fill the
        window, one stalled step no longer trips the warning on its own.
        """
        baseline = self.baseline
        if baseline is None or baseline <= 0 or self._warned:
            return None

        recent = self.recent
        if not recent:
            return None

        window_rate = sum(s.samples_per_second for s in recent) / len(recent)
        ratio = window_rate / baseline
        if ratio >= self.warn_ratio:
            return None

        self._warned = True
        return (
            f"THROUGHPUT DROP at step {recent[-1].step}: {window_rate:.3f} "
            f"samples/sec over the last {len(recent)} step(s) is {ratio:.0%} of the "
            f"early-run average ({baseline:.3f}). "
            f"On this laptop GPU that is the thermal-throttling signature — the run is "
            f"still correct, just slower. Checkpoints every {{save_steps}} steps mean you "
            f"can stop and resume later if you would rather train cool."
        )


@dataclass
class StepThroughputTracker:
    """Turns step start/end timestamps into samples/sec and feeds them to
    a ThroughputMonitor.

    This exists because the previous warning read
    `train_samples_per_second` out of the log dict, and HF emits that key
    ONLY in the single end-of-run summary — never per step. The monitor
    therefore saw at most one sample per run, never reached a baseline,
    and could not fire under any circumstances. Timing the steps here is
    the only way this check measures anything at all.

    Timestamps are passed IN rather than read from the clock, so the whole
    thing is drivable from a test with synthetic timings — no GPU, no
    transformers, no sleeping.
    """

    samples_per_step: int
    monitor: ThroughputMonitor = field(default_factory=ThroughputMonitor)
    warmup_steps_skipped: int = _WARMUP_STEPS_SKIPPED
    _step_started_at: float | None = None
    _timed_steps: int = 0

    def on_step_begin(self, timestamp: float) -> None:
        self._step_started_at = timestamp

    def on_step_end(self, step: int, timestamp: float) -> str | None:
        """Measures begin->end, NOT end->end. Checkpoint saving, logging
        and evaluation all happen after on_step_end and before the next
        on_step_begin, so an end-to-end interval would charge a save's
        several seconds to the following step and report a throughput
        collapse that is really just a checkpoint being written."""
        started_at, self._step_started_at = self._step_started_at, None
        if started_at is None:
            return None

        elapsed = timestamp - started_at
        if elapsed <= 0:
            return None

        self._timed_steps += 1
        if self._timed_steps <= self.warmup_steps_skipped:
            return None

        return self.monitor.record(step, self.samples_per_step / elapsed)


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Highest-numbered checkpoint-N directory, which is what --resume
    picks up. Sorted numerically rather than lexically so checkpoint-900
    does not outrank checkpoint-1000."""
    root = Path(output_dir)
    if not root.exists():
        return None
    checkpoints: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = _CHECKPOINT_RE.match(child.name)
        if match:
            checkpoints.append((int(match.group(1)), child))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


class StratificationError(ValueError):
    """Raised when a dataset cannot be split with every task_type present
    on both sides. Loud by design: a validation set missing a task_type
    reports an eval loss that says nothing about that behaviour, and the
    run would look healthy while being blind to it."""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stratified_split(
    rows: list[dict[str, Any]], *, val_split: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Splits rows into (train, val), holding out `val_split` of EACH
    task_type rather than of the dataset as a whole.

    Stratified because the split is small and the types are few: a plain
    random 10% of ~60 rows can easily contain no `health_safe` example at
    all, and the eval loss would then be silent about the one behaviour
    that matters most. Holding out a share of every type makes the
    validation loss mean the same thing for each.

    Deterministic under `seed` — the same seed gives the same split, so a
    resumed or repeated run is comparable to the original. Groups are
    sorted by name before shuffling so the result does not depend on the
    order rows happened to appear in the file.
    """
    if not 0.0 < val_split < 1.0:
        raise StratificationError(f"val_split must be strictly between 0 and 1, got {val_split}")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("task_type", "unknown")), []).append(row)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    too_small: list[str] = []

    for task_type in sorted(grouped):
        # Sorted by CONTENT before shuffling, not left in file order, so
        # the split depends only on the rows themselves. Regenerating the
        # dataset in a different order then still yields the same split
        # for the same seed, which is what makes two runs comparable.
        group = sorted(grouped[task_type], key=lambda r: json.dumps(r, sort_keys=True))
        random.Random(f"{seed}:{task_type}").shuffle(group)

        # At least one of every type on each side, which is the whole
        # point of stratifying — so a type with fewer than 2 examples
        # cannot be split at all and is reported rather than silently
        # dropped from one side.
        held_out = max(1, int(len(group) * val_split))
        if held_out >= len(group):
            too_small.append(f"{task_type} (n={len(group)})")
            continue

        val.extend(group[:held_out])
        train.extend(group[held_out:])

    if too_small:
        raise StratificationError(
            f"task_type(s) with too few examples to hold any out while keeping at least "
            f"one in training: {', '.join(too_small)}. Either add examples of these types "
            f"or lower val_split; a split that drops a type from one side makes the eval "
            f"loss unreadable for that behaviour."
        )

    train_types = {str(r.get("task_type", "unknown")) for r in train}
    val_types = {str(r.get("task_type", "unknown")) for r in val}
    if train_types != val_types:
        raise StratificationError(
            f"stratification failed: train has {sorted(train_types - val_types)} with no "
            f"validation examples, validation has {sorted(val_types - train_types)} with no "
            f"training examples. Every task_type must appear on both sides."
        )

    return train, val


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target


def count_examples(dataset_path: str | Path) -> int:
    path = Path(dataset_path)
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def build_training_plan(config: TrainingConfig, *, dataset_path: str | Path) -> dict[str, Any]:
    """Everything the dry run reports. Pure arithmetic over the config —
    imports nothing from the ML stack, so this is the part of train.py that
    works on a machine with no CUDA toolchain at all."""
    example_count = count_examples(dataset_path)
    param_count_b = infer_param_count_b(config.base_model)

    # Steps are counted against the TRAIN half, not the whole file — the
    # validation rows are never trained on, so including them would
    # over-report the step count and the wall-time estimate with it. Doing
    # the real split here also means an un-stratifiable dataset fails in
    # the dry run, before a multi-hour job starts, rather than at minute
    # one of the real one.
    train_example_count = example_count
    val_example_count = 0
    if example_count:
        train_rows, val_rows = stratified_split(
            read_jsonl(dataset_path), val_split=config.val_split, seed=config.seed
        )
        train_example_count = len(train_rows)
        val_example_count = len(val_rows)

    steps = total_training_steps(config, example_count=max(1, train_example_count))
    effective_batch = config.batch_size * config.gradient_accumulation_steps

    samples_per_second = _BASELINE_SAMPLES_PER_SEC_7B * (7.0 / max(1.0, param_count_b))
    estimated_seconds = (steps * effective_batch) / max(1e-6, samples_per_second)

    return {
        "base_model": config.base_model,
        "method": config.method,
        "dataset_path": str(dataset_path),
        "example_count": example_count,
        "train_example_count": train_example_count,
        "val_example_count": val_example_count,
        "val_split": config.val_split,
        "param_count_b": param_count_b,
        "epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "effective_batch_size": effective_batch,
        "max_seq_length": config.max_seq_length,
        "total_steps": steps,
        "save_steps": config.save_steps,
        "checkpoint_count": max(1, steps // max(1, config.save_steps)),
        "estimated_peak_vram_gb": estimate_peak_vram_gb(config, param_count_b=param_count_b),
        "available_vram_gb": config.available_vram_gb,
        "estimated_wall_time_hours": round(estimated_seconds / 3600.0, 2),
        "dataloader_num_workers": config.dataloader_num_workers,
        "output_dir": config.output_dir,
    }


def render_plan(plan: dict[str, Any]) -> str:
    fits = plan["estimated_peak_vram_gb"] <= plan["available_vram_gb"]
    lines = [
        "TRAINING PLAN (dry run — nothing was loaded, nothing was trained)",
        "",
        f"  Base model            {plan['base_model']} (~{plan['param_count_b']}B params)",
        f"  Method                {plan['method']}",
        f"  Dataset               {plan['dataset_path']} ({plan['example_count']} examples)",
        f"  Split                 {plan['train_example_count']} train / "
        f"{plan['val_example_count']} validation "
        f"(stratified by task_type, val_split={plan['val_split']})",
        "",
        f"  Epochs                {plan['epochs']}",
        f"  Batch size            {plan['batch_size']} x {plan['gradient_accumulation_steps']} "
        f"accumulation = {plan['effective_batch_size']} effective",
        f"  Max seq length        {plan['max_seq_length']}",
        f"  Total steps           {plan['total_steps']}",
        f"  Checkpoints           every {plan['save_steps']} steps "
        f"({plan['checkpoint_count']} total) -> {plan['output_dir']}",
        f"  Dataloader workers    {plan['dataloader_num_workers']} (capped for Windows spawn)",
        "",
        f"  Estimated peak VRAM   {plan['estimated_peak_vram_gb']:.2f}GB of "
        f"{plan['available_vram_gb']:.2f}GB available "
        f"({'fits' if fits else 'DOES NOT FIT'})",
        f"  Estimated wall time   ~{plan['estimated_wall_time_hours']}h at steady state "
        f"(a throttling laptop will exceed this)",
    ]
    if not fits:
        lines.extend(
            [
                "",
                "  This config is over budget. Remediation ladder, cheapest first:",
                "    1. max_seq_length 1024 -> 512",
                "    2. lora_r 16 -> 8",
                "    3. a 3B base model instead of a 7B",
                "    4. a rented cloud GPU",
            ]
        )
    return "\n".join(lines)


def source_kwargs(base_model: str) -> dict[str, Any]:
    """Extra from_pretrained kwargs for a base model that may be a local path.

    A base_model that names an existing directory is pinned with
    local_files_only=True. WHY: this machine's network is unreliable, and a
    Hub call that hangs is indistinguishable from a crashed run — so the
    weights we already have on disk must never put the network on the
    critical path. A Hub id is left alone so it can still resolve normally.
    """
    if not base_model:
        return {}
    try:
        is_local_dir = Path(base_model).expanduser().is_dir()
    except OSError:
        # A Hub id like "org/model" is a legal string but not a legal path
        # on Windows; that is a "not a local dir" answer, not an error.
        return {}
    return {"local_files_only": True} if is_local_dir else {}


def train(config: TrainingConfig, *, dataset_path: str | Path, resume: bool = False) -> str:
    """Runs the actual fine-tune. Every ML import lives INSIDE this
    function, without exception — importing this module on a machine with
    no CUDA toolchain must keep working, because the config, plan, and
    preflight logic above it are used exactly there."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        EarlyStoppingCallback,
        TrainerCallback,
    )
    from trl import SFTConfig, SFTTrainer

    from datasets import load_dataset

    torch.manual_seed(config.seed)

    tracker = StepThroughputTracker(
        # One on_step_end is one OPTIMIZER step, so a step consumes a full
        # effective batch. The trailing partial batch of an epoch consumes
        # fewer and therefore reads slightly fast, which can only ever
        # suppress this warning, never manufacture one.
        samples_per_step=config.batch_size * config.gradient_accumulation_steps,
    )

    class _ThroughputCallback(TrainerCallback):
        """Thin adapter — all the timing and drift logic lives in
        StepThroughputTracker/ThroughputMonitor at module scope, so both
        can be tested without transformers or a GPU.

        on_step_begin/on_step_end, NOT on_log: HF logs
        train_samples_per_second only in the end-of-run summary, which is
        why the previous version of this callback could never fire."""

        def on_step_begin(self, args, state, control, **kwargs):  # noqa: ANN001
            tracker.on_step_begin(time.monotonic())

        def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
            warning = tracker.on_step_end(state.global_step, time.monotonic())
            if warning:
                print(warning.replace("{save_steps}", str(config.save_steps)), flush=True)

    quantization_config = None
    if config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        )

    # Both from_pretrained calls take the same local/Hub resolution decision.
    from_pretrained_kwargs = source_kwargs(config.base_model)
    if from_pretrained_kwargs:
        print(f"Loading base model from local directory {config.base_model}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, **from_pretrained_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=quantization_config,
        device_map={"": 0},
        # transformers 5 dropped the `torch_dtype` alias for this argument.
        dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        **from_pretrained_kwargs,
    )
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # Split BEFORE the trainer sees anything, in plain Python, and write
    # both halves to disk. The split is then inspectable after the fact —
    # which rows were held out is part of interpreting the eval loss, not
    # an implementation detail to keep in memory.
    train_rows, val_rows = stratified_split(
        read_jsonl(dataset_path), val_split=config.val_split, seed=config.seed
    )
    split_dir = Path(config.output_dir) / "split"
    train_path = write_jsonl(train_rows, split_dir / "train.jsonl")
    val_path = write_jsonl(val_rows, split_dir / "val.jsonl")
    print(
        f"Stratified split (seed={config.seed}): {len(train_rows)} train / "
        f"{len(val_rows)} validation, written to {split_dir}",
        flush=True,
    )

    dataset = load_dataset("json", data_files=str(train_path), split="train")
    eval_dataset = load_dataset("json", data_files=str(val_path), split="train")

    # SFTConfig rather than a bare TrainingArguments: SFTTrainer would
    # silently coerce one, but only SFTConfig carries max_length, and without
    # it max_seq_length reaches the VRAM estimate and nothing else -- which
    # would make the "max_seq_length 1024 -> 512" OOM remedy printed above a
    # no-op, since the trainer would keep tokenising at the library default.
    training_arguments = SFTConfig(
        output_dir=config.output_dir,
        max_length=config.max_seq_length,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        optim=config.optim,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        logging_steps=10,
        # Held-out loss on the same cadence checkpoints are written, so
        # the best checkpoint is one that actually exists on disk.
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=config.bnb_4bit_compute_dtype == "bfloat16",
        dataloader_num_workers=config.dataloader_num_workers,
        seed=config.seed,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=training_arguments,
        processing_class=tokenizer,
        callbacks=[
            _ThroughputCallback(),
            EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience),
        ],
    )

    resume_target: str | bool | None = config.resume_from_checkpoint
    if resume and resume_target is None:
        latest = find_latest_checkpoint(config.output_dir)
        resume_target = str(latest) if latest else None
        if latest:
            print(f"Resuming from {latest}", flush=True)

    trainer.train(resume_from_checkpoint=resume_target)

    adapter_dir = str(Path(config.output_dir) / "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Adapter written to {adapter_dir}", flush=True)
    return adapter_dir


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a custom NEXUS model.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the training JSONL")
    parser.add_argument("--config", type=str, default=None, help="Path to a TrainingConfig YAML")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preflight and print the plan without importing or loading anything",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the newest checkpoint in output_dir"
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable plan output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = TrainingConfig.from_yaml(args.config) if args.config else TrainingConfig()

    # A dry run must import nothing from the ML stack, so it gets the
    # plan-and-data checks only; a real run probes the GPU first.
    result = run_preflight(config, dataset_path=args.dataset, probe_gpu=not args.dry_run)
    try:
        plan = build_training_plan(config, dataset_path=args.dataset)
    except StratificationError as exc:
        # Loud, but not a traceback: an unsplittable dataset is a data
        # problem the user can fix, and it must stop the run either way.
        print(f"Refusing to start: the dataset cannot be split for validation.\n  {exc}")
        return 1

    if args.json:
        print(json.dumps({"plan": plan, "preflight_ok": result.ok}, indent=2))
    else:
        print(render_plan(plan))
        print("")
        print("PREFLIGHT")
        print(result.render())

    if args.dry_run:
        return 0 if result.ok else 1

    if not result.ok:
        print("\nRefusing to start: preflight failed. Fix the items above first.")
        return 1

    train(config, dataset_path=args.dataset, resume=args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
