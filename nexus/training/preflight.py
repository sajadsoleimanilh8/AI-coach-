from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from nexus.training.config import (
    TrainingConfig,
    estimate_peak_vram_gb,
    infer_param_count_b,
    total_training_steps,
)
from nexus.training.validation import validate_dataset

# The RTX 5070 Ti Laptop GPU is Blackwell, compute capability 12.0. PyTorch
# names that arch "sm_120" in torch.cuda.get_arch_list().
BLACKWELL_CAPABILITY = (12, 0)

# The first bitsandbytes release carrying Blackwell kernels. Anything older
# imports cleanly and then dies at the first quantized forward pass with an
# opaque CUDA error, hours into a run — which is why this is checked here.
MIN_BITSANDBYTES_VERSION = (0, 45)

_MIN_TORCH_VERSION = (2, 7)

# Under 20% headroom over the estimate, Windows is likely to start paging
# GPU memory to host RAM rather than raising a clean OOM.
_VRAM_HEADROOM_FACTOR = 1.2

_BYTES_PER_GB = 1024**3

# A rank-16 adapter on a 7B model serializes to roughly 70MB; this scales
# that with both rank and model size.
_ADAPTER_GB_PER_B_PER_RANK = 6.0e-4

_TORCH_INSTALL_COMMAND = (
    "pip install torch --index-url https://download.pytorch.org/whl/cu128"
)


@dataclass
class PreflightResult:
    ok: bool
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def failed(self) -> list[tuple[str, bool, str]]:
        return [check for check in self.checks if not check[1]]

    def render(self) -> str:
        lines = []
        for name, passed, detail in self.checks:
            marker = "PASS" if passed else "FAIL"
            lines.append(f"  [{marker}] {name}: {detail}")
        verdict = "PREFLIGHT OK" if self.ok else "PREFLIGHT FAILED"
        lines.append("")
        lines.append(verdict)
        return "\n".join(lines)


def _parse_version(raw: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in raw.split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _check_torch_cuda() -> tuple[tuple[str, bool, str], object | None]:
    try:
        import torch
    except ImportError:
        return (
            (
                "torch_available",
                False,
                "torch is not installed. Training dependencies are separate from the "
                "API's requirements on purpose. Install them with:\n"
                f"        {_TORCH_INSTALL_COMMAND}\n"
                "        pip install -r nexus/training/requirements-train.txt",
            ),
            None,
        )

    version = getattr(torch, "__version__", "unknown")
    if not torch.cuda.is_available():
        return (
            (
                "torch_available",
                False,
                f"torch {version} is installed but torch.cuda.is_available() is False. "
                "This is almost always a CPU-only wheel. Reinstall from the CUDA 12.8 "
                f"index:\n        {_TORCH_INSTALL_COMMAND}\n"
                "        Training on CPU is not a supported fallback — it would take "
                "weeks, so this fails rather than silently proceeding.",
            ),
            torch,
        )

    parsed = _parse_version(str(version))
    if parsed and parsed < _MIN_TORCH_VERSION:
        return (
            (
                "torch_available",
                False,
                f"torch {version} predates Blackwell support (need >= 2.7). Reinstall:\n"
                f"        {_TORCH_INSTALL_COMMAND}",
            ),
            torch,
        )

    return (("torch_available", True, f"torch {version} with CUDA available."), torch)


def _check_blackwell(torch: object) -> tuple[str, bool, str]:
    """THE most likely failure on this hardware. A torch build without
    sm_120 kernels does not fail cleanly — it either refuses the device or
    silently falls back in ways that read as "training is just slow" — so
    this check prints the full picture (torch version, CUDA version, arch
    list) and the exact remediation."""
    capability = torch.cuda.get_device_capability()  # type: ignore[attr-defined]
    arch_list = list(torch.cuda.get_arch_list())  # type: ignore[attr-defined]
    device_name = torch.cuda.get_device_name(0)  # type: ignore[attr-defined]
    torch_version = getattr(torch, "__version__", "unknown")
    cuda_version = getattr(getattr(torch, "version", None), "cuda", "unknown")

    required_arch = f"sm_{capability[0]}{capability[1]}"

    if required_arch in arch_list:
        return (
            "blackwell_sm120_support",
            True,
            f"{device_name} reports capability {capability[0]}.{capability[1]}; "
            f"{required_arch} is present in this torch build.",
        )

    return (
        "blackwell_sm120_support",
        False,
        f"BLACKWELL KERNEL MISMATCH — this is the failure this hardware hits most often.\n"
        f"        GPU:            {device_name}\n"
        f"        Capability:     {capability[0]}.{capability[1]} (needs {required_arch})\n"
        f"        torch version:  {torch_version}\n"
        f"        CUDA version:   {cuda_version}\n"
        f"        Compiled archs: {', '.join(arch_list) or 'none'}\n"
        f"        {required_arch} is NOT in the compiled arch list, so this torch build has "
        f"no kernels for this GPU.\n"
        f"        A plain `pip install torch` resolves a wheel that predates Blackwell and "
        f"fails confusingly — either the device is refused outright, or work silently lands "
        f"on the CPU and the run just looks slow.\n"
        f"        FIX: uninstall the current build and reinstall from the CUDA 12.8 index:\n"
        f"        pip uninstall -y torch torchvision torchaudio\n"
        f"        {_TORCH_INSTALL_COMMAND}",
    )


def _check_bitsandbytes(*, load_in_4bit: bool) -> tuple[str, bool, str]:
    if not load_in_4bit:
        return (
            "bitsandbytes_blackwell",
            True,
            "load_in_4bit=False, so bitsandbytes quantization kernels are not required.",
        )

    try:
        import bitsandbytes
    except ImportError:
        return (
            "bitsandbytes_blackwell",
            False,
            "bitsandbytes is not installed but method=qlora requires it. Install:\n"
            f"        pip install 'bitsandbytes>={MIN_BITSANDBYTES_VERSION[0]}."
            f"{MIN_BITSANDBYTES_VERSION[1]}'",
        )

    version = getattr(bitsandbytes, "__version__", "0.0")
    parsed = _parse_version(str(version))
    if parsed < MIN_BITSANDBYTES_VERSION:
        return (
            "bitsandbytes_blackwell",
            False,
            f"bitsandbytes {version} predates Blackwell kernel support (need >= "
            f"{MIN_BITSANDBYTES_VERSION[0]}.{MIN_BITSANDBYTES_VERSION[1]}).\n"
            "        An older release imports fine and then dies at the FIRST quantized "
            "forward pass with an opaque CUDA kernel error — which is why this is caught "
            "here rather than an hour into the run.\n"
            f"        FIX: pip install -U 'bitsandbytes>={MIN_BITSANDBYTES_VERSION[0]}."
            f"{MIN_BITSANDBYTES_VERSION[1]}'",
        )

    return (
        "bitsandbytes_blackwell",
        True,
        f"bitsandbytes {version} is at or above the first Blackwell-capable release.",
    )


def _check_free_vram(torch: object, config: TrainingConfig, estimate_gb: float) -> tuple[str, bool, str]:
    """Compares against ACTUAL free memory rather than total — Ollama, a
    browser, or a previous crashed run can already be holding several GB of
    the 12."""
    free_bytes, total_bytes = torch.cuda.mem_get_info()  # type: ignore[attr-defined]
    free_gb = free_bytes / _BYTES_PER_GB
    total_gb = total_bytes / _BYTES_PER_GB
    held_gb = total_gb - free_gb

    if free_gb >= estimate_gb:
        return (
            "free_vram",
            True,
            f"{free_gb:.2f}GB free of {total_gb:.2f}GB total; estimate needs "
            f"{estimate_gb:.2f}GB.",
        )

    return (
        "free_vram",
        False,
        f"Only {free_gb:.2f}GB of {total_gb:.2f}GB VRAM is free, but the estimated peak is "
        f"{estimate_gb:.2f}GB ({held_gb:.2f}GB is already held by other processes).\n"
        "        FIX, in the order worth trying:\n"
        "        1. Stop Ollama — it holds loaded models resident: `ollama stop --all` "
        "(or quit the tray app)\n"
        "        2. Close browsers and anything else using the GPU\n"
        f"        3. Halve max_seq_length ({config.max_seq_length} -> "
        f"{max(128, config.max_seq_length // 2)})\n"
        f"        4. Drop lora_r ({config.lora_r} -> {max(4, config.lora_r // 2)})\n"
        "        5. Move to a smaller base model (a 3B instead of a 7B)",
    )


def _check_windows_spillover(estimate_gb: float, free_gb: float | None) -> tuple[str, bool, str]:
    """Not a pass/fail gate — a diagnostic the user needs BEFORE the run,
    because the symptom is otherwise unreadable. On Windows the WDDM driver
    pages GPU memory to system RAM under pressure instead of raising a
    clean CUDA OOM, so an over-budget run does not crash: it goes 10-50x
    slower and looks like a hung process."""
    if sys.platform != "win32":
        return (
            "windows_vram_spillover",
            True,
            f"Not on Windows (platform={sys.platform}); WDDM paging does not apply.",
        )

    base = (
        "Windows WDDM will SILENTLY PAGE VRAM to system RAM under pressure instead of "
        "raising CUDA OOM. The symptom is not a crash — it is a run that becomes 10-50x "
        "slower and looks hung. If throughput collapses mid-run, read it as a memory "
        "problem, not a stall."
    )

    if free_gb is not None and free_gb < estimate_gb * _VRAM_HEADROOM_FACTOR:
        return (
            "windows_vram_spillover",
            True,
            f"WARNING — thin headroom: {free_gb:.2f}GB free against a {estimate_gb:.2f}GB "
            f"estimate (under the {_VRAM_HEADROOM_FACTOR:.1f}x margin where paging "
            f"typically starts). {base}",
        )

    return ("windows_vram_spillover", True, f"WARNING — {base}")


def _count_examples(dataset_path: Path) -> int:
    if not dataset_path.exists():
        return 0
    with dataset_path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _check_dataset(dataset_path: Path, config: TrainingConfig) -> tuple[str, bool, str]:
    if not dataset_path.exists():
        return (
            "dataset",
            False,
            f"No dataset at {dataset_path}. Build one first:\n"
            "        python -m nexus.training.dataset --output "
            f"{dataset_path}",
        )

    problems = validate_dataset(dataset_path, max_seq_length=config.max_seq_length)
    if problems:
        shown = "\n".join(f"        - {p}" for p in problems[:10])
        more = f"\n        ... and {len(problems) - 10} more" if len(problems) > 10 else ""
        return (
            "dataset",
            False,
            f"{len(problems)} validation problem(s) in {dataset_path}:\n{shown}{more}\n"
            "        FIX: rebuild the dataset, or raise max_seq_length only if VRAM allows "
            "it (it usually does not at 12GB — dropping the over-long examples is the "
            "cheaper fix).",
        )

    return (
        "dataset",
        True,
        f"{_count_examples(dataset_path)} example(s) in {dataset_path}, no validation problems.",
    )


def _check_disk_space(config: TrainingConfig, example_count: int) -> tuple[str, bool, str]:
    param_count_b = infer_param_count_b(config.base_model)
    adapter_gb = param_count_b * config.lora_r * _ADAPTER_GB_PER_B_PER_RANK
    steps = total_training_steps(config, example_count=max(1, example_count))
    checkpoints = max(1, steps // max(1, config.save_steps))
    required_gb = adapter_gb * checkpoints

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(output_dir).free / _BYTES_PER_GB

    if free_gb >= required_gb:
        return (
            "disk_space",
            True,
            f"{free_gb:.1f}GB free at {output_dir}; {checkpoints} checkpoint(s) at "
            f"~{adapter_gb:.2f}GB each need ~{required_gb:.2f}GB.",
        )

    return (
        "disk_space",
        False,
        f"Only {free_gb:.1f}GB free at {output_dir} but {checkpoints} checkpoints at "
        f"~{adapter_gb:.2f}GB each need ~{required_gb:.2f}GB.\n"
        f"        FIX: free disk space, raise save_steps ({config.save_steps} -> "
        f"{config.save_steps * 2}) to keep fewer checkpoints, or point output_dir at a "
        f"larger drive.",
    )


def run_preflight(
    config: TrainingConfig,
    *,
    dataset_path: str | Path | None = None,
    probe_gpu: bool = True,
) -> PreflightResult:
    """Verifies the environment can actually train, in dependency order,
    with an ACTIONABLE message on every failure.

    Checks that depend on a failed one are reported as skipped rather than
    re-failing with a confusing cascade; independent checks (dataset, disk)
    always run, so one invocation surfaces every problem the user has to
    fix rather than making them re-run after each fix.

    probe_gpu=False omits every check that would have to IMPORT the ML
    stack, leaving only the plan-and-data checks. train.py's --dry-run uses
    it so a dry run stays honest about what it did not look at, rather than
    importing torch behind the user's back on a machine where the whole
    point was that the ML stack may not be installed.
    """
    checks: list[tuple[str, bool, str]] = []

    param_count_b = infer_param_count_b(config.base_model)
    estimate_gb = estimate_peak_vram_gb(config, param_count_b=param_count_b)

    free_gb: float | None = None
    if probe_gpu:
        torch_check, torch_module = _check_torch_cuda()
        checks.append(torch_check)

        if torch_check[1] and torch_module is not None:
            checks.append(_check_blackwell(torch_module))
            checks.append(_check_free_vram(torch_module, config, estimate_gb))
            try:
                free_bytes, _total = torch_module.cuda.mem_get_info()  # type: ignore[attr-defined]
                free_gb = free_bytes / _BYTES_PER_GB
            except Exception:  # noqa: BLE001 - a diagnostic must never break preflight
                free_gb = None
        else:
            skipped = "skipped — requires a working torch CUDA install (see torch_available)."
            checks.append(("blackwell_sm120_support", False, skipped))
            checks.append(("free_vram", False, skipped))

        checks.append(_check_bitsandbytes(load_in_4bit=config.load_in_4bit))
        checks.append(_check_windows_spillover(estimate_gb, free_gb))
    else:
        checks.append(
            (
                "gpu_environment",
                True,
                "NOT PROBED — this was a dry run, which imports nothing from the ML stack.\n"
                "        The torch/Blackwell/bitsandbytes/VRAM checks have NOT run, so a "
                "passing dry run says the plan and the data are sound, not that this machine "
                "can train.\n"
                "        Verify the GPU separately: python -m nexus.training.preflight",
            )
        )

    resolved_dataset = Path(dataset_path) if dataset_path else Path(config.output_dir) / "dataset.jsonl"
    dataset_check = _check_dataset(resolved_dataset, config)
    checks.append(dataset_check)

    example_count = _count_examples(resolved_dataset) if dataset_check[1] else 0
    checks.append(_check_disk_space(config, example_count))

    return PreflightResult(ok=all(passed for _name, passed, _detail in checks), checks=checks)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify this machine can actually run a NEXUS fine-tune."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a TrainingConfig YAML")
    parser.add_argument("--dataset", type=str, default=None, help="Path to the training dataset")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = TrainingConfig.from_yaml(args.config) if args.config else TrainingConfig()
    result = run_preflight(config, dataset_path=args.dataset)
    print(result.render())
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
