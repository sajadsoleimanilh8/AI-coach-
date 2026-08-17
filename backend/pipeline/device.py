"""
One device decision per pipeline run, made once and shared by every model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

CPU = "cpu"
CUDA0 = "cuda:0"

MODE_GPU = "gpu"
MODE_FELL_BACK = "gpu_unusable_fell_back"
MODE_CPU = "cpu"


@dataclass(frozen=True)
class DeviceDecision:
    """The device every model in ONE run will use."""

    device: str
    mode: str
    message: str
    detail: str | None = None

    @property
    def is_gpu(self) -> bool:
        return self.device != CPU


def _probe_gpu(torch) -> tuple[bool, str | None]:
    """Run one real op on the GPU. Returns (ok, error_text)."""
    try:
        x = torch.randn(64, 64, device="cuda")
        value = float((x @ x).sum().item())
        torch.cuda.synchronize()
        if value != value:
            return False, "GPU matmul returned NaN"
        return True, None
    except Exception as exc:  # noqa: BLE001 -- ANY failure means "do not use this GPU"
        return False, f"{type(exc).__name__}: {exc}"


def resolve_device(force_cpu: bool = False) -> DeviceDecision:
    """Decide once, for this run, which device every model should use."""
    if force_cpu:
        return DeviceDecision(
            device=CPU, mode=MODE_CPU,
            message="Running on CPU (forced by configuration) -- processing will take longer.",
        )

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return DeviceDecision(
            device=CPU, mode=MODE_CPU,
            message="Running on CPU (PyTorch unavailable) -- processing will take longer.",
            detail=f"{type(exc).__name__}: {exc}",
        )

    try:
        available = torch.cuda.is_available()
    except Exception as exc:  # noqa: BLE001 -- a broken driver can raise here too
        return DeviceDecision(
            device=CPU, mode=MODE_CPU,
            message="Running on CPU (no usable GPU detected) -- processing will take longer.",
            detail=f"{type(exc).__name__}: {exc}",
        )

    if not available:
        logger.info("no CUDA device reported by torch; running on CPU")
        return DeviceDecision(
            device=CPU, mode=MODE_CPU,
            message="Running on CPU (no GPU detected) -- processing will take longer.",
        )

    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        gpu_name = "unknown GPU"

    ok, error_text = _probe_gpu(torch)
    if not ok:
        logger.warning(
            "GPU %s reported available but failed a real test operation (%s) -- "
            "falling back to CPU for this run", gpu_name, error_text,
        )
        return DeviceDecision(
            device=CPU, mode=MODE_FELL_BACK,
            message=(f"GPU detected ({gpu_name}) but not usable on this run -- "
                     f"falling back to CPU, processing will take longer."),
            detail=error_text,
        )

    logger.info("using GPU %s for this run", gpu_name)
    return DeviceDecision(
        device=CUDA0, mode=MODE_GPU,
        message=f"Running on GPU ({gpu_name}).",
    )
