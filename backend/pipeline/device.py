"""
One device decision per pipeline run, made once and shared by every model.

WHY THIS EXISTS
    Nothing in this repo hardcodes a device: tracker.py's model.track(),
    detectors.py's model.predict() and auto_calibration.py's model.predict()
    all omitted `device=`, so ultralytics auto-selected per call. That is
    fine right up until torch REPORTS a usable GPU that then fails on a real
    kernel launch, which is not hypothetical:

      configs/models.yaml pins torch 2.11.0+cu128, whose compiled arch list
      is ['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120'] (measured on
      this machine). A judge running a GTX 1080 (sm_61), a GTX 980 (sm_52)
      or a V100 (sm_70) gets torch.cuda.is_available() == True and then
      "no kernel image is available for execution on the device" the first
      time a model actually runs. Same class of failure for a driver too old
      for the bundled CUDA runtime, or a card with no free memory.

    So `is_available()` is a necessary gate, not a sufficient one. This
    module adds the sufficient part: run one real op on the GPU and only
    trust it if that op actually completes.

WHY THE DECISION IS PASSED EXPLICITLY RATHER THAN LEFT TO AUTO-SELECT
    A failed CUDA call can leave a sticky error on the context: after it,
    torch.cuda.is_available() may STILL return True while every subsequent
    CUDA call raises. If each call site kept auto-selecting, they would each
    re-pick cuda and re-crash. Passing the resolved device string to every
    model call site is what makes one bad probe result in one clean CPU run
    instead of a crash at frame 1 of stage 1.

COST
    One tiny matmul, once per job, in run_pipeline(). Not per frame, not per
    model. Pose estimation is excluded on purpose: it is MediaPipe, not
    torch, and is CPU-only regardless of what this returns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Ultralytics accepts these strings directly as `device=`.
CPU = "cpu"
CUDA0 = "cuda:0"

MODE_GPU = "gpu"
MODE_FELL_BACK = "gpu_unusable_fell_back"
MODE_CPU = "cpu"


@dataclass(frozen=True)
class DeviceDecision:
    """The device every model in ONE run will use.

    `device` is passed verbatim to ultralytics. `message` is written into
    the job's progress message so a slow run is diagnosable rather than
    mysteriously slow -- see run_pipeline()'s first report() call.
    """

    device: str
    mode: str
    message: str
    detail: str | None = None

    @property
    def is_gpu(self) -> bool:
        return self.device != CPU


def _probe_gpu(torch) -> tuple[bool, str | None]:
    """Run one real op on the GPU. Returns (ok, error_text).

    The op is deliberately a real matmul followed by a synchronise: CUDA
    launch failures are asynchronous, so allocating a tensor alone can
    succeed while the actual kernel fails later, somewhere far less
    convenient. .item() and synchronize() force the error to surface HERE,
    where it can be handled, instead of inside stage 1.
    """
    try:
        x = torch.randn(64, 64, device="cuda")
        value = float((x @ x).sum().item())
        torch.cuda.synchronize()
        if value != value:  # NaN -- a "successful" op that computed garbage
            return False, "GPU matmul returned NaN"
        return True, None
    except Exception as exc:  # noqa: BLE001 -- ANY failure means "do not use this GPU"
        return False, f"{type(exc).__name__}: {exc}"


def resolve_device(force_cpu: bool = False) -> DeviceDecision:
    """Decide once, for this run, which device every model should use.

    Never raises. Every failure path ends on CPU, which is the same
    already-proven code path this pipeline has always used -- falling back
    is not a degraded mode, it is just slower.
    """
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
        # THE CASE THIS MODULE EXISTS FOR. Loud, one line, names the GPU and
        # the actual error, so "why is this slow" is answerable from the log.
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
