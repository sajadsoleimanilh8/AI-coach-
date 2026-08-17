from __future__ import annotations

import json
import sys
import types

import pytest

from nexus.training.config import TrainingConfig
from nexus.training.preflight import (
    BLACKWELL_CAPABILITY,
    MIN_BITSANDBYTES_VERSION,
    run_preflight,
)

_GB = 1024**3


def _fake_torch(
    *,
    version: str = "2.7.0+cu128",
    cuda_available: bool = True,
    capability: tuple[int, int] = BLACKWELL_CAPABILITY,
    arch_list: list[str] | None = None,
    free_gb: float = 11.0,
    total_gb: float = 12.0,
    cuda_version: str = "12.8",
) -> types.ModuleType:
    module = types.ModuleType("torch")
    module.__version__ = version
    module.version = types.SimpleNamespace(cuda=cuda_version)
    module.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        get_device_capability=lambda *a, **k: capability,
        get_arch_list=lambda: arch_list if arch_list is not None else ["sm_80", "sm_90", "sm_120"],
        get_device_name=lambda index=0: "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
        mem_get_info=lambda *a, **k: (int(free_gb * _GB), int(total_gb * _GB)),
    )
    return module


def _fake_bitsandbytes(version: str = "0.45.0") -> types.ModuleType:
    module = types.ModuleType("bitsandbytes")
    module.__version__ = version
    return module


def _dataset(tmp_path) -> str:
    path = tmp_path / "dataset.jsonl"
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"Question {i}?"},
                {"role": "assistant", "content": "Answer."},
            ],
            "task_type": "general",
        }
        for i in range(4)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(path)


def _config(tmp_path) -> TrainingConfig:
    return TrainingConfig(output_dir=str(tmp_path / "output"))


def _check(result, name: str) -> tuple[str, bool, str]:
    return next(c for c in result.checks if c[0] == name)


@pytest.fixture
def healthy_env(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "bitsandbytes", _fake_bitsandbytes())


def test_a_healthy_environment_passes(tmp_path, healthy_env) -> None:
    result = run_preflight(_config(tmp_path), dataset_path=_dataset(tmp_path))

    assert result.ok, result.render()


def test_blackwell_check_fails_loudly_when_sm120_is_absent(tmp_path, monkeypatch) -> None:
    """The single most likely failure on this hardware. The message has to
    name the GPU, the torch build, the arch list, and the exact fix — a bare
    'unsupported' would leave the user guessing at the one thing that is
    actually wrong."""
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(
            version="2.7.0+cu121", arch_list=["sm_70", "sm_80", "sm_90"], cuda_version="12.1"
        ),
    )
    monkeypatch.setitem(sys.modules, "bitsandbytes", _fake_bitsandbytes())

    result = run_preflight(_config(tmp_path), dataset_path=_dataset(tmp_path))
    _name, passed, detail = _check(result, "blackwell_sm120_support")

    assert not result.ok
    assert not passed
    assert "sm_120" in detail
    assert "RTX 5070 Ti" in detail
    assert "2.7.0+cu121" in detail
    assert "12.1" in detail
    assert "sm_90" in detail
    assert "download.pytorch.org/whl/cu128" in detail
    assert "pip uninstall" in detail


def test_blackwell_check_passes_when_sm120_is_present(tmp_path, healthy_env) -> None:
    result = run_preflight(_config(tmp_path), dataset_path=_dataset(tmp_path))
    _name, passed, detail = _check(result, "blackwell_sm120_support")

    assert passed
    assert "sm_120" in detail


def test_missing_torch_fails_with_the_install_command(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)

    result = run_preflight(_config(tmp_path), dataset_path=_dataset(tmp_path))
    _name, passed, detail = _check(result, "torch_available")

    assert not passed
    assert "cu128" in detail
    assert "requirements-train.txt" in detail


def test_cpu_only_wheel_fails_rather_than_falling_back(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=False))
    monkeypatch.setitem(sys.modules, "bitsandbytes", _fake_bitsandbytes())

    result = run_preflight(_config(tmp_path), dataset_path=_dataset(tmp_path))
    _name, passed, detail = _check(result, "torch_available")

    assert not passed
    assert "CPU-only wheel" in detail
    assert "not a supported fallback" in detail


def test_old_bitsandbytes_is_caught_before_the_first_forward_pass(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "bitsandbytes", _fake_bitsandbytes("0.43.1"))

    result = run_preflight(_config(tmp_path), dataset_path=_dataset(tmp_path))
    _name, passed, detail = _check(result, "bitsandbytes_blackwell")

    assert not passed
    assert "0.43.1" in detail
    assert "opaque CUDA kernel error" in detail
    assert f"{MIN_BITSANDBYTES_VERSION[0]}.{MIN_BITSANDBYTES_VERSION[1]}" in detail


def test_bitsandbytes_not_required_when_not_quantizing(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)
    config = _config(tmp_path)
    config.load_in_4bit = False

    result = run_preflight(config, dataset_path=_dataset(tmp_path))
    _name, passed, _detail = _check(result, "bitsandbytes_blackwell")

    assert passed


def test_free_vram_is_compared_against_actual_free_not_total(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(free_gb=3.0, total_gb=12.0))
    monkeypatch.setitem(sys.modules, "bitsandbytes", _fake_bitsandbytes())

    result = run_preflight(_config(tmp_path), dataset_path=_dataset(tmp_path))
    _name, passed, detail = _check(result, "free_vram")

    assert not passed
    assert "3.00GB" in detail
    assert "Stop Ollama" in detail
    assert "max_seq_length" in detail


def test_windows_spillover_warns_without_blocking(tmp_path, healthy_env, monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")

    result = run_preflight(_config(tmp_path), dataset_path=_dataset(tmp_path))
    _name, passed, detail = _check(result, "windows_vram_spillover")

    assert passed
    assert "WARNING" in detail
    assert "SILENTLY PAGE" in detail
    assert "looks hung" in detail


def test_spillover_warning_escalates_when_headroom_is_thin(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(free_gb=5.6, total_gb=12.0))
    monkeypatch.setitem(sys.modules, "bitsandbytes", _fake_bitsandbytes())

    result = run_preflight(_config(tmp_path), dataset_path=_dataset(tmp_path))
    _name, _passed, detail = _check(result, "windows_vram_spillover")

    assert "thin headroom" in detail


def test_missing_dataset_fails_with_how_to_build_one(tmp_path, healthy_env) -> None:
    result = run_preflight(_config(tmp_path), dataset_path=str(tmp_path / "absent.jsonl"))
    _name, passed, detail = _check(result, "dataset")

    assert not passed
    assert "nexus.training.dataset" in detail


def test_invalid_dataset_fails_with_the_specific_problems(tmp_path, healthy_env) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"messages": [{"role": "user", "content": "x"}]}\n', encoding="utf-8")

    result = run_preflight(_config(tmp_path), dataset_path=str(path))
    _name, passed, detail = _check(result, "dataset")

    assert not passed
    assert "no assistant turn" in detail


def test_dependent_checks_are_skipped_not_cascaded(tmp_path, monkeypatch) -> None:
    """With torch missing, the Blackwell and VRAM checks cannot run. They
    must say so rather than emit their own confusing failures."""
    monkeypatch.setitem(sys.modules, "torch", None)

    result = run_preflight(_config(tmp_path), dataset_path=_dataset(tmp_path))

    assert "skipped" in _check(result, "blackwell_sm120_support")[2]
    assert "skipped" in _check(result, "free_vram")[2]
    assert _check(result, "dataset")[1] is True


def test_render_marks_pass_and_fail(tmp_path, healthy_env) -> None:
    rendered = run_preflight(_config(tmp_path), dataset_path=_dataset(tmp_path)).render()

    assert "[PASS]" in rendered
    assert "PREFLIGHT OK" in rendered
