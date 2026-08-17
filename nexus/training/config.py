from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

import yaml

_BYTES_PER_PARAM_B_4BIT = 0.55
_BYTES_PER_PARAM_B_BF16 = 2.0

_ADAPTER_AND_OPTIMIZER_GB = 1.0

_ACTIVATION_GB_PER_TOKEN_PER_B = 2.0e-4

_GRADIENT_CHECKPOINTING_RETENTION = 0.4

_PARAM_COUNT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b\b", re.IGNORECASE)
_FALLBACK_PARAM_COUNT_B = 7.0


@dataclass
class TrainingConfig:
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    method: Literal["qlora", "lora"] = "qlora"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_seq_length: int = 1024
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    dataloader_num_workers: int = 4
    save_steps: int = 100
    resume_from_checkpoint: str | None = None
    output_dir: str = "nexus/training/output"
    available_vram_gb: float = 12.0
    seed: int = 42
    val_split: float = 0.1
    eval_steps: int = 25
    save_total_limit: int = 3
    early_stopping_patience: int = 3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    def to_yaml(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(asdict(self), fh, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        with Path(path).open("r", encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def infer_param_count_b(base_model: str) -> float:
    """Pulls "7B"/"3b" out of a HuggingFace model id. Falls back to 7B
    rather than raising: a wrong-but-stated estimate that the caller can
    override beats blocking a config from being described at all."""
    match = _PARAM_COUNT_RE.search(base_model)
    return float(match.group(1)) if match else _FALLBACK_PARAM_COUNT_B


def estimate_peak_vram_gb(config: TrainingConfig, *, param_count_b: float) -> float:
    """Deterministic peak-VRAM estimate from (params, quantization, batch,
    seq_len, gradient_checkpointing).
    """
    per_param = _BYTES_PER_PARAM_B_4BIT if config.load_in_4bit else _BYTES_PER_PARAM_B_BF16
    weights_gb = param_count_b * per_param

    activations_gb = (
        config.batch_size
        * config.max_seq_length
        * param_count_b
        * _ACTIVATION_GB_PER_TOKEN_PER_B
    )
    if config.gradient_checkpointing:
        activations_gb *= _GRADIENT_CHECKPOINTING_RETENTION

    return round(weights_gb + _ADAPTER_AND_OPTIMIZER_GB + activations_gb, 3)


def exceeds_available_vram(config: TrainingConfig, *, param_count_b: float) -> bool:
    return estimate_peak_vram_gb(config, param_count_b=param_count_b) > config.available_vram_gb


def total_training_steps(config: TrainingConfig, *, example_count: int) -> int:
    """Optimizer steps across the whole run — what the dry-run plan reports
    and what save_steps is counted against.
    """
    effective_batch = max(1, config.batch_size * config.gradient_accumulation_steps)
    steps_per_epoch = max(1, math.ceil(example_count / effective_batch))
    return steps_per_epoch * config.num_epochs
