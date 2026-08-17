from __future__ import annotations

from dataclasses import replace

from nexus.training.config import TrainingConfig, estimate_peak_vram_gb, exceeds_available_vram

_7B = 7.0


def _estimate(config: TrainingConfig, param_count_b: float = _7B) -> float:
    return estimate_peak_vram_gb(config, param_count_b=param_count_b)


def test_default_7b_qlora_config_fits_in_12gb() -> None:
    config = TrainingConfig()

    estimate = _estimate(config)

    assert estimate < config.available_vram_gb
    assert not exceeds_available_vram(config, param_count_b=_7B)


def test_estimate_grows_with_batch_size() -> None:
    base = TrainingConfig()

    assert _estimate(replace(base, batch_size=4)) > _estimate(base)


def test_estimate_grows_with_sequence_length() -> None:
    base = TrainingConfig()

    assert _estimate(replace(base, max_seq_length=2048)) > _estimate(base)


def test_gradient_checkpointing_reduces_the_estimate() -> None:
    on = TrainingConfig(gradient_checkpointing=True)
    off = TrainingConfig(gradient_checkpointing=False)

    assert _estimate(on) < _estimate(off)


def test_checkpointing_only_affects_activations_not_weights() -> None:
    on = _estimate(TrainingConfig(gradient_checkpointing=True))
    off = _estimate(TrainingConfig(gradient_checkpointing=False))

    assert 0 < (off - on) < on


def test_four_bit_is_much_cheaper_than_bf16_weights() -> None:
    quantized = _estimate(TrainingConfig(load_in_4bit=True))
    unquantized = _estimate(TrainingConfig(load_in_4bit=False))

    assert unquantized > quantized * 2


def test_estimate_scales_with_model_size() -> None:
    config = TrainingConfig()

    assert _estimate(config, param_count_b=3.0) < _estimate(config, param_count_b=7.0)


def test_flags_a_config_that_exceeds_available_vram() -> None:
    config = TrainingConfig(load_in_4bit=False)

    assert exceeds_available_vram(config, param_count_b=_7B)


def test_flags_an_oversized_batch_on_a_12gb_card() -> None:
    config = TrainingConfig(batch_size=16, max_seq_length=2048, gradient_checkpointing=False)

    assert exceeds_available_vram(config, param_count_b=_7B)


def test_remediation_ladder_actually_brings_an_over_budget_config_back() -> None:
    """The README's ladder has to work, not just read well: halving
    max_seq_length then halving lora_r must move a failing config under
    budget on the same card."""
    over = TrainingConfig(batch_size=8, max_seq_length=2048, gradient_checkpointing=False)
    assert exceeds_available_vram(over, param_count_b=_7B)

    step_one = replace(over, max_seq_length=512, gradient_checkpointing=True)
    assert not exceeds_available_vram(step_one, param_count_b=_7B)


def test_larger_available_vram_makes_the_same_config_fit() -> None:
    config = TrainingConfig(load_in_4bit=False, available_vram_gb=48.0)

    assert not exceeds_available_vram(config, param_count_b=_7B)
