from __future__ import annotations

from nexus.training.config import TrainingConfig, infer_param_count_b, total_training_steps


def test_defaults_are_tuned_for_a_12gb_card() -> None:
    config = TrainingConfig()

    assert config.available_vram_gb == 12.0
    assert config.batch_size == 1
    assert config.gradient_accumulation_steps == 16
    assert config.max_seq_length == 1024
    assert config.load_in_4bit is True
    assert config.gradient_checkpointing is True
    assert config.optim == "paged_adamw_8bit"
    assert config.method == "qlora"
    assert config.bnb_4bit_compute_dtype == "bfloat16"


def test_dataloader_workers_capped_for_windows_spawn() -> None:
    assert TrainingConfig().dataloader_num_workers <= 8


def test_save_steps_frequent_enough_for_a_throttling_laptop() -> None:
    assert TrainingConfig().save_steps == 100


def test_yaml_round_trip_preserves_every_field(tmp_path) -> None:
    original = TrainingConfig(
        base_model="mistralai/Mistral-3B",
        lora_r=8,
        learning_rate=1e-4,
        num_epochs=5,
        max_seq_length=512,
        seed=1234,
        resume_from_checkpoint="output/checkpoint-300",
    )
    path = tmp_path / "config.yaml"

    original.to_yaml(path)
    loaded = TrainingConfig.from_yaml(path)

    assert loaded == original


def test_seed_survives_the_round_trip(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    TrainingConfig(seed=99).to_yaml(path)

    assert TrainingConfig.from_yaml(path).seed == 99


def test_from_yaml_ignores_unknown_keys(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("base_model: some/model\nnot_a_real_field: 1\n", encoding="utf-8")

    config = TrainingConfig.from_yaml(path)

    assert config.base_model == "some/model"
    assert config.seed == TrainingConfig().seed


def test_param_count_inferred_from_model_name() -> None:
    assert infer_param_count_b("mistralai/Mistral-7B-Instruct-v0.3") == 7.0
    assert infer_param_count_b("Qwen/Qwen2.5-3B-Instruct") == 3.0
    assert infer_param_count_b("some/model-1.5b") == 1.5


def test_param_count_falls_back_rather_than_raising() -> None:
    assert infer_param_count_b("org/model-without-a-size") == 7.0


def test_total_steps_accounts_for_accumulation_and_epochs() -> None:
    config = TrainingConfig(batch_size=1, gradient_accumulation_steps=16, num_epochs=3)

    assert total_training_steps(config, example_count=320) == 60


def test_total_steps_ceils_a_trailing_partial_batch_like_hf_does() -> None:
    config = TrainingConfig(batch_size=1, gradient_accumulation_steps=16, num_epochs=3)

    assert total_training_steps(config, example_count=61) == 12


def test_total_steps_matches_hf_arithmetic_across_remainders() -> None:
    config = TrainingConfig(batch_size=2, gradient_accumulation_steps=4, num_epochs=1)

    assert total_training_steps(config, example_count=8) == 1
    assert total_training_steps(config, example_count=9) == 2
    assert total_training_steps(config, example_count=15) == 2
    assert total_training_steps(config, example_count=16) == 2
    assert total_training_steps(config, example_count=17) == 3


def test_total_steps_never_returns_zero_for_a_tiny_dataset() -> None:
    config = TrainingConfig(num_epochs=2)

    assert total_training_steps(config, example_count=1) == 2


def test_local_directory_base_model_pins_from_pretrained_offline(tmp_path) -> None:
    from nexus.training.train import source_kwargs

    model_dir = tmp_path / "Mistral-7B-Instruct-v0.3"
    model_dir.mkdir()

    assert source_kwargs(str(model_dir)) == {"local_files_only": True}


def test_hub_id_base_model_is_left_resolvable() -> None:
    from nexus.training.train import source_kwargs

    assert source_kwargs("mistralai/Mistral-7B-Instruct-v0.3") == {}
    assert source_kwargs("") == {}


def test_nonexistent_local_path_is_not_pinned_offline(tmp_path) -> None:
    from nexus.training.train import source_kwargs

    assert source_kwargs(str(tmp_path / "does-not-exist")) == {}
