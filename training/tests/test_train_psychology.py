"""training/train_psychology.py trains nothing yet -- there are no outcome
labels -- but its loader is the contract a future training run depends on, and
its refusals are the point: a silently skipped row is a silently biased dataset.
"""

from __future__ import annotations

import json

import pytest

from ai.psychology_ai.constants import SCHEMA_VERSION
from ai.psychology_ai.feature_extraction import FEATURE_NAMES
from training import train_psychology as tp


def _row(**overrides) -> dict:
    row = {
        "features": dict.fromkeys(FEATURE_NAMES, 50.0),
        tp.REQUIRED_LABEL: 7.0,
        "assessment_id": "a1",
        "player_id": "p1",
        "schema_version": SCHEMA_VERSION,
    }
    row.update(overrides)
    return row


def _write(tmp_path, rows) -> str:
    path = tmp_path / "labelled.jsonl"
    path.write_text("\n".join(json.dumps(r) if isinstance(r, dict) else r for r in rows), encoding="utf-8")
    return str(path)


def test_no_dataset_is_an_empty_list_never_synthesized_rows():
    assert tp.load_dataset(None) == []


def test_a_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        tp.load_dataset(str(tmp_path / "nope.jsonl"))


def test_valid_rows_load_with_every_feature(tmp_path):
    examples = tp.load_dataset(_write(tmp_path, [_row(), "", _row(assessment_id="a2")]))
    assert len(examples) == 2, "blank lines are skipped, not errors"
    assert set(examples[0].features) == set(FEATURE_NAMES)
    assert examples[0].label == 7.0
    assert examples[1].assessment_id == "a2"


def test_a_row_without_the_outcome_label_is_refused(tmp_path):
    row = _row()
    del row[tp.REQUIRED_LABEL]
    with pytest.raises(ValueError, match="unlabelled"):
        tp.load_dataset(_write(tmp_path, [row]))


def test_a_row_missing_a_feature_is_refused(tmp_path):
    features = dict.fromkeys(FEATURE_NAMES[1:], 50.0)
    with pytest.raises(ValueError, match=FEATURE_NAMES[0]):
        tp.load_dataset(_write(tmp_path, [_row(features=features)]))


def test_a_row_without_a_features_object_is_refused(tmp_path):
    with pytest.raises(ValueError, match="features"):
        tp.load_dataset(_write(tmp_path, [_row(features=[1, 2, 3])]))


def test_features_from_another_schema_version_are_refused(tmp_path):
    """Features computed under different constants are not comparable."""
    with pytest.raises(ValueError, match="schema_version"):
        tp.load_dataset(_write(tmp_path, [_row(schema_version="v0-old")]))


def test_the_error_names_the_offending_line(tmp_path):
    bad = _row()
    del bad[tp.REQUIRED_LABEL]
    with pytest.raises(ValueError, match=r"labelled\.jsonl:2:"):
        tp.load_dataset(_write(tmp_path, [_row(), bad]))


def test_training_with_no_data_reports_and_exits_nonzero(capsys):
    assert tp.train([]) == 1
    assert "No labelled dataset" in capsys.readouterr().err


def test_training_with_data_says_no_model_exists_rather_than_faking_one(tmp_path):
    examples = tp.load_dataset(_write(tmp_path, [_row()]))
    with pytest.raises(NotImplementedError, match="HeuristicReadinessModel"):
        tp.train(examples)


def test_check_mode_explains_the_requirements_and_succeeds(capsys):
    assert tp.main(["--check"]) == 0
    out = capsys.readouterr().out
    assert tp.REQUIRED_LABEL in out
    assert str(len(FEATURE_NAMES)) in out


def test_default_run_exits_nonzero_because_nothing_can_be_trained():
    assert tp.main([]) == 1
