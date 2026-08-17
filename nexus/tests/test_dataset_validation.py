from __future__ import annotations

import json

from nexus.training.validation import estimate_tokens, validate_dataset


def _write(tmp_path, rows: list[dict | str]) -> str:
    path = tmp_path / "dataset.jsonl"
    lines = [row if isinstance(row, str) else json.dumps(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _row(user: str, assistant: str, task_type: str = "general") -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "source": "synthetic",
        "task_type": task_type,
        "weight": 1.0,
    }


def test_clean_dataset_reports_no_problems(tmp_path) -> None:
    path = _write(tmp_path, [_row("Question one?", "Answer one."), _row("Question two?", "Answer two.", "coding")])

    assert validate_dataset(path, max_seq_length=1024) == []


def test_malformed_json_row_is_reported_with_its_line_number(tmp_path) -> None:
    path = _write(tmp_path, [_row("ok?", "fine."), "{not valid json"])

    problems = validate_dataset(path, max_seq_length=1024)

    assert any("line 2" in p and "malformed JSON" in p for p in problems)


def test_missing_messages_is_reported(tmp_path) -> None:
    path = _write(tmp_path, [{"source": "synthetic", "task_type": "general"}])

    assert any("missing or empty 'messages'" in p for p in validate_dataset(path, max_seq_length=1024))


def test_empty_assistant_turn_is_reported(tmp_path) -> None:
    path = _write(tmp_path, [_row("Question?", "   ")])

    assert any("empty assistant turn" in p for p in validate_dataset(path, max_seq_length=1024))


def test_missing_assistant_turn_is_reported(tmp_path) -> None:
    path = _write(
        tmp_path, [{"messages": [{"role": "user", "content": "Only a prompt"}], "task_type": "general"}]
    )

    assert any("no assistant turn" in p for p in validate_dataset(path, max_seq_length=1024))


def test_unknown_role_is_reported(tmp_path) -> None:
    path = _write(
        tmp_path,
        [{"messages": [{"role": "wizard", "content": "x"}, {"role": "assistant", "content": "y"}]}],
    )

    assert any("unknown role" in p for p in validate_dataset(path, max_seq_length=1024))


def test_over_length_example_is_reported(tmp_path) -> None:
    path = _write(tmp_path, [_row("word " * 5000, "Answer.")])

    problems = validate_dataset(path, max_seq_length=1024)

    assert any("exceeds max_seq_length=1024" in p for p in problems)


def test_an_example_within_the_limit_is_not_flagged(tmp_path) -> None:
    path = _write(tmp_path, [_row("short question?", "short answer.")])

    assert not any("max_seq_length" in p for p in validate_dataset(path, max_seq_length=1024))


def test_duplicate_prompts_are_reported(tmp_path) -> None:
    path = _write(
        tmp_path,
        [_row("Same question?", "First answer."), _row("Same question?", "Different answer.")],
    )

    problems = validate_dataset(path, max_seq_length=1024)

    assert any("duplicate prompt" in p and "line 1" in p for p in problems)


def test_different_prompts_with_the_same_answer_are_not_duplicates(tmp_path) -> None:
    path = _write(tmp_path, [_row("Question A?", "Yes."), _row("Question B?", "Yes.")])

    assert not any("duplicate" in p for p in validate_dataset(path, max_seq_length=1024))


def test_class_imbalance_beyond_threshold_is_reported(tmp_path) -> None:
    rows = [_row(f"Question {i}?", "Answer.", "coding") for i in range(9)]
    rows.append(_row("Odd one out?", "Answer.", "general"))
    path = _write(tmp_path, rows)

    problems = validate_dataset(path, max_seq_length=1024, imbalance_threshold=0.6)

    assert any("class imbalance" in p and "coding" in p for p in problems)


def test_balanced_dataset_is_not_flagged_for_imbalance(tmp_path) -> None:
    rows = [_row(f"C{i}?", "A.", "coding") for i in range(5)]
    rows += [_row(f"G{i}?", "A.", "general") for i in range(5)]
    path = _write(tmp_path, rows)

    assert not any("class imbalance" in p for p in validate_dataset(path, max_seq_length=1024))


def test_single_task_type_is_narrow_not_imbalanced(tmp_path) -> None:
    rows = [_row(f"Q{i}?", "A.", "coding") for i in range(10)]
    path = _write(tmp_path, rows)

    assert not any("class imbalance" in p for p in validate_dataset(path, max_seq_length=1024))


def test_empty_dataset_is_reported(tmp_path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    assert "dataset is empty" in validate_dataset(path, max_seq_length=1024)


def test_missing_file_is_reported(tmp_path) -> None:
    problems = validate_dataset(tmp_path / "nope.jsonl", max_seq_length=1024)

    assert any("dataset not found" in p for p in problems)


def test_committed_templates_all_validate() -> None:
    """The shipped behavior templates are training data too — if they do not
    pass their own validator they should not be in the repo."""
    from pathlib import Path

    for template in sorted(Path("nexus/training/templates").glob("*.jsonl")):
        assert validate_dataset(template, max_seq_length=2048) == [], template.name


def test_token_estimate_is_monotonic() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("short") < estimate_tokens("a much longer piece of text here")
