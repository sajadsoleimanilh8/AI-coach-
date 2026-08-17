from __future__ import annotations

import json
from pathlib import Path

_CHARS_PER_TOKEN = 4

_DEFAULT_IMBALANCE_THRESHOLD = 0.6

_VALID_ROLES = {"system", "user", "assistant", "tool"}


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN) if text else 0


def _prompt_signature(messages: list[dict]) -> str:
    """Everything up to (not including) the final assistant turn — two
    examples teaching different answers to the same prompt are the
    duplicate worth catching, not two that merely share a system prompt."""
    prompt_parts = [
        f"{m.get('role')}:{m.get('content')}" for m in messages if m.get("role") != "assistant"
    ]
    return "\n".join(prompt_parts)


def validate_dataset(
    path: str | Path,
    *,
    max_seq_length: int,
    imbalance_threshold: float = _DEFAULT_IMBALANCE_THRESHOLD,
) -> list[str]:
    """Returns problems (empty list = clean)."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        return [f"dataset not found: {dataset_path}"]

    problems: list[str] = []
    seen_prompts: dict[str, int] = {}
    task_type_counts: dict[str, int] = {}
    row_count = 0

    with dataset_path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row_count += 1

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                problems.append(f"line {line_number}: malformed JSON ({exc.msg})")
                continue

            if not isinstance(data, dict):
                problems.append(f"line {line_number}: row is not a JSON object")
                continue

            messages = data.get("messages")
            if not isinstance(messages, list) or not messages:
                problems.append(f"line {line_number}: missing or empty 'messages'")
                continue

            if any(
                not isinstance(m, dict) or m.get("role") not in _VALID_ROLES for m in messages
            ):
                problems.append(f"line {line_number}: message with a missing or unknown role")
                continue

            assistant_turns = [m for m in messages if m.get("role") == "assistant"]
            if not assistant_turns:
                problems.append(f"line {line_number}: no assistant turn to train on")
            elif any(not str(m.get("content", "")).strip() for m in assistant_turns):
                problems.append(f"line {line_number}: empty assistant turn")

            token_estimate = estimate_tokens(
                "\n".join(str(m.get("content", "")) for m in messages)
            )
            if token_estimate > max_seq_length:
                problems.append(
                    f"line {line_number}: ~{token_estimate} tokens exceeds "
                    f"max_seq_length={max_seq_length}"
                )

            signature = _prompt_signature(messages)
            if signature in seen_prompts:
                problems.append(
                    f"line {line_number}: duplicate prompt (first seen on line "
                    f"{seen_prompts[signature]})"
                )
            else:
                seen_prompts[signature] = line_number

            task_type = str(data.get("task_type", "unknown"))
            task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1

    if row_count == 0:
        problems.append("dataset is empty")
        return problems

    problems.extend(_imbalance_problems(task_type_counts, row_count, imbalance_threshold))
    return problems


def _imbalance_problems(
    counts: dict[str, int], total: int, threshold: float
) -> list[str]:
    """Only meaningful once there is more than one class to be imbalanced
    between — a single-task dataset (a templates-only export, say) is
    narrow by construction, not skewed."""
    if len(counts) < 2:
        return []
    dominant_type, dominant_count = max(counts.items(), key=lambda item: item[1])
    share = dominant_count / total
    if share <= threshold:
        return []
    return [
        f"class imbalance: task_type={dominant_type!r} is {share:.0%} of "
        f"{total} examples (threshold {threshold:.0%})"
    ]
