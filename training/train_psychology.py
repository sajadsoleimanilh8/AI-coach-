"""
Training entry point for the psychology mental-readiness model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from ai.psychology_ai.constants import SCHEMA_VERSION  # noqa: E402
from ai.psychology_ai.feature_extraction import FEATURE_NAMES  # noqa: E402

REQUIRED_LABEL = "post_match_performance_rating"

NO_DATASET_MESSAGE = (
    "No labelled dataset exists for mental readiness.\n\n"
    "This script requires one row per assessment containing the 10 normalized\n"
    f"features ({', '.join(FEATURE_NAMES)})\n"
    f"plus a '{REQUIRED_LABEL}' outcome label. Nothing in this repo records\n"
    "that outcome yet, so there is no supervised target to train against.\n\n"
    "The heuristic in ai/psychology_ai/model_interface.py remains the real,\n"
    "disclosed implementation (method='heuristic_proxy'). No model has been\n"
    "trained, and no accuracy figure for one exists."
)


@dataclass(frozen=True)
class LabelledExample:
    """One training row: the stored feature vector plus its outcome label."""

    features: dict[str, float]
    label: float
    assessment_id: str
    player_id: str
    schema_version: str


def load_dataset(path: str | None) -> list[LabelledExample]:
    """Loads labelled examples from a JSONL file."""
    if path is None:
        return []

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    examples: list[LabelledExample] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            features = row.get("features")
            if not isinstance(features, dict):
                raise ValueError(f"{path}:{line_number}: missing 'features' object")

            missing = [name for name in FEATURE_NAMES if name not in features]
            if missing:
                raise ValueError(
                    f"{path}:{line_number}: features missing {', '.join(missing)}"
                )

            if REQUIRED_LABEL not in row:
                raise ValueError(
                    f"{path}:{line_number}: missing outcome label '{REQUIRED_LABEL}'. "
                    "An unlabelled assessment cannot be a training example."
                )

            row_schema = row.get("schema_version", SCHEMA_VERSION)
            if row_schema != SCHEMA_VERSION:
                raise ValueError(
                    f"{path}:{line_number}: schema_version {row_schema!r} does not "
                    f"match the current engine ({SCHEMA_VERSION!r}). Re-extract "
                    "features before training."
                )

            examples.append(
                LabelledExample(
                    features={name: float(features[name]) for name in FEATURE_NAMES},
                    label=float(row[REQUIRED_LABEL]),
                    assessment_id=str(row.get("assessment_id", "")),
                    player_id=str(row.get("player_id", "")),
                    schema_version=row_schema,
                )
            )

    return examples


def train(examples: list[LabelledExample]) -> int:
    """The training step. Not implemented, because nothing can call it with a
    non-empty dataset yet.
    """
    if not examples:
        print(NO_DATASET_MESSAGE, file=sys.stderr)
        return 1

    raise NotImplementedError(
        f"Loaded {len(examples)} labelled examples, but no model implementation "
        "exists to train. Implement a PsychologyReadinessModel subclass (see "
        "ai/psychology_ai/model_interface.py) and fit it here, then benchmark it "
        "against HeuristicReadinessModel on a held-out split before promoting it "
        "to method='ml_trained'."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a mental-readiness model. No labelled dataset exists yet, so "
            "this currently reports that and exits non-zero rather than "
            "producing a model."
        )
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to a JSONL file of labelled examples (features + "
        f"{REQUIRED_LABEL}). Omit to see the current status.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report what this script needs in order to train, and exit 0.",
    )
    args = parser.parse_args(argv)

    if args.check:
        print(NO_DATASET_MESSAGE)
        print(f"\nExpected feature count: {len(FEATURE_NAMES)}")
        print(f"Engine schema version:  {SCHEMA_VERSION}")
        return 0

    return train(load_dataset(args.dataset))


if __name__ == "__main__":
    raise SystemExit(main())
