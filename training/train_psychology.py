"""
Training entry point for the psychology mental-readiness model.

    THERE IS NOTHING TO TRAIN YET, AND THIS SCRIPT TRAINS NOTHING.

That is the honest state of this module, not an oversight, and this file exists
to say so precisely rather than to leave a bare `# TODO` that reads as "someone
forgot".

WHY THERE IS NO MODEL
---------------------
Supervised training needs labelled examples: an input and the outcome it should
have predicted. For mental readiness the input exists -- the 13-item
questionnaire, normalized by ai/psychology_ai/feature_extraction.py into a
bounded 0-100 vector, and persisted on every psychology_assessments row exactly
so it would be available for this. The LABEL does not exist. Nothing in this
repo records what happened afterwards: no match-day performance rating, no
minutes played, no coach assessment, nothing that could stand as "was this
player actually mentally ready?".

Without labels there is no supervised target, no train/validation split worth
the name, and no metric that would mean anything. Fabricating one -- training
against the heuristic's own output -- would produce a model that reproduces the
heuristic's biases while LOOKING like independent evidence for them. That is
strictly worse than the disclosed heuristic, which at least says plainly what
it is (`method="heuristic_proxy"` on every row it writes).

So ai/psychology_ai/model_interface.py::HeuristicReadinessModel is the real,
disclosed implementation today, and PsychologyReadinessModel is the interface a
trained model implements once the data below exists.

WHAT WOULD MAKE THIS SCRIPT REAL
--------------------------------
1. An outcome label per assessment. Realistically a post-match performance
   rating (coach 1-10, or a derived composite of the PlayerMetric scores for
   that player in that match), joined to the assessment via a resolved player
   identity -- NOT via cv_player_id, which is a per-video ByteTrack tracking ID
   and cannot be followed across matches (see backend/api/player_intelligence.py).
   Resolving that identity is its own unbuilt piece of work; backend/auth/ is
   empty scaffolding today.
2. Enough labelled rows for the split to be meaningful. With 10 features, a few
   hundred assessments is the floor at which a model could beat the heuristic
   for reasons other than chance.
3. A baseline comparison. A trained model only earns the `ml_trained` tier if
   it beats HeuristicReadinessModel on held-out data. If it does not, the
   heuristic stays and this script reports that.

The CLI and the loader below are written out so that step is a matter of
implementing `load_dataset` against a real label source rather than designing
the whole pipeline from scratch. Every path that would otherwise invent data
raises or exits non-zero instead.

Usage:
    python training/train_psychology.py --check
    python training/train_psychology.py --dataset path/to/labelled.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

from ai.psychology_ai.constants import SCHEMA_VERSION  # noqa: E402
from ai.psychology_ai.feature_extraction import FEATURE_NAMES  # noqa: E402

# The label this script would train against, and the reason it cannot yet.
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
    """One training row: the stored feature vector plus its outcome label.

    Deliberately typed even though nothing constructs it yet -- it is the
    contract `load_dataset` has to satisfy, and writing it down is most of the
    remaining design work.
    """

    features: dict[str, float]
    label: float
    assessment_id: str
    player_id: str
    schema_version: str


def load_dataset(path: str | None) -> list[LabelledExample]:
    """Loads labelled examples from a JSONL file.

    Returns an empty list when `path` is None -- the current, expected state.
    It never synthesizes rows, and it never falls back to scoring unlabelled
    assessments with the heuristic and calling those labels.

    Raises FileNotFoundError if a path is given and does not exist, and
    ValueError if a row is missing features or the outcome label. Loudly
    refusing malformed input is the point: a silently-skipped row is a silently
    biased dataset.
    """
    if path is None:
        return []

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    examples: list[LabelledExample] = []
    with open(path, encoding="utf-8") as handle:
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
                # Features computed under different constants are not
                # comparable -- the same reason assessments carry a
                # schema_version at all.
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

    Returns a process exit code. When a real dataset does arrive, this is where
    a model implementing PsychologyReadinessModel gets fitted, compared against
    HeuristicReadinessModel on a held-out split, and only promoted to
    `method="ml_trained"` if it wins.
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
