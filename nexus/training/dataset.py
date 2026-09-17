from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from nexus.evaluation.store import EvalStore
from nexus.evaluation.types import EvalCase
from nexus.intelligence.privacy_classifier import PrivacyClassifier
from nexus.logging_setup.logger import get_logger
from nexus.memory.storage import InteractionRecord, make_session_factory
from nexus.training.validation import estimate_tokens

logger = get_logger("training.dataset")

TEMPLATES_DIR = Path(__file__).parent / "templates"

ALL_SOURCES: tuple[str, ...] = ("high_feedback", "verified_high", "eval_failure", "synthetic")

# Below this many records, a ProcessPoolExecutor costs more than it saves:
# Windows spawns workers rather than forking, so every worker pays a fresh
# interpreter start plus a re-import of nexus.intelligence before it
# classifies its first string. 24 cores only start paying off once there
# is real work to hand them.
_PARALLEL_CLASSIFY_THRESHOLD = 400

_P95 = 0.95

# Set once per pool worker by _init_classifier_worker — a PrivacyClassifier
# compiles its full pattern set on construction, which should happen once
# per process, not once per record.
_WORKER_CLASSIFIER: PrivacyClassifier | None = None


@dataclass
class TrainingExample:
    messages: list[dict[str, str]]
    source: str  # high_feedback|eval_failure|verified_high|synthetic
    task_type: str
    weight: float = 1.0

    def to_json_line(self) -> str:
        return json.dumps(
            {
                "messages": self.messages,
                "source": self.source,
                "task_type": self.task_type,
                "weight": self.weight,
            },
            ensure_ascii=False,
        )


@dataclass
class DatasetStats:
    total: int
    by_source: dict[str, int] = field(default_factory=dict)
    by_task_type: dict[str, int] = field(default_factory=dict)
    excluded_private: int = 0
    mean_tokens: float = 0.0
    p95_tokens: int = 0


def _example_text(example: TrainingExample) -> str:
    return "\n".join(m.get("content", "") for m in example.messages)


def _init_classifier_worker() -> None:
    global _WORKER_CLASSIFIER
    _WORKER_CLASSIFIER = PrivacyClassifier()


def _classify_level(text: str) -> str:
    assert _WORKER_CLASSIFIER is not None
    return _WORKER_CLASSIFIER.classify(text).level


def _percentile(values: Sequence[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _load_eval_cases(datasets_dir: Path) -> dict[str, EvalCase]:
    """case_id -> EvalCase across every suite. EvalCaseRecord persists only
    the scored summary (see EvalStore.get_run's note), so the `expected`
    payload an eval_failure example needs as its target has to come back
    from the dataset files themselves."""
    cases: dict[str, EvalCase] = {}
    if not datasets_dir.exists():
        return cases
    for path in sorted(datasets_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                cases[data["id"]] = EvalCase(
                    id=data["id"],
                    suite=data["suite"],
                    input=data["input"],
                    expected=data["expected"],
                    metadata=data.get("metadata", {}),
                )
    return cases


def _render(payload: dict[str, Any]) -> str:
    """Single-key payloads read as plain text; anything richer keeps its
    structure, because a routing/tools case's expectation IS the JSON
    shape and flattening it would teach the wrong target."""
    if len(payload) == 1:
        [value] = payload.values()
        if isinstance(value, str):
            return value
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


class DatasetBuilder:
    """Mines training examples from the four sources the improvement loop
    produces, then re-checks every one against the CURRENT privacy
    classifier before it can leave the machine."""

    def __init__(
        self,
        engine: AsyncEngine,
        privacy_classifier: PrivacyClassifier,
        eval_store: EvalStore,
        *,
        datasets_dir: str | Path = "nexus/evaluation/datasets",
        templates_dir: str | Path = TEMPLATES_DIR,
    ) -> None:
        self._engine = engine
        self._session_factory: async_sessionmaker = make_session_factory(engine)
        self._privacy_classifier = privacy_classifier
        self._eval_store = eval_store
        self._datasets_dir = Path(datasets_dir)
        self._templates_dir = Path(templates_dir)

    async def build(
        self,
        *,
        include_sources: Sequence[str] = ALL_SOURCES,
        min_verification_score: float = 0.8,
        exclude_privacy_levels: Sequence[str] = ("private",),
        max_examples: int | None = None,
    ) -> tuple[list[TrainingExample], DatasetStats]:
        sources = set(include_sources)
        candidates: list[TrainingExample] = []

        if {"high_feedback", "verified_high"} & sources:
            interactions = await self._load_interactions()
            if "high_feedback" in sources:
                candidates.extend(self._from_high_feedback(interactions))
            if "verified_high" in sources:
                candidates.extend(
                    self._from_verified_high(interactions, min_verification_score)
                )
        if "eval_failure" in sources:
            candidates.extend(await self._from_eval_failures())
        if "synthetic" in sources:
            candidates.extend(self._from_templates())

        kept, excluded_private = self._filter_by_privacy(candidates, set(exclude_privacy_levels))

        if max_examples is not None:
            kept = kept[:max_examples]

        return kept, self._stats(kept, excluded_private)

    def write_jsonl(self, examples: Iterable[TrainingExample], path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as fh:
            for example in examples:
                fh.write(example.to_json_line() + "\n")

    async def _load_interactions(self) -> list[InteractionRecord]:
        async with self._session_factory() as db:
            result = await db.execute(select(InteractionRecord))
            return list(result.scalars().all())

    @staticmethod
    def _messages_from_interaction(record: InteractionRecord) -> list[dict[str, str]]:
        prompt = json.loads(record.prompt_json)
        return [*prompt, {"role": "assistant", "content": record.response_text}]

    def _from_high_feedback(self, records: list[InteractionRecord]) -> list[TrainingExample]:
        return [
            TrainingExample(
                messages=self._messages_from_interaction(record),
                source="high_feedback",
                task_type=record.task_type,
                # A human explicitly endorsed this answer, which is the
                # strongest signal available here — weighted above the
                # automated sources deliberately.
                weight=1.5,
            )
            for record in records
            if record.user_feedback == 1
        ]

    def _from_verified_high(
        self, records: list[InteractionRecord], min_verification_score: float
    ) -> list[TrainingExample]:
        return [
            TrainingExample(
                messages=self._messages_from_interaction(record),
                source="verified_high",
                task_type=record.task_type,
                weight=1.0,
            )
            for record in records
            if record.verification_band == "high"
            and record.verification_score is not None
            and record.verification_score >= min_verification_score
            # A record already mined as high_feedback should not be counted
            # twice with a second weight.
            and record.user_feedback != 1
        ]

    async def _from_eval_failures(self) -> list[TrainingExample]:
        run = await self._eval_store.latest_run()
        if run is None:
            return []

        cases = _load_eval_cases(self._datasets_dir)
        examples: list[TrainingExample] = []
        for suite in run.suites:
            for outcome in suite.outcomes:
                if outcome.passed:
                    continue
                case = cases.get(outcome.case_id)
                if case is None:
                    logger.warning(
                        "eval failure %s has no dataset case; skipping", outcome.case_id
                    )
                    continue
                examples.append(
                    TrainingExample(
                        messages=[
                            {"role": "user", "content": _render(case.input)},
                            {"role": "assistant", "content": _render(case.expected)},
                        ],
                        source="eval_failure",
                        task_type=suite.suite,
                        # These are exactly the cases the model got wrong,
                        # so they carry more signal per example than a
                        # record of it already succeeding.
                        weight=2.0,
                    )
                )
        return examples

    def _from_templates(self) -> list[TrainingExample]:
        examples: list[TrainingExample] = []
        if not self._templates_dir.exists():
            return examples
        for path in sorted(self._templates_dir.glob("*.jsonl")):
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    examples.append(
                        TrainingExample(
                            messages=data["messages"],
                            source="synthetic",
                            task_type=data.get("task_type", path.stem),
                            weight=float(data.get("weight", 1.0)),
                        )
                    )
        return examples

    def _filter_by_privacy(
        self, examples: list[TrainingExample], excluded: set[str]
    ) -> tuple[list[TrainingExample], int]:
        """Re-classifies at EXPORT time rather than trusting
        InteractionRecord.privacy_level. The stored label was written by
        whatever classifier version was live at request time; if the
        pattern set has tightened since (or an example came from a source
        that never had a label at all, like a template), the stored value
        is stale and trusting it would export exactly the records the
        current rules say must stay on this machine."""
        if not excluded:
            return list(examples), 0

        texts = [_example_text(example) for example in examples]
        levels = self._classify_all(texts)

        kept: list[TrainingExample] = []
        excluded_count = 0
        for example, level in zip(examples, levels):
            if level in excluded:
                excluded_count += 1
                continue
            kept.append(example)
        return kept, excluded_count

    def _classify_all(self, texts: list[str]) -> list[str]:
        if len(texts) < _PARALLEL_CLASSIFY_THRESHOLD:
            return [self._privacy_classifier.classify(text).level for text in texts]

        with ProcessPoolExecutor(initializer=_init_classifier_worker) as pool:
            return list(pool.map(_classify_level, texts, chunksize=64))

    @staticmethod
    def _stats(examples: list[TrainingExample], excluded_private: int) -> DatasetStats:
        by_source: dict[str, int] = {}
        by_task_type: dict[str, int] = {}
        token_counts: list[int] = []

        for example in examples:
            by_source[example.source] = by_source.get(example.source, 0) + 1
            by_task_type[example.task_type] = by_task_type.get(example.task_type, 0) + 1
            token_counts.append(estimate_tokens(_example_text(example)))

        return DatasetStats(
            total=len(examples),
            by_source=by_source,
            by_task_type=by_task_type,
            excluded_private=excluded_private,
            mean_tokens=(sum(token_counts) / len(token_counts)) if token_counts else 0.0,
            p95_tokens=_percentile(token_counts, _P95),
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a NEXUS training dataset.")
    parser.add_argument("--output", type=str, default="nexus/training/data/dataset.jsonl")
    parser.add_argument(
        "--sources",
        type=str,
        default=",".join(ALL_SOURCES),
        help=f"Comma-separated subset of: {', '.join(ALL_SOURCES)}",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--include-private",
        action="store_true",
        help="Export private-tier records too. Off by default; nothing leaves this machine "
        "without it being asked for explicitly.",
    )
    return parser.parse_args(argv)


async def _main_async(args: argparse.Namespace) -> int:
    from nexus.config.settings import get_settings
    from nexus.memory.storage import create_async_db_engine

    settings = get_settings()
    engine = create_async_db_engine(settings.memory.database_path)
    eval_store = EvalStore(engine)
    await eval_store.init()

    builder = DatasetBuilder(
        engine,
        PrivacyClassifier(),
        eval_store,
        datasets_dir=settings.evaluation.datasets_dir,
    )
    examples, stats = await builder.build(
        include_sources=[s.strip() for s in args.sources.split(",") if s.strip()],
        min_verification_score=settings.training.min_verification_score,
        exclude_privacy_levels=() if args.include_private else tuple(
            settings.training.exclude_privacy_levels
        ),
        max_examples=args.max_examples,
    )

    builder.write_jsonl(examples, args.output)
    write_stats_sidecar(stats, Path(args.output).with_suffix(".stats.json"))

    print(f"Wrote {stats.total} example(s) to {args.output}")
    print(f"  by source:    {stats.by_source}")
    print(f"  by task_type: {stats.by_task_type}")
    print(f"  excluded (privacy re-classification at export): {stats.excluded_private}")
    print(f"  tokens: mean={stats.mean_tokens:.0f} p95={stats.p95_tokens}")

    await engine.dispose()
    return 0


def main(argv: list[str] | None = None) -> int:
    import asyncio

    return asyncio.run(_main_async(_parse_args(argv)))


def write_stats_sidecar(stats: DatasetStats, path: str | Path) -> None:
    """Dataset provenance next to the dataset itself, so a run months later
    can still answer where its training data came from and how much was
    withheld."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.time(),
        "total": stats.total,
        "by_source": stats.by_source,
        "by_task_type": stats.by_task_type,
        "excluded_private": stats.excluded_private,
        "mean_tokens": stats.mean_tokens,
        "p95_tokens": stats.p95_tokens,
    }
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)


if __name__ == "__main__":
    sys.exit(main())
