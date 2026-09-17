from __future__ import annotations

import json
import time

import pytest

from nexus.evaluation.store import EvalStore
from nexus.evaluation.types import CaseOutcome, EvalRun, SuiteResult
from nexus.intelligence.privacy_classifier import PrivacyClassifier
from nexus.memory.storage import InteractionRecord, create_async_db_engine, make_session_factory
from nexus.training.dataset import DatasetBuilder


def _interaction(**overrides) -> InteractionRecord:
    payload = {
        "session_id": "s1",
        "user_id": "u1",
        "task_type": "general",
        "privacy_level": "public",
        "prompt_json": json.dumps([{"role": "user", "content": "What is 2 plus 2?"}]),
        "response_text": "Four.",
        "model_id": "mistral:7b",
        "provider_name": "local",
        "tools_used_json": "[]",
        "verification_band": "",
        "verification_score": None,
        "user_feedback": None,
        "created_at": time.time(),
    }
    payload.update(overrides)
    return InteractionRecord(**payload)


async def _setup(tmp_path, records: list[InteractionRecord], *, eval_run: EvalRun | None = None):
    engine = create_async_db_engine(str(tmp_path / "nexus.db"))
    store = EvalStore(engine)
    await store.init()
    if eval_run is not None:
        await store.save_run(eval_run)

    async with make_session_factory(engine)() as db:
        for record in records:
            db.add(record)
        await db.commit()

    builder = DatasetBuilder(
        engine,
        PrivacyClassifier(),
        store,
        datasets_dir="nexus/evaluation/datasets",
        templates_dir=tmp_path / "templates",  # empty unless a test writes one
    )
    return builder


@pytest.mark.asyncio
async def test_high_feedback_source_mines_thumbs_up_records(tmp_path) -> None:
    builder = await _setup(
        tmp_path,
        [_interaction(user_feedback=1), _interaction(user_feedback=-1), _interaction()],
    )

    examples, stats = await builder.build(include_sources=["high_feedback"])

    assert stats.total == 1
    assert stats.by_source == {"high_feedback": 1}
    assert examples[0].messages[-1] == {"role": "assistant", "content": "Four."}


@pytest.mark.asyncio
async def test_verified_high_source_requires_band_and_score(tmp_path) -> None:
    builder = await _setup(
        tmp_path,
        [
            _interaction(verification_band="high", verification_score=0.9),
            _interaction(verification_band="high", verification_score=0.5),  # under threshold
            _interaction(verification_band="medium", verification_score=0.95),  # wrong band
        ],
    )

    _examples, stats = await builder.build(include_sources=["verified_high"])

    assert stats.by_source == {"verified_high": 1}


@pytest.mark.asyncio
async def test_a_record_is_not_mined_twice_by_both_interaction_sources(tmp_path) -> None:
    builder = await _setup(
        tmp_path,
        [_interaction(user_feedback=1, verification_band="high", verification_score=0.95)],
    )

    _examples, stats = await builder.build(
        include_sources=["high_feedback", "verified_high"]
    )

    assert stats.total == 1
    assert stats.by_source == {"high_feedback": 1}


@pytest.mark.asyncio
async def test_eval_failure_source_pairs_the_case_with_its_expected_answer(tmp_path) -> None:
    # A real, committed case. Pinned rather than picked arbitrarily: some
    # cases carry deliberate PII (the privacy-routing ones) and would be
    # correctly dropped at export, which would make this test about
    # something else entirely.
    failing_case_id = "routing-003"

    run = EvalRun(
        run_id="run-1",
        started_at=1.0,
        finished_at=2.0,
        suites=[
            SuiteResult.from_outcomes(
                "routing",
                [
                    CaseOutcome(
                        case_id=failing_case_id, passed=False, score=0.0, actual={},
                        detail="wrong", latency_seconds=0.0, cost_usd=0.0,
                    ),
                    CaseOutcome(
                        case_id="passing-case", passed=True, score=1.0, actual={},
                        detail="ok", latency_seconds=0.0, cost_usd=0.0,
                    ),
                ],
            )
        ],
        config_snapshot={},
        git_sha=None,
    )
    builder = await _setup(tmp_path, [], eval_run=run)

    examples, stats = await builder.build(include_sources=["eval_failure"])

    # Only the FAILED case becomes a training example — a passing case has
    # nothing to teach.
    assert stats.total == 1
    assert examples[0].source == "eval_failure"
    assert examples[0].messages[0]["role"] == "user"
    assert examples[0].messages[1]["role"] == "assistant"
    assert examples[0].messages[1]["content"]


@pytest.mark.asyncio
async def test_synthetic_source_reads_committed_templates(tmp_path) -> None:
    engine = create_async_db_engine(str(tmp_path / "nexus.db"))
    store = EvalStore(engine)
    await store.init()
    builder = DatasetBuilder(engine, PrivacyClassifier(), store)

    _examples, stats = await builder.build(include_sources=["synthetic"])

    assert stats.total >= 50  # five template files, 10-30 examples each
    assert set(stats.by_source) == {"synthetic"}
    assert "health_safe" in stats.by_task_type


@pytest.mark.asyncio
async def test_private_records_are_excluded_even_when_their_stored_label_says_public(
    tmp_path,
) -> None:
    """THE critical one. privacy_level is re-derived at export rather than
    trusted, so a record written before the classifier learned a pattern —
    or mislabelled outright — still cannot leave the machine."""
    builder = await _setup(
        tmp_path,
        [
            _interaction(
                user_feedback=1,
                privacy_level="public",  # the STORED label is wrong
                prompt_json=json.dumps(
                    [{"role": "user", "content": "Email me at alice@example.com"}]
                ),
            ),
            _interaction(user_feedback=1),  # genuinely public
        ],
    )

    examples, stats = await builder.build(include_sources=["high_feedback"])

    assert stats.total == 1
    assert stats.excluded_private == 1
    assert "alice@example.com" not in json.dumps([e.messages for e in examples])


@pytest.mark.asyncio
async def test_private_content_in_the_response_is_also_caught(tmp_path) -> None:
    builder = await _setup(
        tmp_path,
        [_interaction(user_feedback=1, privacy_level="public", response_text="Your SSN is 123-45-6789.")],
    )

    _examples, stats = await builder.build(include_sources=["high_feedback"])

    assert stats.total == 0
    assert stats.excluded_private == 1


@pytest.mark.asyncio
async def test_exclusions_can_be_widened_to_sensitive(tmp_path) -> None:
    builder = await _setup(
        tmp_path,
        [
            _interaction(
                user_feedback=1,
                prompt_json=json.dumps(
                    [{"role": "user", "content": "I have been feeling anxious lately."}]
                ),
            )
        ],
    )

    _examples, stats = await builder.build(
        include_sources=["high_feedback"], exclude_privacy_levels=("private", "sensitive")
    )

    assert stats.total == 0
    assert stats.excluded_private == 1


@pytest.mark.asyncio
async def test_max_examples_truncates(tmp_path) -> None:
    builder = await _setup(tmp_path, [_interaction(user_feedback=1) for _ in range(5)])

    examples, stats = await builder.build(include_sources=["high_feedback"], max_examples=2)

    assert len(examples) == 2
    assert stats.total == 2


@pytest.mark.asyncio
async def test_write_jsonl_round_trips(tmp_path) -> None:
    builder = await _setup(tmp_path, [_interaction(user_feedback=1)])
    examples, _stats = await builder.build(include_sources=["high_feedback"])

    target = tmp_path / "out" / "dataset.jsonl"
    builder.write_jsonl(examples, target)

    lines = target.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["source"] == "high_feedback"
    assert parsed["messages"][-1]["content"] == "Four."


@pytest.mark.asyncio
async def test_stats_report_token_distribution(tmp_path) -> None:
    builder = await _setup(tmp_path, [_interaction(user_feedback=1) for _ in range(3)])

    _examples, stats = await builder.build(include_sources=["high_feedback"])

    assert stats.mean_tokens > 0
    assert stats.p95_tokens > 0
