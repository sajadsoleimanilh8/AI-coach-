from __future__ import annotations

import pytest

from nexus.core.types import Message
from nexus.memory.storage import create_async_db_engine
from nexus.training.logger import InteractionLogger


async def _make_logger(tmp_path, *, enabled: bool) -> InteractionLogger:
    engine = create_async_db_engine(str(tmp_path / "nexus.db"))
    logger = InteractionLogger(engine, enabled=enabled)
    await logger.init()
    return logger


def _messages() -> list[Message]:
    return [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="What is the capital of France?"),
    ]


async def _record(logger: InteractionLogger, **overrides) -> int | None:
    payload = {
        "session_id": "s1",
        "user_id": "u1",
        "task_type": "general",
        "privacy_level": "public",
        "prompt_messages": _messages(),
        "response_text": "Paris.",
        "model_id": "mistral:7b",
        "provider_name": "local",
    }
    payload.update(overrides)
    return await logger.record(**payload)


@pytest.mark.asyncio
async def test_records_when_enabled(tmp_path) -> None:
    logger = await _make_logger(tmp_path, enabled=True)

    interaction_id = await _record(logger)

    assert interaction_id is not None
    assert await logger.count() == 1


@pytest.mark.asyncio
async def test_records_nothing_when_disabled(tmp_path) -> None:
    logger = await _make_logger(tmp_path, enabled=False)

    interaction_id = await _record(logger)

    assert interaction_id is None
    assert await logger.count() == 0


@pytest.mark.asyncio
async def test_captures_mining_metadata(tmp_path) -> None:
    logger = await _make_logger(tmp_path, enabled=True)

    interaction_id = await _record(
        logger,
        task_type="research",
        privacy_level="sensitive",
        tools_used=["web_search"],
        verification_band="high",
        verification_score=0.91,
    )

    from sqlalchemy import select

    from nexus.memory.storage import InteractionRecord, make_session_factory

    async with make_session_factory(logger._engine)() as db:  # noqa: SLF001
        row = (await db.execute(select(InteractionRecord))).scalar_one()

    assert row.id == interaction_id
    assert row.task_type == "research"
    assert row.privacy_level == "sensitive"
    assert row.verification_band == "high"
    assert row.verification_score == pytest.approx(0.91)
    assert "web_search" in row.tools_used_json
    assert "capital of France" in row.prompt_json
    assert row.response_text == "Paris."
    assert row.user_feedback is None


@pytest.mark.asyncio
async def test_feedback_updates_the_record(tmp_path) -> None:
    logger = await _make_logger(tmp_path, enabled=True)
    interaction_id = await _record(logger)

    assert await logger.set_feedback(interaction_id, 1) is True

    from sqlalchemy import select

    from nexus.memory.storage import InteractionRecord, make_session_factory

    async with make_session_factory(logger._engine)() as db:  # noqa: SLF001
        row = (await db.execute(select(InteractionRecord))).scalar_one()
    assert row.user_feedback == 1


@pytest.mark.asyncio
async def test_feedback_on_unknown_interaction_reports_failure(tmp_path) -> None:
    logger = await _make_logger(tmp_path, enabled=True)

    assert await logger.set_feedback(9999, 1) is False


@pytest.mark.asyncio
async def test_feedback_still_accepted_when_logging_is_disabled(tmp_path) -> None:
    engine = create_async_db_engine(str(tmp_path / "nexus.db"))
    enabled_logger = InteractionLogger(engine, enabled=True)
    await enabled_logger.init()
    interaction_id = await _record(enabled_logger)

    disabled_logger = InteractionLogger(engine, enabled=False)
    assert await disabled_logger.set_feedback(interaction_id, -1) is True


@pytest.mark.asyncio
async def test_purge_expired_removes_only_old_records(tmp_path) -> None:
    import time

    from nexus.memory.storage import InteractionRecord, make_session_factory

    engine = create_async_db_engine(str(tmp_path / "nexus.db"))
    logger = InteractionLogger(engine, enabled=True, retention_days=30)
    await logger.init()

    await _record(logger)
    async with make_session_factory(engine)() as db:
        db.add(
            InteractionRecord(
                session_id="old", user_id="u1", task_type="general", privacy_level="public",
                prompt_json="[]", response_text="stale", model_id="m", provider_name="local",
                created_at=time.time() - 60 * 86400,
            )
        )
        await db.commit()

    assert await logger.purge_expired() == 1
    assert await logger.count() == 1
