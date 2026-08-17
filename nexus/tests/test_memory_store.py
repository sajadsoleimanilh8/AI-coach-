from __future__ import annotations

import time

import pytest

from nexus.core.exceptions import SessionNotFoundError
from nexus.core.types import Message
from nexus.memory.short_term import ShortTermMemoryStore
from nexus.memory.storage import create_async_db_engine


async def _make_store(tmp_path, **kwargs) -> ShortTermMemoryStore:
    engine = create_async_db_engine(str(tmp_path / "nexus.db"))
    store = ShortTermMemoryStore(engine, **kwargs)
    await store.init()
    return store


@pytest.mark.asyncio
async def test_write_then_read_round_trip(tmp_path) -> None:
    store = await _make_store(tmp_path)
    session_id = await store.create_session()

    await store.add_message(session_id, Message(role="user", content="hello"))
    await store.add_message(session_id, Message(role="assistant", content="hi back"))

    history = await store.get_history(session_id)

    assert [(m.role, m.content) for m in history] == [
        ("user", "hello"),
        ("assistant", "hi back"),
    ]


@pytest.mark.asyncio
async def test_sliding_window_keeps_only_most_recent_messages(tmp_path) -> None:
    store = await _make_store(tmp_path, window_size=2)
    session_id = await store.create_session()

    await store.add_message(session_id, Message(role="user", content="one"))
    await store.add_message(session_id, Message(role="assistant", content="two"))
    await store.add_message(session_id, Message(role="user", content="three"))

    history = await store.get_history(session_id)

    assert [m.content for m in history] == ["two", "three"]


@pytest.mark.asyncio
async def test_get_history_for_unknown_session_raises(tmp_path) -> None:
    store = await _make_store(tmp_path)

    with pytest.raises(SessionNotFoundError):
        await store.get_history("does-not-exist")


@pytest.mark.asyncio
async def test_clear_session_removes_history(tmp_path) -> None:
    store = await _make_store(tmp_path)
    session_id = await store.create_session()
    await store.add_message(session_id, Message(role="user", content="hi"))

    await store.clear_session(session_id)

    with pytest.raises(SessionNotFoundError):
        await store.get_history(session_id)


@pytest.mark.asyncio
async def test_expired_sessions_are_removed(tmp_path) -> None:
    store = await _make_store(tmp_path, ttl_seconds=0.01)
    session_id = await store.create_session()
    await store.add_message(session_id, Message(role="user", content="hi"))

    time.sleep(0.05)
    removed = await store.expire_sessions()

    assert removed == 1
    with pytest.raises(SessionNotFoundError):
        await store.get_history(session_id)
