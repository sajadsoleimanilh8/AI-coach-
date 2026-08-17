from __future__ import annotations

import pytest

from nexus.memory.long_term import LongTermMemoryStore
from nexus.memory.storage import create_async_db_engine


async def _make_store(tmp_path) -> LongTermMemoryStore:
    engine = create_async_db_engine(str(tmp_path / "nexus.db"))
    store = LongTermMemoryStore(engine)
    await store.init()
    return store


@pytest.mark.asyncio
async def test_set_then_get_fact(tmp_path) -> None:
    store = await _make_store(tmp_path)
    await store.set_fact("u1", "favorite_team", "Arsenal")

    facts = await store.get_facts("u1")

    assert facts == {"favorite_team": "Arsenal"}


@pytest.mark.asyncio
async def test_set_fact_overwrites_existing_key(tmp_path) -> None:
    store = await _make_store(tmp_path)
    await store.set_fact("u1", "favorite_team", "Arsenal")
    await store.set_fact("u1", "favorite_team", "Chelsea")

    facts = await store.get_facts("u1")

    assert facts == {"favorite_team": "Chelsea"}


@pytest.mark.asyncio
async def test_values_can_be_arbitrary_json_serializable_types(tmp_path) -> None:
    store = await _make_store(tmp_path)
    await store.set_fact("u1", "preferences", {"theme": "dark", "notifications": True})
    await store.set_fact("u1", "recent_scores", [1, 2, 3])

    facts = await store.get_facts("u1")

    assert facts["preferences"] == {"theme": "dark", "notifications": True}
    assert facts["recent_scores"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_delete_fact_removes_only_that_key(tmp_path) -> None:
    store = await _make_store(tmp_path)
    await store.set_fact("u1", "a", 1)
    await store.set_fact("u1", "b", 2)

    await store.delete_fact("u1", "a")

    assert await store.get_facts("u1") == {"b": 2}


@pytest.mark.asyncio
async def test_clear_removes_all_facts_for_a_user(tmp_path) -> None:
    store = await _make_store(tmp_path)
    await store.set_fact("u1", "a", 1)
    await store.set_fact("u1", "b", 2)

    await store.clear("u1")

    assert await store.get_facts("u1") == {}


@pytest.mark.asyncio
async def test_facts_are_scoped_by_user_id(tmp_path) -> None:
    store = await _make_store(tmp_path)
    await store.set_fact("u1", "favorite_team", "Arsenal")
    await store.set_fact("u2", "favorite_team", "Chelsea")

    assert await store.get_facts("u1") == {"favorite_team": "Arsenal"}
    assert await store.get_facts("u2") == {"favorite_team": "Chelsea"}


@pytest.mark.asyncio
async def test_get_facts_for_unknown_user_is_empty(tmp_path) -> None:
    store = await _make_store(tmp_path)
    assert await store.get_facts("nobody") == {}


@pytest.mark.asyncio
async def test_delete_fact_for_unknown_key_is_a_no_op(tmp_path) -> None:
    store = await _make_store(tmp_path)
    await store.set_fact("u1", "a", 1)

    await store.delete_fact("u1", "does-not-exist")

    assert await store.get_facts("u1") == {"a": 1}
