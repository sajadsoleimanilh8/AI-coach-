from __future__ import annotations

import pytest

from nexus.core.cost_tracker import CostTracker
from nexus.core.types import ModelInfo, Usage
from nexus.memory.storage import create_async_db_engine
from nexus.tools.database_tool import DatabaseTool


async def _make_tool(tmp_path) -> DatabaseTool:
    engine = create_async_db_engine(str(tmp_path / "nexus.db"))
    tracker = CostTracker(engine)
    await tracker.init()
    await tracker.record(
        session_id="s1",
        provider_name="openai",
        model_id="gpt-4o-mini",
        usage=Usage(prompt_tokens=1000, completion_tokens=500),
        model_info=ModelInfo(
            id="gpt-4o-mini",
            provider="openai",
            cost_per_1k_input_tokens=0.001,
            cost_per_1k_output_tokens=0.001,
        ),
    )
    return DatabaseTool(engine)


@pytest.mark.asyncio
async def test_select_query_is_allowed(tmp_path) -> None:
    tool = await _make_tool(tmp_path)

    result = await tool.execute({"query": "SELECT session_id, cost_usd FROM cost_records"})

    assert result.success is True
    assert "s1" in result.output


@pytest.mark.asyncio
async def test_select_is_case_insensitive(tmp_path) -> None:
    tool = await _make_tool(tmp_path)

    result = await tool.execute({"query": "select cost_usd from cost_records"})

    assert result.success is True


@pytest.mark.asyncio
async def test_trailing_semicolon_is_tolerated(tmp_path) -> None:
    tool = await _make_tool(tmp_path)

    result = await tool.execute({"query": "SELECT cost_usd FROM cost_records;"})

    assert result.success is True


@pytest.mark.asyncio
async def test_insert_is_rejected(tmp_path) -> None:
    tool = await _make_tool(tmp_path)

    result = await tool.execute(
        {"query": "INSERT INTO cost_records (session_id) VALUES ('hacked')"}
    )

    assert result.success is False
    assert "select" in result.error.lower()


@pytest.mark.asyncio
async def test_drop_is_rejected(tmp_path) -> None:
    tool = await _make_tool(tmp_path)

    result = await tool.execute({"query": "DROP TABLE cost_records"})

    assert result.success is False


@pytest.mark.asyncio
async def test_chained_statement_is_rejected(tmp_path) -> None:
    tool = await _make_tool(tmp_path)

    result = await tool.execute(
        {"query": "SELECT cost_usd FROM cost_records; DROP TABLE cost_records"}
    )

    assert result.success is False
    assert "chaining" in result.error.lower()


@pytest.mark.asyncio
async def test_row_limit_is_respected(tmp_path) -> None:
    engine = create_async_db_engine(str(tmp_path / "nexus.db"))
    tracker = CostTracker(engine)
    await tracker.init()
    for i in range(10):
        await tracker.record(
            session_id=f"s{i}",
            provider_name="local",
            model_id="mistral:7b",
            usage=Usage(),
            model_info=None,
        )
    tool = DatabaseTool(engine, max_rows=3)

    result = await tool.execute({"query": "SELECT session_id FROM cost_records"})

    assert result.success is True
    assert len(result.output.splitlines()) == 4


@pytest.mark.asyncio
async def test_missing_query_argument_fails_cleanly(tmp_path) -> None:
    tool = await _make_tool(tmp_path)

    result = await tool.execute({})

    assert result.success is False
    assert "query" in result.error.lower()


@pytest.mark.asyncio
async def test_invalid_sql_is_reported_as_a_failed_result_not_a_crash(tmp_path) -> None:
    tool = await _make_tool(tmp_path)

    result = await tool.execute({"query": "SELECT * FROM table_that_does_not_exist"})

    assert result.success is False
    assert result.error
