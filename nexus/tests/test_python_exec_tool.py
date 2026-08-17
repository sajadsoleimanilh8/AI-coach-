from __future__ import annotations

import pytest

from nexus.tools.python_exec import PythonExecutionTool


@pytest.mark.asyncio
async def test_success_case_captures_stdout() -> None:
    tool = PythonExecutionTool()

    result = await tool.execute({"code": "print('ok')"})

    assert result.success is True
    assert result.output.strip() == "ok"
    assert result.error is None


@pytest.mark.asyncio
async def test_error_capture_for_raising_code() -> None:
    tool = PythonExecutionTool()

    result = await tool.execute({"code": "raise ValueError('bad input')"})

    assert result.success is False
    assert "bad input" in result.error


@pytest.mark.asyncio
async def test_timeout_case() -> None:
    tool = PythonExecutionTool(timeout_seconds=1)

    result = await tool.execute({"code": "import time; time.sleep(5)"})

    assert result.success is False
    assert "timed out" in result.error.lower()


@pytest.mark.asyncio
async def test_missing_code_argument_fails_without_running_subprocess() -> None:
    tool = PythonExecutionTool()

    result = await tool.execute({})

    assert result.success is False
    assert "code" in result.error.lower()


@pytest.mark.asyncio
async def test_isolated_mode_does_not_inherit_host_environment_customizations() -> None:
    tool = PythonExecutionTool()

    result = await tool.execute({"code": "import sys; print(sys.flags.isolated)"})

    assert result.success is True
    assert result.output.strip() == "1"
