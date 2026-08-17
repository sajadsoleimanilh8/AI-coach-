from __future__ import annotations

import pytest

from nexus.tools.files_tool import FileSystemTool


@pytest.mark.asyncio
async def test_reads_a_file_within_the_allowed_root(tmp_path) -> None:
    (tmp_path / "notes.txt").write_text("hello world", encoding="utf-8")
    tool = FileSystemTool(allowed_root=str(tmp_path))

    result = await tool.execute({"path": "notes.txt"})

    assert result.success is True
    assert result.output == "hello world"


@pytest.mark.asyncio
async def test_reads_a_file_in_a_subdirectory(tmp_path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.txt").write_text("nested content", encoding="utf-8")
    tool = FileSystemTool(allowed_root=str(tmp_path))

    result = await tool.execute({"path": "sub/nested.txt"})

    assert result.success is True
    assert result.output == "nested content"


@pytest.mark.asyncio
async def test_rejects_relative_path_traversal(tmp_path) -> None:
    secret_dir = tmp_path.parent / "outside_root_secret"
    secret_dir.mkdir(exist_ok=True)
    (secret_dir / "passwd").write_text("root:x:0:0", encoding="utf-8")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = FileSystemTool(allowed_root=str(workspace))

    result = await tool.execute({"path": f"../{secret_dir.name}/passwd"})

    assert result.success is False
    assert "escapes" in result.error.lower()


@pytest.mark.asyncio
async def test_rejects_absolute_path_outside_root(tmp_path) -> None:
    outside_file = tmp_path.parent / "definitely_outside.txt"
    outside_file.write_text("secret", encoding="utf-8")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = FileSystemTool(allowed_root=str(workspace))

    result = await tool.execute({"path": str(outside_file)})

    assert result.success is False
    assert "escapes" in result.error.lower()


@pytest.mark.asyncio
async def test_missing_file_returns_a_clean_error(tmp_path) -> None:
    tool = FileSystemTool(allowed_root=str(tmp_path))

    result = await tool.execute({"path": "does-not-exist.txt"})

    assert result.success is False
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_output_is_capped_at_max_chars(tmp_path) -> None:
    (tmp_path / "big.txt").write_text("x" * 1000, encoding="utf-8")
    tool = FileSystemTool(allowed_root=str(tmp_path), max_chars=100)

    result = await tool.execute({"path": "big.txt"})

    assert result.success is True
    assert len(result.output) == 100


@pytest.mark.asyncio
async def test_missing_path_argument_fails_cleanly() -> None:
    tool = FileSystemTool(allowed_root=".")

    result = await tool.execute({})

    assert result.success is False
    assert "path" in result.error.lower()


@pytest.mark.asyncio
async def test_allowed_root_is_created_if_missing(tmp_path) -> None:
    workspace = tmp_path / "does" / "not" / "exist" / "yet"
    assert not workspace.exists()

    FileSystemTool(allowed_root=str(workspace))

    assert workspace.exists()
