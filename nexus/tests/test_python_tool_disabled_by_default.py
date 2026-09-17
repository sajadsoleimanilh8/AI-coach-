"""The unsandboxed python tool must stay off unless explicitly opted in.

PythonExecutionTool runs model-supplied source with no sandbox (see the
docstring there). Combined with an open /api/chat that was remote code
execution for anyone who could reach the port. These tests pin the default so
it cannot drift back on by an edit to nexus.yaml.
"""

from __future__ import annotations

import pytest

from nexus.config.settings import get_settings


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("NEXUS_ENABLE_PYTHON_TOOL", raising=False)


def test_python_tool_absent_from_default_enabled_list(clean_env):
    settings = get_settings(refresh=True)
    assert "python" not in settings.tools.enabled


def test_other_tools_are_still_enabled_by_default(clean_env):
    """Turning python off must not have turned the safe tools off with it."""
    settings = get_settings(refresh=True)
    assert {"web_search", "files", "database"} <= set(settings.tools.enabled)


@pytest.mark.parametrize("flag", ["1", "true", "TRUE", "yes", "on"])
def test_env_flag_opts_back_in(monkeypatch, flag):
    monkeypatch.setenv("NEXUS_ENABLE_PYTHON_TOOL", flag)
    settings = get_settings(refresh=True)
    assert "python" in settings.tools.enabled


@pytest.mark.parametrize("flag", ["0", "false", "no", "off", ""])
def test_falsey_env_flag_does_not_opt_in(monkeypatch, flag):
    monkeypatch.setenv("NEXUS_ENABLE_PYTHON_TOOL", flag)
    settings = get_settings(refresh=True)
    assert "python" not in settings.tools.enabled


def test_opt_in_does_not_duplicate_when_already_listed(monkeypatch):
    """The override is additive; it must not append a second entry if a config
    already named the tool."""
    monkeypatch.setenv("NEXUS_ENABLE_PYTHON_TOOL", "1")
    settings = get_settings(refresh=True)
    assert settings.tools.enabled.count("python") == 1
