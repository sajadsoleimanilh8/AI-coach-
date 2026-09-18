"""Shared fixtures for the NEXUS test suite.

This fixture was previously copy-pasted into 15 test modules in five slightly
different variants, some of which forgot to refresh the cached settings after
changing the environment. It is defined once here.

Two modules (test_eval_cli.py, test_phase14_eval_suites.py) deliberately keep a
function-scoped override of the same name, because they need a fresh database
per test rather than per module; a local fixture of the same name takes
precedence over this one.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from nexus.config.settings import get_settings


@pytest.fixture(scope="module")
def _memory_db_env(tmp_path_factory, request) -> Iterator[str]:
    """Point NEXUS's SQLite memory store at a throwaway file for one module.

    The settings object is cached process-wide, so it is refreshed after the
    environment changes (on setup AND teardown); otherwise a module that ran
    earlier would leave its database path in place for the next one.
    """
    label = f"nexus-{request.module.__name__.rsplit('.', 1)[-1]}"
    db_path = str(tmp_path_factory.mktemp(label) / "nexus.db")
    os.environ["NEXUS_MEMORY__DATABASE_PATH"] = db_path
    get_settings(refresh=True)
    yield db_path
    os.environ.pop("NEXUS_MEMORY__DATABASE_PATH", None)
    get_settings(refresh=True)
