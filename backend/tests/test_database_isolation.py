"""
The test suite must never be bound to a real database.
"""

from __future__ import annotations

import os
import tempfile

from backend.database.session import DATABASE_URL


def test_the_orm_is_bound_to_a_throwaway_database():
    assert DATABASE_URL.startswith("sqlite:///"), (
        f"expected a throwaway SQLite database, got {DATABASE_URL}"
    )

    path = os.path.normcase(os.path.abspath(DATABASE_URL[len("sqlite:///"):]))
    temp_root = os.path.normcase(os.path.abspath(tempfile.gettempdir()))

    assert path.startswith(temp_root), (
        "backend/tests is bound to a database outside the temp directory, and "
        "the fixtures here DELETE from it unconditionally.\n"
        f"  bound to: {path}\n"
        f"  expected: something under {temp_root}\n"
        "Fix the import order (see the repo-root conftest.py), never this test."
    )


def test_the_real_project_database_is_not_the_target():
    """The specific file that was destroyed, named explicitly."""
    path = os.path.normcase(os.path.abspath(DATABASE_URL[len("sqlite:///"):]))

    assert not path.endswith(os.path.normcase("sports_strategy.db")), (
        "the suite is bound to the real project database"
    )
