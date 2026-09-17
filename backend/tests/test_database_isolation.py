"""
The test suite must never be bound to a real database.

WHY THIS TEST EXISTS. The fixtures in this directory issue unqualified
deletes -- `session.query(Match).delete()` and friends. Running `pytest` from
the repo root used to bind the ORM to backend/database/sports_strategy.db,
so the documented-adjacent command destroyed real Match / Video /
ProcessingJob / PlayerMetric rows while reporting a green suite. It was not a
failing test, which is exactly why it survived: nothing asserted the one
precondition that made every other test in this directory safe.

Measured before the fix, by probing the bound URL from inside a backend test:

    pytest                 -> sqlite:///.../backend/database/sports_strategy.db
    pytest backend/tests   -> sqlite:///<temp>/test_suite.db

See the repo-root conftest.py for the full mechanism (initial conftests are
loaded before pytest_sessionstart; directory conftests are not) and the fix.

This test is cheap, has no fixtures of its own, and fails loudly the moment
that redirection breaks again.
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
    """The specific file that was destroyed, named explicitly.

    A path check is the general rule; this is the one case that actually
    happened, pinned by name so a future refactor of the rule above cannot
    quietly stop covering it.
    """
    path = os.path.normcase(os.path.abspath(DATABASE_URL[len("sqlite:///"):]))

    assert not path.endswith(os.path.normcase("sports_strategy.db")), (
        "the suite is bound to the real project database"
    )
