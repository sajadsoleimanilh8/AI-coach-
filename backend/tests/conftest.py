"""
Test-wide database isolation.
"""

import os
import tempfile

_TEST_DB_DIR = tempfile.mkdtemp(prefix="ssc_backend_tests_")
_TEST_DB_PATH = os.path.join(_TEST_DB_DIR, "test_suite.db")

os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB_PATH}"


def pytest_sessionstart(session):
    """Fail loudly rather than delete real data."""
    from backend.database.session import DATABASE_URL

    if DATABASE_URL != f"sqlite:///{_TEST_DB_PATH}":
        raise RuntimeError(
            "backend/tests is bound to an unexpected database and would DELETE "
            f"from it.\n  expected: sqlite:///{_TEST_DB_PATH}\n  actual:   {DATABASE_URL}\n"
            "Something imported backend.database.session before conftest.py set "
            "DATABASE_URL. Refusing to run."
        )
