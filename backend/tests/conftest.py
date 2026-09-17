"""
Test-wide database isolation.

WHY THIS FILE EXISTS
Each test module in this directory used to point DATABASE_URL at its own temp
SQLite file at module scope, before importing backend.database.session, on the
assumption that its own assignment is the one that binds the engine. That
assumption only holds for whichever module pytest imports *first*.

backend/tests/test_calibration_debug.py imports backend.api.calibration_debug
without setting DATABASE_URL at all, which transitively imports
backend.database.session and binds `engine` to the real, default
sports_strategy.db. pytest collects alphabetically, so that happens before
test_psychology_api.py / test_prematch_health_api.py / test_video_upload.py ever
run their own assignment -- by then the module is cached in sys.modules and the
later os.environ writes are dead code.

The consequence was not a failing test, which is why it went unnoticed: the
suite passed while every `clean_tables`-style fixture in this directory ran
DELETE against the developer's real database. A full `pytest backend/tests`
destroyed real Match/Video/ProcessingJob rows, including in-flight pipeline
jobs.

pytest imports conftest.py before collecting any test module, so setting
DATABASE_URL here is the one place that reliably wins. The per-module
assignments are now redundant but harmless -- they are left alone so this file
is the single behavioural change.
"""

import os
import tempfile

_TEST_DB_DIR = tempfile.mkdtemp(prefix="ssc_backend_tests_")
_TEST_DB_PATH = os.path.join(_TEST_DB_DIR, "test_suite.db")

# Must happen before any test module (and therefore any backend.database
# import) is loaded. Unconditional on purpose: honouring a pre-existing
# DATABASE_URL is what would let a developer's real database be pointed at.
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB_PATH}"


def pytest_sessionstart(session):
    """Fail loudly rather than delete real data.

    The fixtures in this directory issue unqualified DELETEs. If the engine is
    ever bound to something other than this session's throwaway file -- a new
    import-order regression, someone re-adding an early real-DB import -- that
    is a data-loss bug, not a test failure, so refuse to run at all.
    """
    from backend.database.session import DATABASE_URL

    if f"sqlite:///{_TEST_DB_PATH}" != DATABASE_URL:
        raise RuntimeError(
            "backend/tests is bound to an unexpected database and would DELETE "
            f"from it.\n  expected: sqlite:///{_TEST_DB_PATH}\n  actual:   {DATABASE_URL}\n"
            "Something imported backend.database.session before conftest.py set "
            "DATABASE_URL. Refusing to run."
        )
