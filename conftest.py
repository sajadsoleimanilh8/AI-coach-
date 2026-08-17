"""
Repo-root test isolation. THIS FILE PREVENTS DATA LOSS -- do not delete it.
"""

import os
import tempfile

import pytest

_ROOT_TEST_DB_DIR = tempfile.mkdtemp(prefix="ssc_pytest_")
_ROOT_TEST_DB_PATH = os.path.join(_ROOT_TEST_DB_DIR, "test_suite.db")

os.environ["DATABASE_URL"] = f"sqlite:///{_ROOT_TEST_DB_PATH}"


def _is_throwaway(url: str) -> bool:
    """Whether this URL is one of the temp databases a conftest created."""
    if not url.startswith("sqlite:///"):
        return False
    path = os.path.normcase(os.path.abspath(url[len("sqlite:///"):]))
    temp_root = os.path.normcase(os.path.abspath(tempfile.gettempdir()))
    return path.startswith(temp_root)


@pytest.fixture(scope="session", autouse=True)
def _refuse_to_run_against_a_real_database():
    """Abort the session if the ORM is bound to anything but a throwaway DB."""
    try:
        from backend.database.session import DATABASE_URL
    except Exception:
        return

    if not _is_throwaway(DATABASE_URL):
        pytest.exit(
            "REFUSING TO RUN: the ORM is bound to a database this suite did not "
            "create, and backend/tests fixtures issue unqualified DELETEs "
            "against Match / Video / ProcessingJob / PlayerMetric.\n"
            f"  bound to: {DATABASE_URL}\n"
            f"  expected: a throwaway database under {tempfile.gettempdir()}\n"
            "Something imported backend.database.session before this conftest "
            "could redirect it. Do not 'fix' this by relaxing the check -- find "
            "the early import.",
            returncode=3,
        )
