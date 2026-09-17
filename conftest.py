"""
Repo-root test isolation. THIS FILE PREVENTS DATA LOSS -- do not delete it.

WHAT WENT WRONG
    backend/tests/*.py contain fixtures that issue UNQUALIFIED deletes:

        session.query(Match).delete()
        session.query(Video).delete()
        session.query(ProcessingJob).delete()
        session.query(PlayerMetric).delete()

    Those are safe only while `backend.database.session.engine` is bound to a
    throwaway database. backend/tests/conftest.py exists to guarantee that, by
    setting DATABASE_URL before anything imports that module.

    It does not guarantee it when the suite is run from the repo root.

    pytest loads "initial" conftest.py files -- the ones whose hooks are
    registered before pytest_sessionstart -- only for the paths named on the
    command line. `pytest` with no arguments names the rootdir, so the only
    initial conftest is THIS one. backend/tests/conftest.py is loaded later,
    during collection of that directory, by which time:

      - something under ai/ has already imported backend.database.session
        transitively (ai/psychology_ai/constants.py,
        ai/simulation_ai/what_if_analysis/engine.py and others import from
        backend.*), so `engine` is already bound to the REAL database;
      - its os.environ["DATABASE_URL"] assignment is therefore dead code, the
        module being cached in sys.modules; and
      - its pytest_sessionstart guard -- written precisely to catch this --
        never runs at all, because sessionstart fired before that conftest
        was registered.

    Measured, by probing the bound URL from inside a backend test:

        pytest                 -> sqlite:///.../backend/database/sports_strategy.db
        pytest backend/tests   -> sqlite:///<temp>/test_suite.db

    So the documented command in VERIFY.md ("pytest backend/tests") was safe
    and the obvious one ("pytest") destroyed the developer's real Match /
    Video / ProcessingJob / PlayerMetric rows, silently, while reporting a
    green suite. It is not a failing test, which is why it went unnoticed.

THE FIX
    Set DATABASE_URL here, at the repo root, where pytest is guaranteed to
    load it before it collects or imports anything else. Every invocation --
    `pytest`, `pytest backend/tests`, `pytest ai nexus`, a single test id --
    now starts with the redirection already in place.

    backend/tests/conftest.py is kept as-is. Its assignment becomes a no-op
    when this file has already run, and it still covers the case of someone
    invoking pytest from inside backend/ with a different rootdir.

    The guard below is the belt to that braces: it runs as a session-scoped
    autouse fixture rather than as a hook, so it cannot be outrun by import
    order the way pytest_sessionstart was.
"""

import os
import tempfile

import pytest

_ROOT_TEST_DB_DIR = tempfile.mkdtemp(prefix="ssc_pytest_")
_ROOT_TEST_DB_PATH = os.path.join(_ROOT_TEST_DB_DIR, "test_suite.db")

# Unconditional. Honouring a pre-existing DATABASE_URL is exactly what would
# let a real database be pointed at -- an env var set for a dev server in the
# same shell must not silently become the test target.
os.environ["DATABASE_URL"] = f"sqlite:///{_ROOT_TEST_DB_PATH}"


def _is_throwaway(url: str) -> bool:
    """Whether this URL is one of the temp databases a conftest created.

    Deliberately a positive test against the known temp locations rather than
    a blocklist of real ones: a database this suite did not create is not
    proven safe just because it is not named sports_strategy.db.
    """
    if not url.startswith("sqlite:///"):
        return False
    path = os.path.normcase(os.path.abspath(url[len("sqlite:///"):]))
    temp_root = os.path.normcase(os.path.abspath(tempfile.gettempdir()))
    return path.startswith(temp_root)


@pytest.fixture(scope="session", autouse=True)
def _refuse_to_run_against_a_real_database():
    """Abort the session if the ORM is bound to anything but a throwaway DB.

    A fixture, not a hook: hooks from directory-level conftests are registered
    after sessionstart and can be skipped entirely, which is the failure this
    file exists to close. A session-scoped autouse fixture runs before the
    first test regardless of where it lives or what imported what.

    Importing backend.database.session here is safe and deliberate -- by this
    point DATABASE_URL is already set above, so this import cannot itself be
    the one that binds the engine to the wrong place.
    """
    try:
        from backend.database.session import DATABASE_URL
    except Exception:
        # No backend/SQLAlchemy in this environment (the light requirements
        # file, or a frontend-only checkout). Nothing can be deleted, so
        # there is nothing to guard.
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
