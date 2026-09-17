"""The Alembic migrations and the SQLAlchemy models must not drift apart.

The schema previously had two sources of truth -- the models and a hand-written
backend/database/schema.sql -- and they had already diverged: schema.sql
described 7 tables while the models defined far more. The models are now the
single source, Alembic is generated from them, and this test fails the moment a
model change lands without a matching migration.

Everything runs against a temporary SQLite file, never the project database.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect

REPO_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = REPO_ROOT / "alembic.ini"


def _config(db_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(REPO_ROOT / "backend" / "database" / "alembic"))
    cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


@pytest.fixture
def migrated_db(tmp_path, monkeypatch):
    """A database built purely by running the migrations."""
    url = f"sqlite:///{tmp_path / 'migrated.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    command.upgrade(_config(url), "head")
    return url


def test_exactly_one_head_revision():
    """Two heads mean two people generated migrations in parallel; upgrades
    then fail with 'Multiple head revisions are present'."""
    heads = ScriptDirectory.from_config(_config("sqlite://")).get_heads()
    assert len(heads) == 1, f"expected a single head, found {heads}"


def test_migrations_match_the_models(migrated_db):
    """alembic check, inline: after upgrading to head, autogenerate must find
    nothing left to do."""
    from backend.database import models  # noqa: F401  (registers the tables)
    from backend.database.session import Base

    engine = create_engine(migrated_db)
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(connection, opts={"compare_type": True})
            diff = compare_metadata(context, Base.metadata)
    finally:
        engine.dispose()

    assert diff == [], (
        "models and migrations have drifted; run:\n"
        "  alembic revision --autogenerate -m \"<describe the change>\"\n"
        f"pending operations: {diff}"
    )


def test_migrated_schema_has_the_same_tables_as_the_models(migrated_db):
    from backend.database import models  # noqa: F401
    from backend.database.session import Base

    engine = create_engine(migrated_db)
    try:
        tables = set(inspect(engine).get_table_names()) - {"alembic_version"}
    finally:
        engine.dispose()
    assert tables == set(Base.metadata.tables)


def test_the_stale_hand_written_schema_files_are_gone():
    """Regression: these were the second source of truth that drifted."""
    assert not (REPO_ROOT / "backend" / "database" / "schema.sql").exists()
    assert not (REPO_ROOT / "backend" / "database" / "migrations").exists()


def test_alembic_url_comes_from_the_environment(tmp_path, monkeypatch):
    """env.py must never fall back to a URL written in alembic.ini, or a
    migration could be applied to a different database than the app uses."""
    ini = ALEMBIC_INI.read_text(encoding="utf-8")
    assert "\nsqlalchemy.url =\n" in ini or "\nsqlalchemy.url = \n" in ini, \
        "alembic.ini should not hardcode a database URL"
    assert "DATABASE_URL" in (REPO_ROOT / "backend/database/alembic/env.py").read_text(encoding="utf-8")


def test_startup_does_not_create_tables_on_non_sqlite():
    """On Postgres the schema belongs to Alembic; create_all() at startup would
    let the app and the migration history disagree silently."""
    source = (REPO_ROOT / "backend" / "api" / "main.py").read_text(encoding="utf-8")
    assert 'engine.dialect.name == "sqlite"' in source
    assert source.index('engine.dialect.name == "sqlite"') < source.index("Base.metadata.create_all")
