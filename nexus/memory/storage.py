from __future__ import annotations

from pathlib import Path

from sqlalchemy import ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class SessionRecord(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[float] = mapped_column()
    last_active_at: Mapped[float] = mapped_column()

    messages: Mapped[list[MessageRecord]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="MessageRecord.created_at",
    )


class MessageRecord(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[float] = mapped_column()

    session: Mapped[SessionRecord] = relationship(back_populates="messages")


class CostRecord(Base):
    __tablename__ = "cost_records"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(32), index=True)
    provider_name: Mapped[str] = mapped_column(String(64))
    model_id: Mapped[str] = mapped_column(String(128))
    prompt_tokens: Mapped[int] = mapped_column()
    completion_tokens: Mapped[int] = mapped_column()
    cost_usd: Mapped[float] = mapped_column()
    created_at: Mapped[float] = mapped_column()


class LatencyRecord(Base):
    __tablename__ = "latency_records"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(32), index=True)
    provider_name: Mapped[str] = mapped_column(String(64))
    model_id: Mapped[str] = mapped_column(String(128))
    latency_seconds: Mapped[float] = mapped_column()
    created_at: Mapped[float] = mapped_column()


class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # uuid hex
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    source_name: Mapped[str] = mapped_column(String(512))
    source_type: Mapped[str] = mapped_column(String(16))  # pdf|docx|txt|md|csv|code
    chunk_count: Mapped[int] = mapped_column()
    ingested_at: Mapped[float] = mapped_column()


class DocumentChunkRecord(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int] = mapped_column()
    chunk_text: Mapped[str] = mapped_column(Text)
    embedding_json: Mapped[str] = mapped_column(Text)  # JSON-encoded list[float]
    created_at: Mapped[float] = mapped_column()


class LongTermFactRecord(Base):
    __tablename__ = "long_term_facts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    key: Mapped[str] = mapped_column(String(256))
    value_json: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[float] = mapped_column()

    __table_args__ = (UniqueConstraint("user_id", "key", name="uq_user_key"),)


class PersonalSignalRecord(Base):
    """Raw evidence. Append-only — never mutate or delete individual rows
    outside of a full user wipe; PersonalState is always re-derived from
    signals (PersonalStateEngine.get_state()) rather than stored directly,
    so the whole history stays recomputable as weighting logic evolves."""

    __tablename__ = "personal_signals"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    dimension: Mapped[str] = mapped_column(String(64), index=True)
    value: Mapped[float] = mapped_column()
    source: Mapped[str] = mapped_column(String(32))
    note: Mapped[str] = mapped_column(Text, default="")
    recorded_at: Mapped[float] = mapped_column(index=True)


class PersonalProfileRecord(Base):
    __tablename__ = "personal_profiles"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    profile_json: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[float] = mapped_column()


class EvalRunRecord(Base):
    __tablename__ = "eval_runs"

    run_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    started_at: Mapped[float] = mapped_column()
    finished_at: Mapped[float] = mapped_column()
    git_sha: Mapped[str] = mapped_column(String(64), default="")
    config_json: Mapped[str] = mapped_column(Text)
    summary_json: Mapped[str] = mapped_column(Text)
    # Which provider set produced this run. Defaults to "unknown" so rows
    # written before this column existed keep a truthful label rather than
    # being silently assumed comparable — compare_runs() refuses on
    # "unknown" precisely because those rows cannot be trusted either way.
    provider_mode: Mapped[str] = mapped_column(String(16), default="unknown")
    pinned_model_id: Mapped[str] = mapped_column(String(128), default="")


class EvalCaseRecord(Base):
    __tablename__ = "eval_cases"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(32), index=True)
    suite: Mapped[str] = mapped_column(String(64), index=True)
    case_id: Mapped[str] = mapped_column(String(128))
    passed: Mapped[bool] = mapped_column()
    score: Mapped[float] = mapped_column()
    detail: Mapped[str] = mapped_column(Text)
    latency_seconds: Mapped[float] = mapped_column()
    cost_usd: Mapped[float] = mapped_column()


class InteractionRecord(Base):
    """Full request/response record for training-data mining. Separate from
    MessageRecord (conversation replay) because it carries the routing/
    verification metadata that makes a record useful as a training example,
    and has its own consent and retention rules."""

    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(32), index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    task_type: Mapped[str] = mapped_column(String(64), index=True)
    privacy_level: Mapped[str] = mapped_column(String(16), index=True)
    prompt_json: Mapped[str] = mapped_column(Text)
    response_text: Mapped[str] = mapped_column(Text)
    model_id: Mapped[str] = mapped_column(String(128))
    provider_name: Mapped[str] = mapped_column(String(64))
    tools_used_json: Mapped[str] = mapped_column(Text, default="[]")
    verification_band: Mapped[str] = mapped_column(String(16), default="")
    verification_score: Mapped[float | None] = mapped_column(nullable=True)
    user_feedback: Mapped[int | None] = mapped_column(nullable=True)
    created_at: Mapped[float] = mapped_column(index=True)


class ModelCapabilityRecord(Base):
    """One measured capability score per (model, capability, eval run).
    Append-only like PersonalSignalRecord: CapabilityMatrix always
    re-derives an effective score from the rows rather than mutating a
    single "current" value, so changing the blending formula later
    re-interprets the whole measurement history instead of losing it."""

    __tablename__ = "model_capabilities"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(128), index=True)
    capability_key: Mapped[str] = mapped_column(String(64), index=True)
    measured_score: Mapped[float] = mapped_column()
    sample_size: Mapped[int] = mapped_column()
    source_run_id: Mapped[str] = mapped_column(String(32), default="")
    measured_at: Mapped[float] = mapped_column(index=True)


class GraphNodeRecord(Base):
    __tablename__ = "graph_nodes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    doc_id: Mapped[str] = mapped_column(String(32), index=True)
    name: Mapped[str] = mapped_column(String(256), index=True)
    node_type: Mapped[str] = mapped_column(String(64), default="entity")
    created_at: Mapped[float] = mapped_column()


class GraphEdgeRecord(Base):
    __tablename__ = "graph_edges"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    source_node_id: Mapped[int] = mapped_column(index=True)
    target_node_id: Mapped[int] = mapped_column(index=True)
    relation: Mapped[str] = mapped_column(String(128))
    weight: Mapped[float] = mapped_column(default=1.0)
    doc_id: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[float] = mapped_column()


def _sqlite_url(database_path: str) -> str:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{path.as_posix()}"


def create_async_db_engine(database_path: str) -> AsyncEngine:
    return create_async_engine(_sqlite_url(database_path))


# Columns added to already-shipped tables. create_all() only creates
# MISSING TABLES — it will not alter a table that already exists — so a
# database written by an earlier version keeps its old shape and every
# query naming a new column fails. Each entry is (table, column, DDL type
# + default) and is applied only when absent, which makes this idempotent
# and safe to run on a fresh database too.
_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("eval_runs", "provider_mode", "VARCHAR(16) DEFAULT 'unknown'"),
    ("eval_runs", "pinned_model_id", "VARCHAR(128) DEFAULT ''"),
)


def _apply_added_columns(connection) -> None:  # noqa: ANN001 - sync SQLAlchemy Connection
    from sqlalchemy import inspect, text

    inspector = inspect(connection)
    existing_tables = set(inspector.get_table_names())
    for table, column, ddl in _ADDED_COLUMNS:
        if table not in existing_tables:
            continue  # create_all() just built it with the column present
        columns = {c["name"] for c in inspector.get_columns(table)}
        if column in columns:
            continue
        connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))


async def init_db(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_apply_added_columns)


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)
