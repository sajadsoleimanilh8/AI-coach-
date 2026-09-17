from __future__ import annotations

from abc import ABC, abstractmethod

from nexus.core.types import Message


class MemoryStore(ABC):
    """Common interface for conversation memory backends.

    Phase 1 ships a single SQLite-backed implementation
    (`nexus.memory.short_term.ShortTermMemoryStore`); this interface exists so
    it can later be swapped for a Postgres- or Redis-backed store, or for the
    long-term/episodic/semantic layers, without touching callers.
    """

    @abstractmethod
    async def create_session(self) -> str:
        """Create a new session and return its session_id."""

    @abstractmethod
    async def add_message(self, session_id: str, message: Message) -> None:
        """Append a message to a session, creating the session if needed."""

    @abstractmethod
    async def get_history(self, session_id: str) -> list[Message]:
        """Return the stored messages for a session, oldest first."""

    @abstractmethod
    async def clear_session(self, session_id: str) -> None:
        """Delete a session and all of its messages."""

    @abstractmethod
    async def expire_sessions(self) -> int:
        """Delete sessions past their TTL. Returns the number removed."""
