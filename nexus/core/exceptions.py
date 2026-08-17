from __future__ import annotations


class NexusError(Exception):
    """Base class for all NEXUS-raised errors."""


class ConfigError(NexusError):
    """Configuration failed to load or validate."""


class ProviderError(NexusError):
    """Base class for AIProvider failures."""


class ProviderUnavailableError(ProviderError):
    """The provider's backend could not be reached (e.g. Ollama is down)."""


class ModelNotFoundError(ProviderError):
    """The requested model is not available on the provider's backend."""


class ContextLengthExceededError(ProviderError):
    """The request would exceed the target model's context window."""


class MemoryStoreError(NexusError):
    """Base class for MemoryStore failures."""


class SessionNotFoundError(MemoryStoreError):
    """No session exists for the given session_id (or it has expired)."""


class RagError(NexusError):
    """Base class for document ingestion / retrieval failures."""


class UnsupportedDocumentType(RagError):
    """No parser is registered for the document's source_type."""


class ToolError(NexusError):
    """Base class for tool-registry/execution failures."""


class ToolNotFoundError(ToolError):
    """No tool is registered under the requested name."""


class AgentError(NexusError):
    """Base class for agent-framework failures."""


class AgentNotFoundError(AgentError):
    """No agent is registered (or enabled) under the requested name."""


class PersonalStateError(NexusError):
    """Base class for personal-intelligence (Phase 7) failures."""


class InvalidSignalError(PersonalStateError):
    """A recorded signal's dimension or value failed validation."""
