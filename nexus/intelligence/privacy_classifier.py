from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from nexus.core.types import TaskType
from nexus.intelligence.task_classifier import _default_signal_map

PrivacyLevel = Literal["public", "sensitive", "private"]

_PRIVATE_CONFIDENCE_BASE = 0.6
_SENSITIVE_CONFIDENCE_BASE = 0.4
_CONFIDENCE_PER_SIGNAL = 0.1
_PUBLIC_CONFIDENCE = 0.5

_DEFAULT_PRIVATE_PATTERNS: list[str] = [
    r"[\w.+-]+@[\w-]+\.[\w.-]+",
    r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b",
    r"\bsk-[A-Za-z0-9]{16,}\b",
    r"\bAKIA[0-9A-Z]{16}\b",
    r"\bghp_[A-Za-z0-9]{30,}\b",
    r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
    r"i (?:was|have been) diagnosed with",
    r"\bmy medication\b",
    r"\bmy prescription\b",
    r"\bmy diagnosis\b",
    r"\bmy bank account\b",
    r"\bmy account number\b",
    r"\brouting number\b",
    r"\biban\b",
]

_DEFAULT_SENSITIVE_PATTERNS: list[str] = [
    r"i(?:'ve| have) been feeling tired",
    r"my sleep has been",
    r"my personal project",
    r"my business plan",
    r"feeling (?:anxious|depressed|stressed)",
]


def _health_and_personal_signals() -> list[str]:
    signal_map = _default_signal_map()
    return signal_map.get(TaskType.HEALTH, []) + signal_map.get(TaskType.PRIVATE_PERSONAL, [])


@dataclass
class PrivacyClassifierConfig:
    extra_private_patterns: list[str] = field(default_factory=list)
    extra_sensitive_patterns: list[str] = field(default_factory=list)


@dataclass
class PrivacyClassification:
    level: PrivacyLevel
    confidence: float
    matched_signals: list[str] = field(default_factory=list)
    reason: str = ""


class PrivacyClassifier:
    """Deterministic pattern-based privacy classifier."""

    def __init__(self, signal_config: PrivacyClassifierConfig | None = None) -> None:
        config = signal_config or PrivacyClassifierConfig()

        private_sources = _DEFAULT_PRIVATE_PATTERNS + list(config.extra_private_patterns)
        sensitive_sources = (
            _DEFAULT_SENSITIVE_PATTERNS
            + _health_and_personal_signals()
            + list(config.extra_sensitive_patterns)
        )

        self._private_patterns = [(raw, re.compile(raw, re.IGNORECASE)) for raw in private_sources]
        self._sensitive_patterns = [
            (raw, re.compile(raw, re.IGNORECASE)) for raw in sensitive_sources
        ]

    def classify(self, query: str) -> PrivacyClassification:
        private_matches = [
            f"private:{raw}" for raw, pattern in self._private_patterns if pattern.search(query)
        ]
        if private_matches:
            confidence = min(1.0, _PRIVATE_CONFIDENCE_BASE + _CONFIDENCE_PER_SIGNAL * len(private_matches))
            return PrivacyClassification(
                level="private",
                confidence=confidence,
                matched_signals=private_matches,
                reason=(
                    f"Matched {len(private_matches)} private-tier signal(s): "
                    f"{', '.join(private_matches)}. Private always overrides sensitive."
                ),
            )

        sensitive_matches = [
            f"sensitive:{raw}" for raw, pattern in self._sensitive_patterns if pattern.search(query)
        ]
        if sensitive_matches:
            confidence = min(
                1.0, _SENSITIVE_CONFIDENCE_BASE + _CONFIDENCE_PER_SIGNAL * len(sensitive_matches)
            )
            return PrivacyClassification(
                level="sensitive",
                confidence=confidence,
                matched_signals=sensitive_matches,
                reason=(
                    f"Matched {len(sensitive_matches)} sensitive-tier signal(s): "
                    f"{', '.join(sensitive_matches)}."
                ),
            )

        return PrivacyClassification(
            level="public",
            confidence=_PUBLIC_CONFIDENCE,
            matched_signals=[],
            reason="No private or sensitive signals matched; defaulting to public.",
        )
