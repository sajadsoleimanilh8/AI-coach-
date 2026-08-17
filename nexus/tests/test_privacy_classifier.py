from __future__ import annotations

import pytest

from nexus.intelligence.privacy_classifier import (
    PrivacyClassifier,
    PrivacyClassifierConfig,
    PrivacyLevel,
)

_PRIVATE_CASES: list[str] = [
    "My email is john.doe@example.com, can you reply there?",
    "Call me at 555-123-4567 when you get a chance",
    "My API key is sk-ABCDEFGHIJ1234567890, please don't leak it",
    "My SSN is 123-45-6789, is that safe to share?",
    "Here's my credit card number 4111 1111 1111 1111",
    "I was diagnosed with diabetes last year, does that change your advice?",
]

_SENSITIVE_CASES: list[str] = [
    "I've been feeling tired lately and my sleep has been off",
    "I'm feeling anxious about my personal project deadline",
    "My business plan isn't going well and I've been feeling stressed",
]

_PUBLIC_CASES: list[str] = [
    "What's the capital of France?",
    "Can you help me write a Python function to sort a list?",
    "Explain how photosynthesis works",
]


@pytest.mark.parametrize("query", _PRIVATE_CASES)
def test_hard_pii_and_secrets_classify_as_private(query: str) -> None:
    result = PrivacyClassifier().classify(query)

    assert result.level == "private"
    assert result.matched_signals
    assert result.confidence >= 0.6


@pytest.mark.parametrize("query", _SENSITIVE_CASES)
def test_soft_personal_state_without_hard_pii_classifies_as_sensitive(query: str) -> None:
    result = PrivacyClassifier().classify(query)

    assert result.level == "sensitive"
    assert result.matched_signals


@pytest.mark.parametrize("query", _PUBLIC_CASES)
def test_ordinary_technical_queries_classify_as_public(query: str) -> None:
    result = PrivacyClassifier().classify(query)

    assert result.level == "public"
    assert result.matched_signals == []


def test_private_signal_wins_even_when_sensitive_signals_also_present() -> None:
    query = "I've been feeling tired and my email is john.doe@example.com"

    result = PrivacyClassifier().classify(query)

    assert result.level == "private"
    assert any(signal.startswith("private:") for signal in result.matched_signals)


def test_extra_private_patterns_are_additive_not_replacing() -> None:
    config = PrivacyClassifierConfig(extra_private_patterns=[r"\bmy secret code\b"])
    classifier = PrivacyClassifier(config)

    builtin_result = classifier.classify("Reach me at jane@example.com")
    assert builtin_result.level == "private"

    custom_result = classifier.classify("Here is my secret code, don't share it")
    assert custom_result.level == "private"


def test_public_confidence_and_level_type() -> None:
    result = PrivacyClassifier().classify("Tell me a fun fact about space.")
    level: PrivacyLevel = result.level
    assert level == "public"
    assert 0.0 <= result.confidence <= 1.0
