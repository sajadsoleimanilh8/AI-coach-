"""
Deterministic derivation of coach-facing phrases from a mental-readiness
assessment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nexus.sports.psychology_adapter import PsychologyAssessment

_FACTOR_PHRASES: dict[str, tuple[str, str, str]] = {
    "focus": ("high focus", "reduced focus", "moderate focus"),
    "confidence": ("strong confidence", "low confidence", "moderate confidence"),
    "stress": (
        "low pre-match stress",
        "elevated pre-match stress",
        "moderate pre-match stress",
    ),
    "error_recovery": (
        "good error recovery",
        "slow error recovery",
        "moderate error recovery",
    ),
    "motivation": ("strong motivation", "low motivation", "moderate motivation"),
    "pressure_response": (
        "handles pressure well",
        "sensitive to pressure",
        "neutral pressure response",
    ),
}

_DIMENSION_ORDER = (
    "focus",
    "confidence",
    "stress",
    "error_recovery",
    "motivation",
    "pressure_response",
)

_POSITIVE = "positive"
_NEGATIVE = "negative"
_NEUTRAL = "neutral"


@dataclass(frozen=True)
class PsychologyFindings:
    """What the LLM is allowed to talk about, and nothing else."""

    key_positive_factors: list[str] = field(default_factory=list)
    key_negative_factors: list[str] = field(default_factory=list)
    neutral_factors: list[str] = field(default_factory=list)


def _phrase(dimension: str, label: str) -> str | None:
    """The fixed phrase for one classified dimension, or None if either the
    dimension or the label is unrecognised.
    """
    phrases = _FACTOR_PHRASES.get(dimension)
    if phrases is None:
        return None
    positive, negative, neutral = phrases
    if label == _POSITIVE:
        return positive
    if label == _NEGATIVE:
        return negative
    if label == _NEUTRAL:
        return neutral
    return None


def derive_psychology_factors(assessment: PsychologyAssessment) -> PsychologyFindings:
    """Deterministic phrase derivation over the assessment's factors dict."""
    positives: list[str] = []
    negatives: list[str] = []
    neutrals: list[str] = []

    factors = assessment.factors or {}
    for dimension in _DIMENSION_ORDER:
        label = factors.get(dimension)
        if label is None:
            continue
        phrase = _phrase(dimension, label)
        if phrase is None:
            continue
        if label == _POSITIVE:
            positives.append(phrase)
        elif label == _NEGATIVE:
            negatives.append(phrase)
        else:
            neutrals.append(phrase)

    return PsychologyFindings(
        key_positive_factors=positives,
        key_negative_factors=negatives,
        neutral_factors=neutrals,
    )


def build_prompt_context(
    assessment: PsychologyAssessment, findings: PsychologyFindings
) -> str:
    """The exact text handed to the LLM."""
    positives = (
        "\n".join(f"  - {item}" for item in findings.key_positive_factors) or "  - none"
    )
    negatives = (
        "\n".join(f"  - {item}" for item in findings.key_negative_factors) or "  - none"
    )
    neutrals = (
        "\n".join(f"  - {item}" for item in findings.neutral_factors) or "  - none"
    )

    lines = [
        f"Player: {assessment.player_id}",
        f"Match: {assessment.match_id or 'not linked to a match yet'}",
        f"Mental readiness: {assessment.mental_readiness}/100",
        f"Focus: {assessment.focus}/100",
        f"Confidence: {assessment.confidence}/100",
        f"Stress: {assessment.stress}/100 (higher = more reported stress)",
        f"Pressure risk: {assessment.pressure_risk}",
        f"Mental performance risk: {assessment.mental_performance_risk}",
        "Positive factors:",
        positives,
        "Negative factors:",
        negatives,
        "Neutral factors:",
        neutrals,
        f"Scoring method: {assessment.method} (schema {assessment.schema_version}, "
        f"confidence {assessment.confidence_level})",
        f"Data source: {assessment.data_source}",
        f"Computed at: {assessment.computed_at}",
    ]

    if assessment.historical_context:
        lines.append(
            "Historical observed-performance proxies (from match tracking data, "
            "NOT self-reported and NOT measures of a mental state):"
        )
        lines.extend(f"  - {item}" for item in assessment.historical_context)

    return "\n".join(lines)
