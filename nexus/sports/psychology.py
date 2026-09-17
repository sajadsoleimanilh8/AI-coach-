"""
Deterministic derivation of coach-facing phrases from a mental-readiness
assessment.

Pure Python, zero LLM calls -- the same role nexus/sports/tactical.py's
derive_findings() plays for tactical metrics, and the same place in the
pipeline:

    psychology_adapter (HTTP)  ->  derive_psychology_factors (this module)
                               ->  CoachAssistant (the only LLM call)

Nothing here computes, adjusts, or invents a number. It reads the `factors`
dict the backend already classified and turns each label into a fixed phrase.
The mapping is a lookup table, so the same assessment always produces the same
phrases, for every player.

WHY THE PHRASE TABLE IS DUPLICATED FROM ai/psychology_ai/model_interface.py
--------------------------------------------------------------------------
It is not really duplication so much as the cost of the boundary this repo
deliberately maintains: nexus/ must not import backend/ or ai/, so it cannot
reach the engine's own vocabulary table and has to carry its own. The two are
kept word-identical on purpose, and the backend's own key_positive_factors /
key_negative_factors lists travel on the same response
(`assessment.raw`) so any drift between them is directly checkable rather than
silent.

This is a mental-READINESS estimate from a self-report. Not emotion detection,
not a psychological or clinical assessment, not a diagnosis -- which is why
every phrase below is performance language and none of it is clinical.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nexus.sports.psychology_adapter import PsychologyAssessment

# Mirrors _FACTOR_PHRASES in ai/psychology_ai/model_interface.py, word for
# word. (positive phrase, negative phrase, neutral phrase) per dimension.
_FACTOR_PHRASES: dict[str, tuple[str, str, str]] = {
    "focus": ("high focus", "reduced focus", "moderate focus"),
    "confidence": ("strong confidence", "low confidence", "moderate confidence"),
    # A "positive" stress factor means LOW reported stress -- the phrases spell
    # that out so the label cannot be misread on its own.
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

# Stable report order, so two runs over the same assessment produce the same
# list order and not just the same set. Dict iteration order would already be
# insertion order in practice, but that would make the guarantee an accident of
# how the JSON happened to be parsed.
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
    """What the LLM is allowed to talk about, and nothing else.

    The three lists are phrases, not numbers. The numbers travel separately on
    the assessment itself and are passed through to the response unchanged.
    """

    key_positive_factors: list[str] = field(default_factory=list)
    key_negative_factors: list[str] = field(default_factory=list)
    # Neutral dimensions are kept rather than dropped so the model can say "the
    # rest were unremarkable" without having to infer which ones those were.
    neutral_factors: list[str] = field(default_factory=list)


def _phrase(dimension: str, label: str) -> str | None:
    """The fixed phrase for one classified dimension, or None if either the
    dimension or the label is unrecognised.

    Unknown input produces no phrase rather than a guessed one: if the backend
    ever adds a seventh dimension, this reports the six it understands instead
    of inventing wording for the new one.
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
    """Deterministic phrase derivation over the assessment's factors dict.

    Pure: no I/O, no clock, no randomness, no LLM call. The same assessment
    always yields the same PsychologyFindings.
    """
    positives: list[str] = []
    negatives: list[str] = []
    neutrals: list[str] = []

    factors = assessment.factors or {}
    # Iterate the known order, then anything unrecognised is simply absent --
    # see _phrase().
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
    """The exact text handed to the LLM.

    Every number it is allowed to mention appears here; anything absent is
    something it must say it does not know rather than estimate. Assembled here
    rather than in coach.py so the deterministic layer owns the full boundary
    of what the model can see.
    """
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
        # Explicitly fenced off as observed-performance proxies so the model
        # cannot present them as measurements of how the player feels.
        lines.append(
            "Historical observed-performance proxies (from match tracking data, "
            "NOT self-reported and NOT measures of a mental state):"
        )
        lines.extend(f"  - {item}" for item in assessment.historical_context)

    return "\n".join(lines)
