"""Post-generation guard against ungrounded narrative claims.

The system prompt forbids inventing match chronology and events, and a low
temperature makes the model comply most of the time -- but "most of the
time" is not a guarantee, and a live qwen2.5:3b still produces "in the first
half..." on some runs from a 15-second clip that has no halves in it.

Prompt instructions are a request. This module is the enforcement: it reads
the generated narrative back and checks it against what the context actually
supported. That keeps the no-fabrication property a property of the SYSTEM
rather than of whichever model happens to be installed.

Numbers are already covered by the grounding tests; this catches the
qualitative failure mode they miss -- prose that quotes no figure at all but
asserts a timeline or an event that was never measured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from nexus.sports.timeline import TacticalTimeline

# Phrases asserting a position within the match. Only fabrications when the
# timeline did not supply chronology -- with a real timeline, talking about
# phases of play is exactly what we want.
_CHRONOLOGY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bfirst half\b", "first half"),
    (r"\bsecond half\b", "second half"),
    (r"\bat half[- ]time\b", "half-time"),
    (r"\bafter the break\b", "after the break"),
    (r"\bas the match (?:progressed|wore on|developed)\b", "match progression"),
    (r"\b(?:later|earlier) (?:in|on) (?:in )?the match\b", "relative match time"),
    (r"\bopening (?:minutes|spell|stages)\b", "opening spell"),
    (r"\bclosing (?:minutes|spell|stages)\b", "closing spell"),
    (r"\bby full[- ]time\b", "full time"),
    (r"\bthroughout the match,? (?:several|we|the team) (?:adjustments|made|shifted)",
     "in-match adjustment narrative"),
)

# Events that can only be known from data this pipeline does not produce.
_EVENT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bscored\b", "goal scored"),
    (r"\bconceded\b", "goal conceded"),
    (r"\bbreakthroughs?\b", "breakthrough"),
    (r"\bsubstitutions?\b", "substitution"),
    (r"\b(?:injury|injuries|injured)\b", "injury"),
    (r"\bsuspensions?\b", "suspension"),
    (r"\bsent off\b", "dismissal"),
    (r"\bpenalt(?:y|ies)\b", "penalty"),
)


@dataclass
class NarrativeAudit:
    """What the guard found in one generated narrative."""

    chronology_claims: list[str] = field(default_factory=list)
    event_claims: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.chronology_claims and not self.event_claims

    @property
    def warnings(self) -> list[str]:
        """Coach-facing lines naming what was asserted without support."""
        out = [
            f"narrative asserted match chronology ({claim}) that no measured "
            "data supports"
            for claim in self.chronology_claims
        ]
        out += [
            f"narrative asserted a match event ({claim}) that no measured "
            "data supports"
            for claim in self.event_claims
        ]
        return out

    def correction_instruction(self) -> str:
        """A specific retry instruction, naming what to remove."""
        parts: list[str] = []
        if self.chronology_claims:
            parts.append(
                "references to when things happened within the match ("
                + ", ".join(sorted(set(self.chronology_claims)))
                + ")"
            )
        if self.event_claims:
            parts.append(
                "references to match events ("
                + ", ".join(sorted(set(self.event_claims)))
                + ")"
            )
        return (
            "Your previous reply contained "
            + " and ".join(parts)
            + ". None of that appears in the data you were given, so it was "
            "invented. Rewrite the report using ONLY the measured values "
            "supplied. Describe what the aggregates say about the team; do "
            "not narrate a sequence of events, and write the adjustments as "
            "instructions for next time, not as things already done."
        )


def audit_narrative(
    narrative: str, *, timeline: TacticalTimeline | None = None
) -> NarrativeAudit:
    """Flag chronology/event claims the context could not support.

    When `timeline` supplies phase data, chronology language is legitimate
    and is not flagged -- the guard exists to catch claims with no backing,
    not to ban time from a report that measured it.
    """
    audit = NarrativeAudit()
    lowered = narrative.lower()

    timeline_has_phases = timeline is not None and timeline.phases.is_available
    if not timeline_has_phases:
        for pattern, label in _CHRONOLOGY_PATTERNS:
            if re.search(pattern, lowered):
                audit.chronology_claims.append(label)

    for pattern, label in _EVENT_PATTERNS:
        if re.search(pattern, lowered):
            audit.event_claims.append(label)

    return audit
