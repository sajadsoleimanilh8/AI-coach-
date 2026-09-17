from __future__ import annotations

import re
from dataclasses import dataclass, field

from nexus.logging_setup.logger import get_logger

logger = get_logger("health.safety")

MEDICAL_DISCLAIMER = (
    "This reflects patterns detected in your own recorded data only — it is not "
    "medical advice and does not diagnose any condition. For anything concerning, "
    "or before changing any treatment or medication, consult a qualified "
    "healthcare professional."
)

URGENT_REFERRAL_MESSAGE = (
    "What you're describing may be a medical emergency. Please seek immediate "
    "professional medical attention now — contact emergency services or go to "
    "the nearest emergency room. This assistant cannot evaluate or treat this "
    "and will not attempt to."
)

_OUTPUT_BLOCKED_FALLBACK = (
    "The specific claim above could not be shared safely. " + MEDICAL_DISCLAIMER
)

# Compiled once at module scope (never per-call) — text that must
# short-circuit the whole health path rather than be analyzed at all.
RED_FLAG_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("chest_pain", re.compile(r"\bchest\s+pain\b", re.IGNORECASE)),
    (
        "breathing_difficulty",
        re.compile(
            r"\b(shortness of breath|can'?t breathe|cannot breathe|"
            r"difficulty breathing|struggling to breathe|gasping for air)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "fainting",
        re.compile(
            r"\b(fainted|fainting|passed out|loss of consciousness|"
            r"lost consciousness|blacked out)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "self_harm",
        re.compile(
            r"\b(suicidal|suicide|kill myself|want to die|end my life|"
            r"self[- ]harm|hurt myself|cutting myself)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "severe_sudden_pain",
        re.compile(
            r"\b(worst pain of my life|sudden (severe|extreme|excruciating) pain|"
            r"excruciating pain|unbearable pain)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "uncontrolled_bleeding",
        re.compile(
            r"\b(uncontrolled bleeding|won'?t stop bleeding|"
            r"bleeding (heavily|a lot) and (it )?(won'?t|can'?t) stop|severe bleeding)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "stroke_symptoms",
        re.compile(
            r"\b(facial droop|face is drooping|slurred speech|"
            r"one[- ]sided weakness|numbness on one side|"
            r"can'?t move (one side|my arm|my leg))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "pregnancy_complication",
        re.compile(
            r"\b(pregnant and (bleeding|in severe pain)|"
            r"severe pregnancy (pain|bleeding)|pregnancy complication)\b",
            re.IGNORECASE,
        ),
    ),
]

# Compiled once at module scope — output the health path must never ship,
# checked on every model-generated health response.
PROHIBITED_OUTPUT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "diagnostic_claim",
        re.compile(
            r"\b(you have|this is|that'?s|you'?re suffering from|"
            r"you are suffering from|diagnos(is|ed) (of|with))\s+"
            r"(?:[a-z\-]+\s+){0,5}?"
            r"(disease|disorder|syndrome|condition|diabetes|anxiety disorder|"
            r"depression|cancer|infection|arthritis|asthma)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "medication_instruction",
        re.compile(
            r"\b(take|stop taking|increase your dose of|decrease your dose of|"
            r"double your dose of|start taking)\s+"
            r"(\d+\s*(mg|mcg|milligrams|micrograms)\s*(of\s+)?)?"
            r"(\d+\s+)?"
            r"[a-z][a-z\-]{2,30}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "dosage_with_drug",
        re.compile(
            r"\b\d+\s*(mg|mcg|milligrams|micrograms|ml|milliliters)\b"
            r"[^.]{0,25}\b(ibuprofen|acetaminophen|tylenol|advil|aspirin|"
            r"paracetamol|medication|pill|tablet|dose)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "discourages_care",
        re.compile(
            r"\b(no need to|you don'?t need to|there'?s no need to) "
            r"(see|consult|visit) a (doctor|physician|professional|specialist)\b",
            re.IGNORECASE,
        ),
    ),
]


@dataclass
class SafetyVerdict:
    allowed: bool
    rewritten_text: str | None
    triggered_rules: list[str] = field(default_factory=list)
    requires_professional_referral: bool = False


def check_input(text: str) -> SafetyVerdict:
    """Red-flag input short-circuits the whole health path: never analyzed,
    never reassured, never delayed — escalate immediately to an urgent-care
    referral."""
    triggered = [name for name, pattern in RED_FLAG_PATTERNS if pattern.search(text)]
    if triggered:
        logger.warning("health input matched red-flag rule(s): %s", ", ".join(triggered))
        return SafetyVerdict(
            allowed=False,
            rewritten_text=URGENT_REFERRAL_MESSAGE,
            triggered_rules=triggered,
            requires_professional_referral=True,
        )
    return SafetyVerdict(allowed=True, rewritten_text=None)


def check_output(text: str, *, patterns_summary: str | None = None) -> SafetyVerdict:
    """Post-filter on any LLM-generated health text. `patterns_summary`, if
    given, lets the caller (HealthAgent, which has the structured
    HealthAnalysis on hand) supply an already-safe restatement of the
    detected patterns for the rewrite; without it, a generic safe fallback
    is used — check_output must remain callable on raw text alone."""
    triggered = [name for name, pattern in PROHIBITED_OUTPUT_PATTERNS if pattern.search(text)]
    if triggered:
        logger.warning(
            "health output blocked, matched prohibited rule(s): %s", ", ".join(triggered)
        )
        base = patterns_summary or _OUTPUT_BLOCKED_FALLBACK
        return SafetyVerdict(
            allowed=False,
            rewritten_text=f"{base}\n\n{MEDICAL_DISCLAIMER}",
            triggered_rules=triggered,
            requires_professional_referral=True,
        )
    return SafetyVerdict(allowed=True, rewritten_text=None)
