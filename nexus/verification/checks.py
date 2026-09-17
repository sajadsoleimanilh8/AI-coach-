from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from nexus.core.vector_store import RetrievedChunk
from nexus.verification.types import CheckResult, CheckStatus

_DEFAULT_WEIGHT = 1.0


@dataclass
class CheckContext:
    text: str
    evidence: list[RetrievedChunk] | None = None


# ---------------------------------------------------------------------- #
# check_arithmetic
# ---------------------------------------------------------------------- #

_RELATIVE_TOLERANCE = 0.01  # 1% — absorbs the model's own rounding, not a real error

_ARITHMETIC_PATTERNS: list[tuple[re.Pattern[str], Callable[[float, float], float]]] = [
    (re.compile(r"(-?\d+(?:\.\d+)?)\s*\+\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)"), lambda a, b: a + b),
    (re.compile(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)"), lambda a, b: a - b),
    (
        re.compile(r"(-?\d+(?:\.\d+)?)\s*[*x×]\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE),
        lambda a, b: a * b,
    ),
    (
        re.compile(
            r"(-?\d+(?:\.\d+)?)\s*(?:times|multiplied by)\s*(-?\d+(?:\.\d+)?)\s*(?:is|equals|=)\s*(-?\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
        lambda a, b: a * b,
    ),
    (
        re.compile(
            r"(-?\d+(?:\.\d+)?)\s*%\s*of\s*(-?\d+(?:\.\d+)?)\s*(?:is|equals|=)\s*(-?\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
        lambda a, b: (a / 100.0) * b,
    ),
    (
        re.compile(
            r"(-?\d+(?:\.\d+)?)\s*(?:/|÷|divided by)\s*(-?\d+(?:\.\d+)?)\s*(?:is|equals|=)\s*(-?\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
        lambda a, b: a / b,
    ),
]


def check_arithmetic(context: CheckContext) -> CheckResult:
    matched_text: list[str] = []
    mismatches: list[str] = []

    for pattern, op in _ARITHMETIC_PATTERNS:
        for m in pattern.finditer(context.text):
            a, b, claimed = float(m.group(1)), float(m.group(2)), float(m.group(3))
            try:
                actual = op(a, b)
            except ZeroDivisionError:
                continue
            matched_text.append(m.group(0))
            tolerance = max(abs(actual) * _RELATIVE_TOLERANCE, 0.01)
            if abs(actual - claimed) > tolerance:
                mismatches.append(f"{m.group(0)!r} (computed {actual:g}, claimed {claimed:g})")

    if not matched_text:
        return CheckResult(
            name="arithmetic", status=CheckStatus.INCONCLUSIVE, weight=_DEFAULT_WEIGHT,
            detail="No arithmetic assertions found in the text.",
        )
    if mismatches:
        return CheckResult(
            name="arithmetic", status=CheckStatus.FAIL, weight=_DEFAULT_WEIGHT,
            detail=f"{len(mismatches)} arithmetic mismatch(es) found.", evidence=mismatches,
        )
    return CheckResult(
        name="arithmetic", status=CheckStatus.PASS, weight=_DEFAULT_WEIGHT,
        detail=f"All {len(matched_text)} arithmetic assertion(s) check out.", evidence=matched_text,
    )


# ---------------------------------------------------------------------- #
# check_citation_support
# ---------------------------------------------------------------------- #

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_MIN_SENTENCE_WORDS = 6
_OVERLAP_FAIL_THRESHOLD = 0.05


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def check_citation_support(context: CheckContext) -> CheckResult:
    if not context.evidence:
        return CheckResult(
            name="citation_support", status=CheckStatus.INCONCLUSIVE, weight=_DEFAULT_WEIGHT,
            detail="No retrieved evidence chunks were provided to check against.",
        )

    evidence_tokens: set[str] = set()
    for chunk in context.evidence:
        evidence_tokens |= _tokenize(chunk.chunk_text)

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(context.text) if s.strip()]
    substantive = [s for s in sentences if len(s.split()) >= _MIN_SENTENCE_WORDS]
    if not substantive:
        return CheckResult(
            name="citation_support", status=CheckStatus.INCONCLUSIVE, weight=_DEFAULT_WEIGHT,
            detail="No substantive sentences to check against evidence.",
        )

    unsupported: list[str] = []
    for sentence in substantive:
        sentence_tokens = _tokenize(sentence)
        if not sentence_tokens:
            continue
        overlap = len(sentence_tokens & evidence_tokens) / len(sentence_tokens)
        if overlap < _OVERLAP_FAIL_THRESHOLD:
            unsupported.append(sentence)

    if unsupported:
        return CheckResult(
            name="citation_support", status=CheckStatus.FAIL, weight=_DEFAULT_WEIGHT,
            detail=f"{len(unsupported)} substantive sentence(s) have near-zero lexical overlap with "
                   f"retrieved evidence — a hallucination signal.",
            evidence=unsupported,
        )
    return CheckResult(
        name="citation_support", status=CheckStatus.PASS, weight=_DEFAULT_WEIGHT,
        detail="All substantive sentences have lexical support in the retrieved evidence.",
    )


# ---------------------------------------------------------------------- #
# check_internal_contradiction
# ---------------------------------------------------------------------- #

# Subject capture is deliberately bounded to 1-2 words immediately before
# the verb (not "as much preceding text as fits") — an unbounded/greedy
# capture lets leftward context bleed into the subject (e.g. "The score"
# vs. "Later the score" for the same actual subject "score"), which
# silently defeats same-subject grouping below.
_NUMERIC_ASSERTION_RE = re.compile(
    r"\b([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\s+(?:is|was|equals)\s+(-?\d+(?:\.\d+)?)\b"
)
_POSITIVE_ASSERTION_RE = re.compile(
    r"\b([a-z]+(?:\s+[a-z]+)?)\s+is\s+(?!not\b)([a-z]+(?:\s+[a-z]+)?)(?:[.,;]|$)", re.IGNORECASE
)
_NEGATIVE_ASSERTION_RE = re.compile(
    r"\b([a-z]+(?:\s+[a-z]+)?)\s+is not\s+([a-z]+(?:\s+[a-z]+)?)(?:[.,;]|$)", re.IGNORECASE
)


def check_internal_contradiction(context: CheckContext) -> CheckResult:
    text = context.text
    contradictions: list[str] = []
    checked_count = 0

    by_subject: dict[str, list[float]] = {}
    for m in _NUMERIC_ASSERTION_RE.finditer(text):
        subject = m.group(1).strip().lower()
        by_subject.setdefault(subject, []).append(float(m.group(2)))
    for subject, values in by_subject.items():
        if len(values) < 2:
            continue
        checked_count += 1
        distinct = sorted(set(values))
        if len(distinct) > 1:
            contradictions.append(f"{subject!r} asserted as {distinct}")

    positives = {
        (m.group(1).strip().lower(), m.group(2).strip().lower())
        for m in _POSITIVE_ASSERTION_RE.finditer(text)
    }
    negatives = {
        (m.group(1).strip().lower(), m.group(2).strip().lower())
        for m in _NEGATIVE_ASSERTION_RE.finditer(text)
    }
    checked_count += len({s for s, _ in positives} & {s for s, _ in negatives})
    for subject, predicate in positives & negatives:
        contradictions.append(f"{subject!r} is asserted as both {predicate!r} and not {predicate!r}")

    if contradictions:
        return CheckResult(
            name="internal_contradiction", status=CheckStatus.FAIL, weight=_DEFAULT_WEIGHT,
            detail=f"{len(contradictions)} internal contradiction(s) detected.", evidence=contradictions,
        )
    if checked_count > 0:
        return CheckResult(
            name="internal_contradiction", status=CheckStatus.PASS, weight=_DEFAULT_WEIGHT,
            detail=f"Checked {checked_count} repeated assertion(s); no contradiction found.",
        )
    # Deliberately conservative (documented, not accidental): the absence
    # of a detected pattern doesn't prove the text is consistent, it only
    # means this cheap heuristic found nothing repeatable to compare — a
    # false FAIL here would be far more damaging to trust than an honest
    # "couldn't check", so this biases toward INCONCLUSIVE.
    return CheckResult(
        name="internal_contradiction", status=CheckStatus.INCONCLUSIVE, weight=_DEFAULT_WEIGHT,
        detail="No repeated same-subject assertion found to compare for contradiction.",
    )


# ---------------------------------------------------------------------- #
# check_unsupported_certainty
# ---------------------------------------------------------------------- #

_CERTAINTY_RE = re.compile(
    r"\b(definitely|guaranteed|100% certain|always|never|there is no chance|"
    r"without a doubt|absolutely certain|no doubt)\b",
    re.IGNORECASE,
)
_REASONING_MARKER_RE = re.compile(
    r"\b(because|since|therefore|due to|given that|as a result|this is because)\b", re.IGNORECASE
)
_CITATION_MARKER_RE = re.compile(
    r"\[\d+\]|according to|based on the (?:document|source|data|evidence)", re.IGNORECASE
)


def check_unsupported_certainty(context: CheckContext) -> CheckResult:
    matches = _CERTAINTY_RE.findall(context.text)
    if not matches:
        return CheckResult(
            name="unsupported_certainty", status=CheckStatus.INCONCLUSIVE, weight=_DEFAULT_WEIGHT,
            detail="No absolute-certainty language found.",
        )

    has_citation = bool(_CITATION_MARKER_RE.search(context.text))
    has_reasoning = bool(_REASONING_MARKER_RE.search(context.text))
    if has_citation or has_reasoning:
        return CheckResult(
            name="unsupported_certainty", status=CheckStatus.PASS, weight=_DEFAULT_WEIGHT,
            detail="Certainty language is backed by a citation or a reasoning chain.",
        )
    return CheckResult(
        name="unsupported_certainty", status=CheckStatus.FAIL, weight=_DEFAULT_WEIGHT,
        detail=f"{len(matches)} absolute-certainty phrase(s) with no citation or reasoning chain — "
               f"overclaiming, not necessarily wrong.",
        evidence=sorted(set(m.lower() for m in matches)),
    )


# ---------------------------------------------------------------------- #
# check_refusal_consistency
# ---------------------------------------------------------------------- #

# Subject capture has no anchor on its right side (unlike _ASSERTION_RE's
# forced " is "), so neither pure greedy nor pure non-greedy word-count
# bounding works: greedy over-captures trailing clauses ("winner without
# more data"), non-greedy under-captures multi-word subjects ("exact"
# instead of "exact score"). A negative lookahead against common
# preposition/conjunction stop-words lets it stop at the real clause
# boundary either way, capturing exactly the noun phrase.
_REFUSAL_STOPWORDS = (
    "without|with|in|on|at|before|after|due|because|and|or|but|given|since|for|from|by"
)
_REFUSAL_RE = re.compile(
    r"\b(?:cannot|can'?t|unable to|no way to) (?:determine|confirm|know|say|verify)\s+"
    rf"(?:the\s+)?([a-z]+(?:\s+(?!(?:{_REFUSAL_STOPWORDS})\b)[a-z]+){{0,2}})",
    re.IGNORECASE,
)
# Predicate class includes "-" — a score like "3-1" is a realistic
# assertion value that a plain [a-z0-9 ] class silently fails to match
# past the hyphen, making the whole assertion invisible to the refusal-
# contradiction comparison below.
_ASSERTION_RE = re.compile(
    r"\bthe\s+([a-z][a-z0-9 ]{2,40}?)\s+is\s+([a-z0-9][a-z0-9 \-]{0,40}?)(?:[.,;]|$)", re.IGNORECASE
)


def check_refusal_consistency(context: CheckContext) -> CheckResult:
    refusals = [m.group(1).strip().lower() for m in _REFUSAL_RE.finditer(context.text)]
    if not refusals:
        return CheckResult(
            name="refusal_consistency", status=CheckStatus.INCONCLUSIVE, weight=_DEFAULT_WEIGHT,
            detail="No refusal language found.",
        )

    assertions = {m.group(1).strip().lower() for m in _ASSERTION_RE.finditer(context.text)}
    contradicted = [r for r in refusals if r in assertions]
    if contradicted:
        return CheckResult(
            name="refusal_consistency", status=CheckStatus.FAIL, weight=_DEFAULT_WEIGHT,
            detail="The answer says it cannot determine something, then asserts it anyway.",
            evidence=contradicted,
        )
    return CheckResult(
        name="refusal_consistency", status=CheckStatus.PASS, weight=_DEFAULT_WEIGHT,
        detail=f"Checked {len(refusals)} refusal statement(s); no later contradiction found.",
    )


# Module-level so new checks are added here without touching the engine.
DETERMINISTIC_CHECKS: list[Callable[[CheckContext], CheckResult]] = [
    check_arithmetic,
    check_citation_support,
    check_internal_contradiction,
    check_unsupported_certainty,
    check_refusal_consistency,
]
