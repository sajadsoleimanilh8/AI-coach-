from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from nexus.core.types import Usage

_HIGH_THRESHOLD = 0.85
_MEDIUM_THRESHOLD = 0.65
_LOW_THRESHOLD = 0.45


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    weight: float
    detail: str
    evidence: list[str] = field(default_factory=list)


class ConfidenceBand(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"
    UNVERIFIED = "unverified"


def band_for_score(score: float | None) -> ConfidenceBand:
    if score is None:
        return ConfidenceBand.UNVERIFIED
    if score >= _HIGH_THRESHOLD:
        return ConfidenceBand.HIGH
    if score >= _MEDIUM_THRESHOLD:
        return ConfidenceBand.MEDIUM
    if score >= _LOW_THRESHOLD:
        return ConfidenceBand.LOW
    return ConfidenceBand.UNCERTAIN


def score_checks(checks: list[CheckResult]) -> float | None:
    """score = sum(weight for PASS) / sum(weight for PASS|FAIL)."""
    scored = [c for c in checks if c.status in (CheckStatus.PASS, CheckStatus.FAIL)]
    denominator = sum(c.weight for c in scored)
    if denominator == 0:
        return None
    numerator = sum(c.weight for c in scored if c.status == CheckStatus.PASS)
    return numerator / denominator


@dataclass
class VerificationReport:
    checks: list[CheckResult]
    score: float | None
    band: ConfidenceBand
    summary: str
    uncertainty_notes: list[str] = field(default_factory=list)
    escalated: bool = False
    extra_usage: Usage = field(default_factory=Usage)
