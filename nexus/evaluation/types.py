from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# Which provider set produced a run. Comparing across these is meaningless
# — a fakes run scores whatever its scripts say, so diffing it against a
# real-provider run manufactures regressions that describe the harness,
# not the model. compare_runs() refuses rather than emitting that diff.
# "unknown" covers rows written before this field existed; it is never
# assumed compatible with anything, including itself.
ProviderMode = Literal["real", "fake", "unknown"]


@dataclass
class SkippedSuite:
    """A suite deliberately NOT run, with the reason. Recorded explicitly
    so a pinned run can never quietly report a model-independent suite as
    if the pinned model had earned its score."""

    suite: str
    reason: str


@dataclass
class EvalCase:
    id: str
    suite: str
    input: dict[str, Any]
    expected: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseOutcome:
    case_id: str
    passed: bool
    score: float
    actual: dict[str, Any]
    detail: str
    latency_seconds: float
    cost_usd: float


@dataclass
class SuiteResult:
    suite: str
    outcomes: list[CaseOutcome]
    pass_rate: float
    mean_score: float
    total_cost_usd: float
    total_latency_seconds: float

    @classmethod
    def from_outcomes(cls, suite: str, outcomes: list[CaseOutcome]) -> SuiteResult:
        n = len(outcomes)
        return cls(
            suite=suite,
            outcomes=outcomes,
            pass_rate=(sum(1 for o in outcomes if o.passed) / n) if n else 0.0,
            mean_score=(sum(o.score for o in outcomes) / n) if n else 0.0,
            total_cost_usd=sum(o.cost_usd for o in outcomes),
            total_latency_seconds=sum(o.latency_seconds for o in outcomes),
        )


@dataclass
class EvalRun:
    run_id: str
    started_at: float
    finished_at: float
    suites: list[SuiteResult]
    config_snapshot: dict[str, Any]
    git_sha: str | None
    # Defaulted so every existing construction site (and every run loaded
    # from a row written before these columns existed) stays valid. The
    # default is "unknown" rather than "fake" on purpose: guessing a mode
    # is exactly the failure this field exists to prevent.
    provider_mode: ProviderMode = "unknown"
    pinned_model_id: str | None = None
    skipped_suites: list[SkippedSuite] = field(default_factory=list)
