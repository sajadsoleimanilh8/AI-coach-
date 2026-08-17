from __future__ import annotations

from dataclasses import dataclass

from nexus.core.exceptions import ProviderUnavailableError
from nexus.core.router import ModelRouter
from nexus.core.types import Message, TaskType
from nexus.sports.adapter import SportsDataAdapter, SportsMetric
from nexus.sports.prematch_health import PreMatchAssessment, PreMatchHealthClient
from nexus.sports.psychology import (
    PsychologyFindings,
    build_prompt_context,
    derive_psychology_factors,
)
from nexus.sports.psychology_adapter import PsychologyAssessment, PsychologyClient
from nexus.sports.tactical import TacticalFinding, derive_findings

_SYSTEM_PROMPT = (
    "You are a football (soccer) tactical coach assistant. Narrate ONLY the "
    "findings and unavailable-metric list given to you below — never invent, "
    "estimate, or infer a value for anything not explicitly provided. Where a "
    "metric could not be measured, say so plainly and explain why (e.g. low "
    "upstream tracking confidence); never smooth over a gap with a plausible-"
    "sounding guess. Keep the tone practical and coach-facing."
)

_PREMATCH_SYSTEM_PROMPT = (
    "You are a football (soccer) coach assistant reporting on a player's "
    "pre-match readiness. Every number below — physical readiness, fatigue, "
    "recovery, performance risk, workload risk — has ALREADY been computed "
    "deterministically from the player's own questionnaire answers. Your only "
    "job is to explain what those given numbers and factors mean for team "
    "selection and session planning.\n\n"
    "Hard rules:\n"
    "- Never calculate, re-derive, adjust, average, or estimate any score. "
    "Use the exact values given; if you state a number, it must be one that "
    "appears verbatim below.\n"
    "- Never introduce a metric that is not listed. If something a coach "
    "would want is absent, say it is not available rather than inferring it.\n"
    "- This is a performance-readiness estimate built from self-reported "
    "answers, NOT a medical assessment, diagnosis, or injury prediction. Do "
    "not give medical advice, diagnose a condition, or suggest a player is "
    "injured. Reported pain is something to flag for a qualified person to "
    "look at, never something for you to interpret clinically.\n"
    "- Self-reported data is subjective; say so where it matters.\n"
    "Keep it short, practical, and coach-facing."
)

_PSYCHOLOGY_SYSTEM_PROMPT = (
    "You are a football (soccer) coach assistant reporting on a player's "
    "pre-match mental readiness. Every number below — mental readiness, focus, "
    "confidence, stress, pressure risk, mental performance risk — has ALREADY "
    "been computed deterministically from the player's own answers to a "
    "13-item self-report questionnaire. Your only job is to explain what those "
    "given numbers and factors mean for team selection, in-match management, "
    "and what to watch.\n\n"
    "Hard rules:\n"
    "- Never calculate, re-derive, adjust, average, rescale, or estimate any "
    "score. Use the exact values given; if you state a number, it must be one "
    "that appears verbatim below.\n"
    "- Never introduce a metric that is not listed. If something a coach would "
    "want is absent, say it is not available rather than inferring it.\n"
    "- This is a mental-READINESS estimate built from self-reported answers. "
    "It is NOT emotion detection, NOT a psychological or clinical assessment, "
    "and NOT a diagnosis. Do not diagnose, do not use clinical language "
    "(anxiety, depression, disorder), and do not speculate about the player's "
    "emotional state or mental health. Reported stress is a performance factor "
    "to manage, never a condition for you to interpret.\n"
    "- Any 'observed-performance proxy' listed is derived from match tracking "
    "data. Describe it as on-pitch performance only; never as evidence of what "
    "the player feels.\n"
    "- Self-reported data is subjective; say so where it matters.\n"
    "Keep it short, practical, and coach-facing."
)


def unavailable_reason(metric: SportsMetric) -> str:
    """One human-readable line naming a metric and why it couldn't be used —
    reused by both CoachAssistant and SportsAgent so the phrasing stays
    consistent everywhere an "unavailable metrics" list is surfaced."""
    if metric.confidence == "low_upstream_confidence":
        return f"{metric.metric_name}: unavailable — upstream tracking confidence too low to compute"
    if metric.value is None:
        return f"{metric.metric_name}: unavailable — not yet computed (sample_size={metric.sample_size})"
    return f"{metric.metric_name}: unavailable — confidence={metric.confidence}"


def _findings_text(findings: list[TacticalFinding]) -> str:
    if not findings:
        return "No tactical findings could be derived from available metrics."
    return "\n".join(
        f"- [{f.area}] {f.assessment} (confidence={f.confidence}): {f.explanation}"
        for f in findings
    )


def _unavailable_text(reasons: list[str]) -> str:
    return "\n".join(f"- {reason}" for reason in reasons) if reasons else "None."


@dataclass
class CoachReport:
    match_id: str
    findings: list[TacticalFinding]
    unavailable_metrics: list[str]
    coverage: float
    narrative: str
    model_used: str


@dataclass
class PreMatchCoachReport:
    """A narrated pre-match readiness report."""

    player_id: str
    match_id: str | None
    assessment: PreMatchAssessment
    narrative: str
    model_used: str


@dataclass
class PsychologyCoachReport:
    """A narrated mental-readiness report."""

    player_id: str
    match_id: str | None
    assessment: PsychologyAssessment
    findings: PsychologyFindings
    narrative: str
    model_used: str


class CoachAssistant:
    """adapter -> derive_findings (deterministic) -> LLM narrates ONLY the
    derived findings plus what couldn't be measured. The narrative is the
    single LLM-generated field on CoachReport; everything else is Python
    arithmetic over the adapter's output."""

    def __init__(
        self,
        router: ModelRouter,
        adapter: SportsDataAdapter,
        prematch_client: PreMatchHealthClient | None = None,
        psychology_client: PsychologyClient | None = None,
    ) -> None:
        self._router = router
        self._adapter = adapter
        self._prematch_client = prematch_client
        self._psychology_client = psychology_client

    async def build_report(self, match_id: str, player_id: int | None = None) -> CoachReport:
        analysis = (
            await self._adapter.get_player_analysis(match_id, player_id)
            if player_id is not None
            else await self._adapter.get_match_analysis(match_id)
        )
        findings = derive_findings(analysis)
        unavailable_metrics = [unavailable_reason(m) for m in analysis.unavailable]

        decision, provider = await self._router.route_with_failover(task_type=TaskType.SPORTS)

        user_content = (
            f"Match {match_id}" + (f", player {player_id}" if player_id is not None else "")
            + f"\n\nDerived findings:\n{_findings_text(findings)}\n\n"
            f"Unavailable metrics:\n{_unavailable_text(unavailable_metrics)}\n\n"
            "Write a short coach report covering the findings above."
        )
        result = await provider.generate(
            [
                Message(role="system", content=_SYSTEM_PROMPT),
                Message(role="user", content=user_content),
            ],
            model_id=decision.model_id,
            temperature=0.7,
        )

        return CoachReport(
            match_id=match_id,
            findings=findings,
            unavailable_metrics=unavailable_metrics,
            coverage=analysis.coverage,
            narrative=result.content,
            model_used=result.model_used,
        )

    async def build_prematch_report(
        self, player_id: str, match_id: str | None = None
    ) -> PreMatchCoachReport | None:
        """Fetch the player's latest pre-match assessment and narrate it."""
        if self._prematch_client is None:
            raise ProviderUnavailableError(
                "Pre-match health client is not configured on this CoachAssistant"
            )

        assessment = await self._prematch_client.get_latest(player_id)
        if assessment is None:
            return None

        decision, provider = await self._router.route_with_failover(task_type=TaskType.SPORTS)

        user_content = (
            "Pre-match readiness assessment (already computed — do not "
            "recalculate any value):\n\n"
            f"{assessment.as_prompt_context()}\n\n"
            "Write a short pre-match readiness report for the coach: what "
            "these numbers mean for this player's availability and intensity "
            "for the upcoming match, and what to watch. Reference only the "
            "values above."
        )
        result = await provider.generate(
            [
                Message(role="system", content=_PREMATCH_SYSTEM_PROMPT),
                Message(role="user", content=user_content),
            ],
            model_id=decision.model_id,
            temperature=0.7,
        )

        return PreMatchCoachReport(
            player_id=player_id,
            match_id=assessment.match_id or match_id,
            assessment=assessment,
            narrative=result.content,
            model_used=result.model_used,
        )

    async def build_psychology_report(
        self, player_id: str, match_id: str | None = None
    ) -> PsychologyCoachReport | None:
        """Fetch the player's latest mental-readiness assessment and narrate it."""
        if self._psychology_client is None:
            raise ProviderUnavailableError(
                "Psychology client is not configured on this CoachAssistant"
            )

        assessment = await self._psychology_client.get_latest(player_id, match_id)
        if assessment is None:
            return None

        findings = derive_psychology_factors(assessment)

        decision, provider = await self._router.route_with_failover(task_type=TaskType.SPORTS)

        user_content = (
            "Pre-match mental-readiness assessment (already computed — do not "
            "recalculate any value):\n\n"
            f"{build_prompt_context(assessment, findings)}\n\n"
            "Write a short mental-readiness report for the coach: what these "
            "numbers mean for this player's involvement and in-match "
            "management, and what to watch. Reference only the values above."
        )
        result = await provider.generate(
            [
                Message(role="system", content=_PSYCHOLOGY_SYSTEM_PROMPT),
                Message(role="user", content=user_content),
            ],
            model_id=decision.model_id,
            temperature=0.7,
        )

        return PsychologyCoachReport(
            player_id=player_id,
            match_id=assessment.match_id or match_id,
            assessment=assessment,
            findings=findings,
            narrative=result.content,
            model_used=result.model_used,
        )
