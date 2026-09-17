from __future__ import annotations

from dataclasses import dataclass, field

from nexus.core.exceptions import ProviderUnavailableError
from nexus.core.router import ModelRouter
from nexus.core.types import Message, TaskType
from nexus.logging_setup.logger import get_logger
from nexus.sports.adapter import SportsDataAdapter, SportsMetric
from nexus.sports.game_plan import GamePlan, build_game_plan
from nexus.sports.narrative_guard import audit_narrative
from nexus.sports.prematch_health import PreMatchAssessment, PreMatchHealthClient
from nexus.sports.psychology import (
    PsychologyFindings,
    build_prompt_context,
    derive_psychology_factors,
)
from nexus.sports.psychology_adapter import PsychologyAssessment, PsychologyClient
from nexus.sports.tactical import TacticalFinding, derive_findings
from nexus.sports.timeline import TacticalTimeline, timeline_unavailable_reasons

_SYSTEM_PROMPT = (
    "You are a football (soccer) tactical coach assistant. Narrate ONLY the "
    "findings, game plan, and unavailable-metric list given to you below — "
    "never invent, estimate, or infer a value for anything not explicitly "
    "provided. Where a metric could not be measured, say so plainly and "
    "explain why (e.g. low upstream tracking confidence); never smooth over a "
    "gap with a plausible-sounding guess. Keep the tone practical and "
    "coach-facing.\n\n"
    "Hard rules:\n"
    "- Every number you write must appear verbatim in the context below. Do "
    "not compute totals, averages, percentages, differences, or rankings of "
    "your own, and do not convert between units or scales.\n"
    "- The tactical read, the shape, the principles of play and the in-game "
    "adjustments have ALREADY been derived deterministically from the data. "
    "Explain and prioritise them; do not add adjustments of your own, and do "
    "not propose a different shape.\n"
    "- Always state which metrics were unavailable. An unmeasured metric is "
    "never evidence of a strength, a weakness, or a normal value.\n"
    "- If the context is marked PARTIAL, open by saying the report is partial "
    "and why, and keep every conclusion hedged accordingly. Do not write "
    "confident prose over thin data.\n"
    "- Describe NO match events and NO chronology. The context holds "
    "match-wide aggregates, not a running account of what happened. Never "
    "write about halves, periods, opening or closing spells, what happened "
    "'as the match progressed', or anything getting better or worse over "
    "time, unless a 'tactical timeline' section below actually supplies it. "
    "Inventing a narrative arc is a fabrication even when it quotes no "
    "number.\n"
    "- Do not attribute events to either side: no goals, chances, "
    "breakthroughs, turnovers, substitutions, or passages of play unless "
    "they appear in the context.\n"
    "- 'Phase 4' anywhere in this context names a stage of the data "
    "pipeline, never a period of the match. The 'phases' of a tactical "
    "timeline are states of play (e.g. build-up, transition), not "
    "chronological quarters.\n"
    "- The adjustments are prospective instructions for the next match, not "
    "changes that were already made during this one. Write them as things to "
    "do, never as things that were done.\n"
    "Structure the reply as: Tactical read; Game plan (shape + principles); "
    "In-game adjustments; What could not be measured."
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


# All three report prompts ask the model to restate values that were already
# computed, never to reason freely. 0.7 was the generation default and is
# simply wrong for that: sampling entropy on a grounded narration task buys
# nothing and is what lets a small model wander into inventing match events
# (a live qwen2.5:3b produced "in the second half..." and player injuries
# that appear nowhere in the context). Near-greedy decoding is the correct
# setting for transcription-shaped work.
#
# It lowers the rate of invented claims but does not eliminate them, which
# is why narrative_guard verifies the output rather than trusting it.
NARRATION_TEMPERATURE = 0.15

logger = get_logger("sports.coach")


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
    game_plan: GamePlan | None = None
    timeline: TacticalTimeline | None = None
    # Non-empty when the narrative made claims the data could not support
    # and a corrective retry did not clear them.
    narrative_warnings: list[str] = field(default_factory=list)

    @property
    def is_partial(self) -> bool:
        return self.game_plan is not None and self.game_plan.is_partial


@dataclass
class PreMatchCoachReport:
    """A narrated pre-match readiness report.

    `narrative` is the only LLM-generated field. `assessment` holds the
    backend's computed numbers exactly as received, so a caller can always
    check what the model was told against what it said."""

    player_id: str
    match_id: str | None
    assessment: PreMatchAssessment
    narrative: str
    model_used: str


@dataclass
class PsychologyCoachReport:
    """A narrated mental-readiness report.

    `narrative` is the only LLM-generated field. `assessment` holds the
    backend's computed numbers exactly as received and `findings` the
    deterministically-derived phrases, so a caller can always check what the
    model was told against what it said."""

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
        # Optional so existing construction sites keep working unchanged;
        # build_prematch_report() fails loudly rather than silently degrading
        # if it was never wired up.
        self._prematch_client = prematch_client
        # Same contract as _prematch_client above.
        self._psychology_client = psychology_client

    async def build_report(self, match_id: str, player_id: int | None = None) -> CoachReport:
        analysis = (
            await self._adapter.get_player_analysis(match_id, player_id)
            if player_id is not None
            else await self._adapter.get_match_analysis(match_id)
        )
        findings = derive_findings(analysis)
        game_plan = build_game_plan(analysis)

        # Timeline gaps are unavailable metrics too: without them the coach
        # would report a match with no measured pressing exactly like a match
        # where pressing was measured and found absent.
        unavailable_metrics = [unavailable_reason(m) for m in analysis.unavailable]
        unavailable_metrics += timeline_unavailable_reasons(analysis.timeline)

        decision, provider = await self._router.route_with_failover(task_type=TaskType.SPORTS)

        timeline_block = (
            analysis.timeline.as_prompt_context()
            if analysis.timeline is not None
            else "No tactical timeline was served for this match."
        )

        user_content = (
            f"Match {match_id}" + (f", player {player_id}" if player_id is not None else "")
            + f"\nMetric coverage: {analysis.coverage:.1%}"
            + (
                "\nTHIS IS A LOW-COVERAGE MATCH: write an explicitly partial "
                "report and say what is missing."
                if game_plan.is_partial
                else ""
            )
            + f"\n\nDerived findings:\n{_findings_text(findings)}\n\n"
            f"Tactical timeline (phases / pressing / transitions / territory):\n"
            f"{timeline_block}\n\n"
            f"Game plan (already derived — explain, do not extend):\n"
            f"{game_plan.as_prompt_context()}\n\n"
            f"Unavailable metrics:\n{_unavailable_text(unavailable_metrics)}\n\n"
            "Write a short coach report covering the tactical read, the game "
            "plan, the in-game adjustments, and what could not be measured."
        )
        messages = [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(role="user", content=user_content),
        ]
        result = await provider.generate(
            messages, model_id=decision.model_id, temperature=NARRATION_TEMPERATURE
        )

        # The prompt asks the model not to invent chronology or events; this
        # checks whether it complied. A small model complies most of the
        # time, which is not the same as always, so compliance is verified
        # rather than assumed. One corrective retry fixes most lapses; if
        # the second attempt is still unsupported, the narrative is kept but
        # the report carries an explicit warning -- surfacing a flawed
        # narrative honestly beats silently shipping it as fact.
        audit = audit_narrative(result.content, timeline=analysis.timeline)
        narrative_warnings: list[str] = []
        if not audit.is_clean:
            logger.info(
                "coach narrative for match %s contained unsupported claims %s; retrying",
                match_id,
                audit.chronology_claims + audit.event_claims,
            )
            retry = await provider.generate(
                messages
                + [
                    Message(role="assistant", content=result.content),
                    Message(role="user", content=audit.correction_instruction()),
                ],
                model_id=decision.model_id,
                temperature=NARRATION_TEMPERATURE,
            )
            retry_audit = audit_narrative(retry.content, timeline=analysis.timeline)
            if retry_audit.is_clean:
                result = retry
            else:
                # Keep whichever attempt made fewer unsupported claims.
                if len(retry_audit.warnings) < len(audit.warnings):
                    result, retry_audit = retry, retry_audit
                else:
                    retry_audit = audit
                narrative_warnings = retry_audit.warnings
                logger.warning(
                    "coach narrative for match %s still contained unsupported "
                    "claims after a corrective retry; returning it flagged: %s",
                    match_id,
                    narrative_warnings,
                )

        return CoachReport(
            match_id=match_id,
            findings=findings,
            unavailable_metrics=unavailable_metrics,
            coverage=analysis.coverage,
            narrative=result.content,
            model_used=result.model_used,
            game_plan=game_plan,
            timeline=analysis.timeline,
            narrative_warnings=narrative_warnings,
        )

    async def build_prematch_report(
        self, player_id: str, match_id: str | None = None
    ) -> PreMatchCoachReport | None:
        """Fetch the player's latest pre-match assessment and narrate it.

        Same shape as build_report above: everything quantitative is computed
        upstream (here, by the football backend's deterministic scorer), and
        exactly one LLM call turns it into prose. The model is handed the
        assessment as text and is told, in the system prompt, that it may not
        recompute any of it.

        Returns None when the player has no assessment yet -- the caller
        surfaces that as "not submitted", never as a zeroed-out report.
        Backend outages propagate as ProviderUnavailableError.
        """
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
            temperature=NARRATION_TEMPERATURE,
        )

        return PreMatchCoachReport(
            player_id=player_id,
            # The assessment's own match_id is authoritative -- it is what the
            # player actually submitted against. The path's match_id is only
            # a fallback label for the case where the questionnaire was filed
            # before any Match row existed.
            match_id=assessment.match_id or match_id,
            assessment=assessment,
            narrative=result.content,
            model_used=result.model_used,
        )

    async def build_psychology_report(
        self, player_id: str, match_id: str | None = None
    ) -> PsychologyCoachReport | None:
        """Fetch the player's latest mental-readiness assessment and narrate it.

        Same shape as build_prematch_report above, and the same guarantee:
        everything quantitative is computed upstream (by the football backend's
        deterministic ai/psychology_ai/ engine), the phrases are derived from it
        in pure Python by derive_psychology_factors, and exactly one LLM call
        turns the result into prose. The model is handed that context as text
        and is told, in the system prompt, that it may not recompute any of it.

        Returns None when the player has no assessment yet -- the caller
        surfaces that as "not submitted", never as a zeroed-out report. Backend
        outages propagate as ProviderUnavailableError.
        """
        if self._psychology_client is None:
            raise ProviderUnavailableError(
                "Psychology client is not configured on this CoachAssistant"
            )

        assessment = await self._psychology_client.get_latest(player_id, match_id)
        if assessment is None:
            return None

        # Deterministic derivation happens BEFORE the model is involved, and
        # its output is the only thing the model gets to see.
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
            temperature=NARRATION_TEMPERATURE,
        )

        return PsychologyCoachReport(
            player_id=player_id,
            # The assessment's own match_id is authoritative, for the same
            # reason as in build_prematch_report above.
            match_id=assessment.match_id or match_id,
            assessment=assessment,
            findings=findings,
            narrative=result.content,
            model_used=result.model_used,
        )
