"""Grounding tests for the coach report.

These enforce the three properties the coaching path exists to guarantee:

1. Every number in the narrative appears in the context the model was given.
2. Unavailable metrics are always listed, never silently dropped.
3. A low-coverage match yields an explicitly partial report, not confident
   prose over thin data.

(1) is checked against a deliberately hostile provider that fabricates
numbers. That is the point: the test must fail when a model invents a value,
so it has to be exercised by a model that does. A well-behaved stub would
make the test pass for the wrong reason.
"""

from __future__ import annotations

import re

import pytest

from nexus.core.providers import AIProvider
from nexus.core.router import RoutingDecision
from nexus.core.types import GenerationChunk, GenerationResult, ModelInfo, RoutingPolicy, Usage
from nexus.sports.adapter import MatchAnalysis, SportsDataAdapter, SportsMetric
from nexus.sports.coach import CoachAssistant
from nexus.sports.game_plan import LOW_COVERAGE_THRESHOLD, build_game_plan
from nexus.sports.timeline import TIMELINE_SECTIONS, parse_timeline

# Matches integers and decimals, including the percent/decimal forms the
# report uses. Deliberately greedy: a false positive here is a test we have
# to look at, a false negative is an ungrounded number reaching a coach.
_NUMBER = re.compile(r"\d+(?:\.\d+)?")

# Numbers that are structural rather than claims about the match: list
# markers, section numbering, and the thresholds named in the prompt itself.
_STRUCTURAL = {"1", "2", "3", "4", "5", "40", "65", "100"}


def _numbers(text: str) -> set[str]:
    return {n for n in _NUMBER.findall(text)} - _STRUCTURAL


def _metric(
    name: str,
    value,
    confidence: str = "normal",
    sample_size: int = 10,
    sub_scores: dict | None = None,
    player_id: int | None = None,
) -> SportsMetric:
    return SportsMetric(
        metric_name=name,
        value=value,
        method="deterministic",
        confidence=confidence,
        sample_size=sample_size,
        sub_scores=sub_scores or {},
        player_id=player_id,
        team_id="team-home",
    )


class _FakeAdapter(SportsDataAdapter):
    def __init__(self, analysis: MatchAnalysis) -> None:
        self._analysis = analysis

    async def get_match_analysis(self, match_id: str) -> MatchAnalysis:
        return self._analysis

    async def get_player_analysis(self, match_id: str, player_id: int) -> MatchAnalysis:
        return self._analysis


class _RecordingProvider(AIProvider):
    """Captures the prompt and returns whatever narrative the test wants."""

    name = "fake"

    def __init__(self, narrative: str = "ok") -> None:
        self.received_messages = None
        self._narrative = narrative

    async def generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        self.received_messages = messages
        return GenerationResult(
            content=self._narrative,
            model_used=model_id,
            provider_name=self.name,
            usage=Usage(),
        )

    async def stream_generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        yield GenerationChunk(delta="unused", done=True, usage=Usage())

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> bool:
        return True

    def count_tokens(self, text: str, *, model_id: str) -> int:
        return len(text)

    @property
    def prompt_text(self) -> str:
        return "\n".join(m.content for m in (self.received_messages or []))


class _FakeRouter:
    def __init__(self, provider: AIProvider) -> None:
        self._provider = provider

    async def route_with_failover(self, **kwargs):
        return (
            RoutingDecision(
                provider_name=self._provider.name,
                model_id="fake-model",
                task_type=kwargs.get("task_type"),
                policy=RoutingPolicy.BALANCED,
                reason="fake",
            ),
            self._provider,
        )


def _rich_analysis() -> MatchAnalysis:
    """A well-covered match: 4 of 5 team metrics available."""
    team = [
        _metric("compactness_score", 72.5, sample_size=539),
        _metric("formation_stability_score", 64.0, sample_size=161),
        _metric("formation", "4-3-3", sample_size=12),
        _metric(
            "weak_zone_map",
            22.0,
            sample_size=539,
            sub_scores={
                "zone_1_1": 0.0,
                "zone_6_3": 0.885,
                "under_occupied_zone_count": 22,
                "basis": "own_density_only",
            },
        ),
        _metric("pressing_intensity_score", None, confidence="low_upstream_confidence", sample_size=0),
    ]
    players = [
        _metric("decision_making_score", 30.0, player_id=7),
        _metric("press_resistance_score", 35.0, player_id=8),
    ]
    unavailable = [m for m in team + players if not m.is_available]
    available = [m for m in team + players if m.is_available]
    return MatchAnalysis(
        match_id="m-rich",
        team_metrics=team,
        player_metrics=players,
        unavailable=unavailable,
        coverage=len(available) / len(team + players),
        team_ids=["team-home"],
    )


def _sparse_analysis() -> MatchAnalysis:
    """A deliberately low-coverage match: 1 of 6 metrics available."""
    team = [
        _metric("compactness_score", 55.0, sample_size=12),
        _metric("formation_stability_score", None, confidence="low_upstream_confidence", sample_size=0),
        _metric("pressing_intensity_score", None, confidence="low_upstream_confidence", sample_size=0),
    ]
    players = [
        _metric("decision_making_score", None, confidence="low_upstream_confidence", player_id=7),
        _metric("passing_vision_score", None, confidence="low_upstream_confidence", player_id=7),
        _metric("first_touch_score", None, confidence="low_upstream_confidence", player_id=8),
    ]
    unavailable = [m for m in team + players if not m.is_available]
    available = [m for m in team + players if m.is_available]
    return MatchAnalysis(
        match_id="m-sparse",
        team_metrics=team,
        player_metrics=players,
        unavailable=unavailable,
        coverage=len(available) / len(team + players),
        team_ids=["team-home"],
    )


def _timeline_payload() -> dict:
    return {
        "match_id": "m-rich",
        "phases": {
            "share_by_phase": {"build_up": 0.42, "settled_offense": 0.31, "transition": 0.27},
            "confidence": "normal",
        },
        "pressing": {
            "intensity_by_phase": {"first_half": 61.0},
            "success_rate": 0.33,
            "confidence": "normal",
        },
        "transitions": {"count": 18, "mean_speed_mps": 4.75, "confidence": "normal"},
        "territory": {"share_by_third": {"final": 0.29}, "field_tilt": 0.46},
    }


# --------------------------------------------------------------------------
# 1. No number in the narrative that is absent from the context
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_number_in_the_narrative_appears_in_the_context() -> None:
    provider = _RecordingProvider()
    coach = CoachAssistant(_FakeRouter(provider), _FakeAdapter(_rich_analysis()))

    report = await coach.build_report("m-rich")

    ungrounded = _numbers(report.narrative) - _numbers(provider.prompt_text)
    assert not ungrounded, f"narrative contains numbers absent from the context: {ungrounded}"


@pytest.mark.asyncio
async def test_the_grounding_check_actually_catches_a_fabricating_model() -> None:
    """The guard above is only worth anything if it fails on a bad model.

    Without this, a provider that returned "ok" would make the grounding
    test pass forever while proving nothing.
    """
    fabricated = (
        "Their compactness of 88.4 and pressing intensity of 71.2 mean they "
        "won 63% of second balls."
    )
    provider = _RecordingProvider(fabricated)
    coach = CoachAssistant(_FakeRouter(provider), _FakeAdapter(_rich_analysis()))

    report = await coach.build_report("m-rich")

    ungrounded = _numbers(report.narrative) - _numbers(provider.prompt_text)
    assert ungrounded, "the fabricated numbers should have been detected as ungrounded"
    assert {"88.4", "71.2", "63"} <= ungrounded


@pytest.mark.asyncio
async def test_real_metric_values_are_present_verbatim_in_the_context() -> None:
    """A number the model IS allowed to use must actually be in the prompt."""
    provider = _RecordingProvider()
    coach = CoachAssistant(_FakeRouter(provider), _FakeAdapter(_rich_analysis()))

    await coach.build_report("m-rich")

    assert "72.5" in provider.prompt_text
    assert "64.0" in provider.prompt_text
    assert "compactness_score" in provider.prompt_text


@pytest.mark.asyncio
async def test_timeline_numbers_reach_the_context_when_a_timeline_exists() -> None:
    analysis = _rich_analysis()
    analysis.timeline = parse_timeline(_timeline_payload(), match_id="m-rich")
    provider = _RecordingProvider()
    coach = CoachAssistant(_FakeRouter(provider), _FakeAdapter(analysis))

    report = await coach.build_report("m-rich")

    assert "0.420" in provider.prompt_text  # phase share
    assert "0.330" in provider.prompt_text  # pressing success rate
    assert "18" in provider.prompt_text  # transition count
    assert "4.75" in provider.prompt_text  # transition speed
    areas = {f.area for f in report.findings}
    assert {"game_phases", "pressing_over_time", "transitions", "territory"} <= areas


# --------------------------------------------------------------------------
# 2. Unavailable metrics are always listed
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unavailable_metrics_are_always_listed_and_reach_the_prompt() -> None:
    provider = _RecordingProvider()
    coach = CoachAssistant(_FakeRouter(provider), _FakeAdapter(_rich_analysis()))

    report = await coach.build_report("m-rich")

    assert report.unavailable_metrics
    assert any("pressing_intensity_score" in r for r in report.unavailable_metrics)
    assert "Unavailable metrics" in provider.prompt_text
    assert "pressing_intensity_score" in provider.prompt_text


@pytest.mark.asyncio
async def test_a_missing_timeline_is_reported_section_by_section() -> None:
    """A match with no timeline must not read like a match measured to have
    no pressing and no transitions."""
    provider = _RecordingProvider()
    coach = CoachAssistant(_FakeRouter(provider), _FakeAdapter(_rich_analysis()))

    report = await coach.build_report("m-rich")

    for section in TIMELINE_SECTIONS:
        assert any(
            f"tactical_timeline.{section}" in reason for reason in report.unavailable_metrics
        ), f"timeline section {section} was not reported as unavailable"


@pytest.mark.asyncio
async def test_every_unavailable_metric_is_named_in_the_prompt() -> None:
    analysis = _sparse_analysis()
    provider = _RecordingProvider()
    coach = CoachAssistant(_FakeRouter(provider), _FakeAdapter(analysis))

    await coach.build_report("m-sparse")

    for metric in analysis.unavailable:
        assert metric.metric_name in provider.prompt_text


@pytest.mark.asyncio
async def test_unmeasured_metrics_never_become_weaknesses_or_strengths() -> None:
    """An absent metric says nothing; scoring it would invent a finding."""
    plan = build_game_plan(_sparse_analysis())

    scored = {w.key for w in plan.opponent_weaknesses} | {s.key for s in plan.opponent_strengths}
    assert "formation_stability_score" not in scored
    assert "pressing_intensity_score" not in scored
    assert "formation_stability_score" in plan.unmeasured


# --------------------------------------------------------------------------
# 3. Low coverage produces an explicit partial report
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_low_coverage_produces_an_explicit_partial_report() -> None:
    analysis = _sparse_analysis()
    assert analysis.coverage < LOW_COVERAGE_THRESHOLD

    provider = _RecordingProvider()
    coach = CoachAssistant(_FakeRouter(provider), _FakeAdapter(analysis))

    report = await coach.build_report("m-sparse")

    assert report.is_partial
    assert report.game_plan is not None and report.game_plan.is_partial
    assert "coverage" in report.game_plan.partial_reason
    assert "PARTIAL PLAN" in provider.prompt_text
    assert "LOW-COVERAGE MATCH" in provider.prompt_text


@pytest.mark.asyncio
async def test_the_partial_flag_is_not_set_on_a_well_covered_match() -> None:
    """The partial flag has to mean something, so it must not be always-on."""
    analysis = _rich_analysis()
    analysis.timeline = parse_timeline(_timeline_payload(), match_id="m-rich")
    assert analysis.coverage >= LOW_COVERAGE_THRESHOLD

    coach = CoachAssistant(_FakeRouter(_RecordingProvider()), _FakeAdapter(analysis))
    report = await coach.build_report("m-rich")

    assert not report.is_partial
    assert report.game_plan is not None and not report.game_plan.is_partial


def test_a_missing_timeline_alone_marks_the_plan_partial() -> None:
    """Even at good coverage, absent phase/pressing/transition/territory
    context is a real limit on the plan and has to be declared."""
    plan = build_game_plan(_rich_analysis())

    assert plan.is_partial
    assert "tactical timeline" in plan.partial_reason


# --------------------------------------------------------------------------
# Game plan structure
# --------------------------------------------------------------------------


def test_every_adjustment_is_tied_to_a_named_observed_weakness() -> None:
    plan = build_game_plan(_rich_analysis())

    assert plan.adjustments, "expected adjustments from a match with real weaknesses"
    weakness_labels = {w.label for w in plan.opponent_weaknesses}
    for adjustment in plan.adjustments:
        assert adjustment.targets_weakness in weakness_labels
        assert adjustment.supporting_metrics


def test_every_principle_and_adjustment_cites_supporting_metrics() -> None:
    plan = build_game_plan(_rich_analysis())

    for principle in plan.principles:
        assert principle.supporting_metrics
        assert principle.grounded_in
    for adjustment in plan.adjustments:
        assert adjustment.supporting_metrics


def test_the_plan_stays_within_three_to_five_principles_and_adjustments() -> None:
    plan = build_game_plan(_rich_analysis())

    assert len(plan.principles) <= 5
    assert len(plan.adjustments) <= 5


def test_no_shape_is_proposed_when_nothing_measured_supports_one() -> None:
    """Silence beats a default formation dressed up as a recommendation."""
    empty = MatchAnalysis(
        match_id="m-empty", team_metrics=[], player_metrics=[], unavailable=[], coverage=0.0
    )
    plan = build_game_plan(empty)

    assert plan.proposed_shape is not None
    assert plan.proposed_shape.shape == "no shape proposed"
    assert plan.is_partial


def test_the_game_plan_context_names_every_number_it_asserts() -> None:
    """Whatever the plan puts in the prompt must carry its own evidence."""
    plan = build_game_plan(_rich_analysis())
    context = plan.as_prompt_context()

    for weakness in plan.opponent_weaknesses:
        assert weakness.evidence in context
    for strength in plan.opponent_strengths:
        assert strength.evidence in context


# --------------------------------------------------------------------------
# 4. No fabricated match events or chronology
#
# The numeric checks above miss a real failure mode seen with a live 3B
# model: prose that quotes no number at all but invents a narrative arc
# ("in the second half...", "as the match progressed..."). The context holds
# match-wide aggregates, so any chronology is fabricated.
# --------------------------------------------------------------------------

# Phrases that assert a within-match timeline or an event. Matched
# case-insensitively against the narrative.
_CHRONOLOGY_PHRASES = (
    "first half",
    "second half",
    "as the match progressed",
    "later in the match",
    "early in the match",
    "opening minutes",
    "closing stages",
    "by full time",
    "after the break",
)

_EVENT_PHRASES = (
    "scored",
    "conceded",
    "breakthrough",
    "substitution",
    "we saw a shift",
)


def _forbidden_phrases(narrative: str, phrases) -> list[str]:
    lowered = narrative.lower()
    return [phrase for phrase in phrases if phrase in lowered]


def test_the_prompt_forbids_inventing_match_chronology_and_events() -> None:
    """The rules have to be IN the prompt; a live model cannot be asserted
    on in a unit test, so the contract is checked at the prompt boundary."""
    from nexus.sports.coach import _SYSTEM_PROMPT

    lowered = _SYSTEM_PROMPT.lower()
    assert "no chronology" in lowered or "no match events" in lowered
    assert "halves" in lowered
    assert "as the match progressed" in lowered
    assert "phase 4" in lowered
    assert "prospective instructions for the next match" in lowered


@pytest.mark.asyncio
async def test_a_narrative_inventing_a_second_half_is_detected() -> None:
    """Guard on the detector itself: it must flag the exact drift a live
    qwen2.5:3b produced, or it proves nothing."""
    drifted = (
        "In the second half, the team's pressing became more aggressive, "
        "leading to some breakthroughs. As the match progressed, they read "
        "the game better."
    )
    provider = _RecordingProvider(drifted)
    coach = CoachAssistant(_FakeRouter(provider), _FakeAdapter(_rich_analysis()))

    report = await coach.build_report("m-rich")

    assert _forbidden_phrases(report.narrative, _CHRONOLOGY_PHRASES)
    assert _forbidden_phrases(report.narrative, _EVENT_PHRASES)


@pytest.mark.asyncio
async def test_a_clean_narrative_trips_neither_detector() -> None:
    clean = (
        "Their block measured compact and their shape unstable. Attack the "
        "unstable shape by switching play early. Pressing intensity could "
        "not be measured."
    )
    provider = _RecordingProvider(clean)
    coach = CoachAssistant(_FakeRouter(provider), _FakeAdapter(_rich_analysis()))

    report = await coach.build_report("m-rich")

    assert not _forbidden_phrases(report.narrative, _CHRONOLOGY_PHRASES)
    assert not _forbidden_phrases(report.narrative, _EVENT_PHRASES)


# --------------------------------------------------------------------------
# 5. The narrative guard ENFORCES what the prompt only asks for
# --------------------------------------------------------------------------

from nexus.sports.narrative_guard import audit_narrative  # noqa: E402


class _SequenceProvider(_RecordingProvider):
    """Returns a scripted sequence, so a corrective retry can be observed."""

    def __init__(self, narratives: list[str]) -> None:
        super().__init__()
        self._narratives = list(narratives)
        self.call_count = 0
        self.all_prompts: list[str] = []

    async def generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        self.received_messages = messages
        self.all_prompts.append("\n".join(m.content for m in messages))
        content = self._narratives[min(self.call_count, len(self._narratives) - 1)]
        self.call_count += 1
        return GenerationResult(
            content=content, model_used=model_id, provider_name=self.name, usage=Usage()
        )


def test_audit_flags_chronology_without_a_timeline() -> None:
    audit = audit_narrative("In the first half they pressed high.", timeline=None)
    assert not audit.is_clean
    assert "first half" in audit.chronology_claims


def test_audit_allows_chronology_when_a_timeline_supplied_phases() -> None:
    """The guard bans UNSUPPORTED time claims, not time itself."""
    timeline = parse_timeline(_timeline_payload(), match_id="m-rich")
    audit = audit_narrative("In the first half they pressed high.", timeline=timeline)
    assert not audit.chronology_claims


def test_audit_flags_events_even_with_a_timeline() -> None:
    """No timeline section reports goals or injuries, so those are always
    unsupported."""
    timeline = parse_timeline(_timeline_payload(), match_id="m-rich")
    audit = audit_narrative("Their striker scored after an injury.", timeline=timeline)
    assert "goal scored" in audit.event_claims
    assert "injury" in audit.event_claims


def test_audit_passes_a_grounded_narrative() -> None:
    audit = audit_narrative(
        "Their block measured compact; pressing intensity was not measurable.",
        timeline=None,
    )
    assert audit.is_clean
    assert audit.warnings == []


@pytest.mark.asyncio
async def test_an_unsupported_narrative_triggers_one_corrective_retry() -> None:
    provider = _SequenceProvider(
        ["In the second half they scored.", "Their block measured compact."]
    )
    coach = CoachAssistant(_FakeRouter(provider), _FakeAdapter(_rich_analysis()))

    report = await coach.build_report("m-rich")

    assert provider.call_count == 2, "expected exactly one corrective retry"
    assert report.narrative == "Their block measured compact."
    assert report.narrative_warnings == [], "the clean retry should clear the warnings"
    assert "was invented" in provider.all_prompts[1]


@pytest.mark.asyncio
async def test_a_still_unsupported_retry_is_returned_flagged_not_hidden() -> None:
    """Two bad attempts must not silently ship as fact."""
    provider = _SequenceProvider(
        ["In the second half they scored.", "In the first half they scored again."]
    )
    coach = CoachAssistant(_FakeRouter(provider), _FakeAdapter(_rich_analysis()))

    report = await coach.build_report("m-rich")

    assert provider.call_count == 2
    assert report.narrative_warnings, "an unsupported narrative must be flagged"
    assert any("chronology" in w for w in report.narrative_warnings)


@pytest.mark.asyncio
async def test_a_clean_first_attempt_costs_no_extra_generation() -> None:
    provider = _SequenceProvider(["Their block measured compact."])
    coach = CoachAssistant(_FakeRouter(provider), _FakeAdapter(_rich_analysis()))

    report = await coach.build_report("m-rich")

    assert provider.call_count == 1, "a clean narrative must not trigger a retry"
    assert report.narrative_warnings == []
