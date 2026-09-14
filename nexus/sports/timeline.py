"""Phase 4's tactical timeline, as consumed by the coach.

Phase 4 is not implemented in this repo yet: `ai/team_intelligence/
{possession,transition}_analysis/` are empty and no backend route serves a
timeline. This module defines the shape the coach expects and parses it when
the backend starts serving it, so the coaching path is wired now and gains
phase/pressing/transition/territory context the moment Phase 4 lands.

Until then `parse_timeline` returns None and the coach reports the timeline
as unavailable -- which is the honest answer, and specifically NOT the same
as reporting a match with no pressing or no transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# The route the adapter probes. A 404 here is expected pre-Phase-4.
TIMELINE_PATH = "/api/tactical/timeline/{match_id}"

# Every field the coach will quote, so absence is reported per-section
# rather than as one opaque "no timeline".
TIMELINE_SECTIONS: tuple[str, ...] = ("phases", "pressing", "transitions", "territory")


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    return float(value) if isinstance(value, (int, float)) else None


@dataclass(frozen=True)
class PhaseSegment:
    """One contiguous stretch of the match in a single game phase."""

    phase: str
    start_second: float
    end_second: float
    team_id: str | None = None

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_second - self.start_second)


@dataclass
class PhaseBreakdown:
    segments: list[PhaseSegment] = field(default_factory=list)
    share_by_phase: dict[str, float] = field(default_factory=dict)
    confidence: str = "normal"
    sample_size: int = 0

    @property
    def is_available(self) -> bool:
        return bool(self.share_by_phase) or bool(self.segments)

    def as_lines(self) -> list[str]:
        if not self.is_available:
            return []
        lines = [
            f"phase_share[{phase}]={share:.3f}"
            for phase, share in sorted(self.share_by_phase.items(), key=lambda kv: -kv[1])
        ]
        if self.segments:
            lines.append(f"phase_segments_counted={len(self.segments)}")
        return lines


@dataclass
class PressingSummary:
    """Pressing measured over time rather than as one match aggregate."""

    intensity_by_phase: dict[str, float] = field(default_factory=dict)
    success_rate: float | None = None
    high_press_share: float | None = None
    confidence: str = "normal"
    sample_size: int = 0

    @property
    def is_available(self) -> bool:
        return bool(self.intensity_by_phase) or self.success_rate is not None

    def as_lines(self) -> list[str]:
        lines = [
            f"pressing_intensity[{phase}]={value:.1f}"
            for phase, value in sorted(self.intensity_by_phase.items())
        ]
        if self.success_rate is not None:
            lines.append(f"pressing_success_rate={self.success_rate:.3f}")
        if self.high_press_share is not None:
            lines.append(f"high_press_share={self.high_press_share:.3f}")
        return lines


@dataclass
class TransitionSummary:
    """Turnovers and what happened in the seconds after them."""

    count: int = 0
    mean_speed_mps: float | None = None
    mean_duration_seconds: float | None = None
    recovery_time_seconds: float | None = None
    confidence: str = "normal"
    sample_size: int = 0

    @property
    def is_available(self) -> bool:
        return self.count > 0 or self.mean_speed_mps is not None

    def as_lines(self) -> list[str]:
        lines = [f"transition_count={self.count}"]
        if self.mean_speed_mps is not None:
            lines.append(f"transition_mean_speed_mps={self.mean_speed_mps:.2f}")
        if self.mean_duration_seconds is not None:
            lines.append(f"transition_mean_duration_s={self.mean_duration_seconds:.2f}")
        if self.recovery_time_seconds is not None:
            lines.append(f"transition_recovery_time_s={self.recovery_time_seconds:.2f}")
        return lines


@dataclass
class TerritorySummary:
    """Where on the pitch the match was actually played."""

    share_by_third: dict[str, float] = field(default_factory=dict)
    field_tilt: float | None = None
    mean_block_height_m: float | None = None
    confidence: str = "normal"
    sample_size: int = 0

    @property
    def is_available(self) -> bool:
        return bool(self.share_by_third) or self.field_tilt is not None

    def as_lines(self) -> list[str]:
        lines = [
            f"territory_share[{third}]={share:.3f}"
            for third, share in sorted(self.share_by_third.items())
        ]
        if self.field_tilt is not None:
            lines.append(f"field_tilt={self.field_tilt:.3f}")
        if self.mean_block_height_m is not None:
            lines.append(f"mean_block_height_m={self.mean_block_height_m:.1f}")
        return lines


@dataclass
class TacticalTimeline:
    """Phase 4's per-match timeline. Each section stands or falls alone."""

    match_id: str
    phases: PhaseBreakdown = field(default_factory=PhaseBreakdown)
    pressing: PressingSummary = field(default_factory=PressingSummary)
    transitions: TransitionSummary = field(default_factory=TransitionSummary)
    territory: TerritorySummary = field(default_factory=TerritorySummary)
    team_id: str | None = None
    schema_version: str = ""

    @property
    def available_sections(self) -> list[str]:
        return [s for s in TIMELINE_SECTIONS if getattr(self, s).is_available]

    @property
    def missing_sections(self) -> list[str]:
        return [s for s in TIMELINE_SECTIONS if not getattr(self, s).is_available]

    @property
    def is_available(self) -> bool:
        return bool(self.available_sections)

    def as_prompt_context(self) -> str:
        """Every timeline number the LLM is allowed to quote, one per line."""
        blocks: list[str] = []
        for section in TIMELINE_SECTIONS:
            lines = getattr(self, section).as_lines()
            if lines:
                blocks.append(f"{section}:\n" + "\n".join(f"  {line}" for line in lines))
        if not blocks:
            return "No tactical timeline sections were available."
        return "\n".join(blocks)


def _phase_breakdown(raw: dict[str, Any]) -> PhaseBreakdown:
    segments = [
        PhaseSegment(
            phase=str(item.get("phase", "unknown")),
            start_second=_numeric(item.get("start_second")) or 0.0,
            end_second=_numeric(item.get("end_second")) or 0.0,
            team_id=item.get("team_id"),
        )
        for item in raw.get("segments") or []
        if isinstance(item, dict)
    ]

    share = {
        str(k): v
        for k, raw_value in (raw.get("share_by_phase") or {}).items()
        if (v := _numeric(raw_value)) is not None
    }
    # Derive shares from segments only when the payload did not state them --
    # a served value always wins over one we compute here.
    if not share and segments:
        total = sum(s.duration_seconds for s in segments)
        if total > 0:
            by_phase: dict[str, float] = {}
            for segment in segments:
                by_phase[segment.phase] = by_phase.get(segment.phase, 0.0) + segment.duration_seconds
            share = {phase: value / total for phase, value in by_phase.items()}

    return PhaseBreakdown(
        segments=segments,
        share_by_phase=share,
        confidence=str(raw.get("confidence", "normal")),
        sample_size=int(_numeric(raw.get("sample_size")) or len(segments)),
    )


def _pressing(raw: dict[str, Any]) -> PressingSummary:
    return PressingSummary(
        intensity_by_phase={
            str(k): v
            for k, raw_value in (raw.get("intensity_by_phase") or {}).items()
            if (v := _numeric(raw_value)) is not None
        },
        success_rate=_numeric(raw.get("success_rate")),
        high_press_share=_numeric(raw.get("high_press_share")),
        confidence=str(raw.get("confidence", "normal")),
        sample_size=int(_numeric(raw.get("sample_size")) or 0),
    )


def _transitions(raw: dict[str, Any]) -> TransitionSummary:
    events = raw.get("events") or []
    return TransitionSummary(
        count=int(_numeric(raw.get("count")) or len(events)),
        mean_speed_mps=_numeric(raw.get("mean_speed_mps")),
        mean_duration_seconds=_numeric(raw.get("mean_duration_seconds")),
        recovery_time_seconds=_numeric(raw.get("recovery_time_seconds")),
        confidence=str(raw.get("confidence", "normal")),
        sample_size=int(_numeric(raw.get("sample_size")) or len(events)),
    )


def _territory(raw: dict[str, Any]) -> TerritorySummary:
    return TerritorySummary(
        share_by_third={
            str(k): v
            for k, raw_value in (raw.get("share_by_third") or {}).items()
            if (v := _numeric(raw_value)) is not None
        },
        field_tilt=_numeric(raw.get("field_tilt")),
        mean_block_height_m=_numeric(raw.get("mean_block_height_m")),
        confidence=str(raw.get("confidence", "normal")),
        sample_size=int(_numeric(raw.get("sample_size")) or 0),
    )


def parse_timeline(payload: Any, *, match_id: str) -> TacticalTimeline | None:
    """Parse a Phase 4 timeline payload; None when there is nothing usable.

    Tolerant by design: a partial payload yields a timeline with only the
    sections that were actually served, because a coach is better off with
    three measured sections and one honest gap than with nothing.
    """
    if not isinstance(payload, dict):
        return None

    timeline = TacticalTimeline(
        match_id=str(payload.get("match_id") or match_id),
        phases=_phase_breakdown(payload.get("phases") or {}),
        pressing=_pressing(payload.get("pressing") or {}),
        transitions=_transitions(payload.get("transitions") or {}),
        territory=_territory(payload.get("territory") or {}),
        team_id=payload.get("team_id"),
        schema_version=str(payload.get("schema_version", "")),
    )
    return timeline if timeline.is_available else None


def timeline_unavailable_reasons(timeline: TacticalTimeline | None) -> list[str]:
    """One line per timeline section the coach could not read."""
    if timeline is None:
        return [
            f"tactical_timeline.{section}: unavailable -- no tactical timeline is "
            "served for this match (Phase 4 timeline not computed)"
            for section in TIMELINE_SECTIONS
        ]
    return [
        f"tactical_timeline.{section}: unavailable -- the timeline was served but "
        "this section was empty"
        for section in timeline.missing_sections
    ]
