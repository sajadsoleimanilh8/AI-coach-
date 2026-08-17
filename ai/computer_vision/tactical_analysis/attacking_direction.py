"""Which way is each team attacking?"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ai.computer_vision.tactical_analysis.constants import PITCH_LENGTH_M

LEFT_TO_RIGHT = "left_to_right"
RIGHT_TO_LEFT = "right_to_left"
UNKNOWN = "unknown"

MIN_TEAM_SEPARATION_M = 5.0

MIN_POINTS_PER_TEAM = 30


@dataclass
class AttackingDirectionResult:
    """Per-team direction plus the evidence it was derived from."""

    by_team: dict[str, str] = field(default_factory=dict)
    separation_m: float | None = None
    mean_x_by_team: dict[str, float] = field(default_factory=dict)
    sample_by_team: dict[str, int] = field(default_factory=dict)
    reason: str = ""
    spans_possible_half_boundary: bool = False
    method: str = "heuristic_proxy"

    @property
    def resolved(self) -> bool:
        return bool(self.by_team) and all(d != UNKNOWN for d in self.by_team.values())

    @property
    def confidence(self) -> str:
        """A MetricConfidence value, for callers writing metric rows."""
        if not self.resolved:
            return "low_upstream_confidence"
        if self.spans_possible_half_boundary:
            return "low_upstream_confidence"
        if min(self.sample_by_team.values(), default=0) < MIN_POINTS_PER_TEAM * 4:
            return "low_sample"
        return "normal"

    def for_team(self, team_id: str | None) -> str:
        """Direction for one team, or UNKNOWN. Callers must handle UNKNOWN
        by declining to score direction-dependent metrics -- never by
        falling back to a default direction."""
        if team_id is None:
            return UNKNOWN
        return self.by_team.get(str(team_id), UNKNOWN)


def infer_attacking_directions(trajectories: dict) -> AttackingDirectionResult:
    """Infers each team's attacking direction from tracked pitch positions."""
    xs_by_team: dict[str, list[float]] = defaultdict(list)
    halves: dict[str, list[list[float]]] = defaultdict(lambda: [[], []])

    frame_ids = [p.frame_id for pts in trajectories.values() for p in pts
                 if p.pitch_x_m is not None and p.team_id is not None]
    midpoint = (min(frame_ids) + max(frame_ids)) / 2.0 if frame_ids else 0.0

    for points in trajectories.values():
        for p in points:
            if p.pitch_x_m is None or p.team_id is None:
                continue
            tid = str(p.team_id)
            xs_by_team[tid].append(float(p.pitch_x_m))
            halves[tid][0 if p.frame_id <= midpoint else 1].append(float(p.pitch_x_m))

    result = AttackingDirectionResult(
        sample_by_team={t: len(v) for t, v in xs_by_team.items()},
        mean_x_by_team={t: sum(v) / len(v) for t, v in xs_by_team.items() if v},
    )

    usable = {t: v for t, v in xs_by_team.items() if len(v) >= MIN_POINTS_PER_TEAM}
    if len(usable) < 2:
        result.by_team = {t: UNKNOWN for t in xs_by_team}
        result.reason = (
            f"need two teams with >={MIN_POINTS_PER_TEAM} positioned points; "
            f"got {{{', '.join(f'{t}:{len(v)}' for t, v in sorted(xs_by_team.items()))}}}. "
            "Most likely cause: calibration invalid, so no pitch coordinates exist."
        )
        return result

    means = {t: sum(v) / len(v) for t, v in usable.items()}
    ordered = sorted(means, key=lambda t: means[t])
    left_team, right_team = ordered[0], ordered[-1]
    separation = means[right_team] - means[left_team]
    result.separation_m = round(separation, 3)

    if separation < MIN_TEAM_SEPARATION_M:
        result.by_team = {t: UNKNOWN for t in xs_by_team}
        result.reason = (
            f"teams' mean x differ by only {separation:.1f} m "
            f"(<{MIN_TEAM_SEPARATION_M} m); ordering is not distinguishable from noise"
        )
        return result

    result.by_team = {t: UNKNOWN for t in xs_by_team}
    result.by_team[left_team] = LEFT_TO_RIGHT
    result.by_team[right_team] = RIGHT_TO_LEFT
    result.reason = (
        f"mean x: {left_team}={means[left_team]:.1f} m, {right_team}={means[right_team]:.1f} m "
        f"on a {PITCH_LENGTH_M:.0f} m pitch (separation {separation:.1f} m)"
    )

    order_flipped = []
    for idx in (0, 1):
        a = halves[left_team][idx]
        b = halves[right_team][idx]
        if len(a) >= MIN_POINTS_PER_TEAM and len(b) >= MIN_POINTS_PER_TEAM:
            order_flipped.append((sum(a) / len(a)) > (sum(b) / len(b)))
    if len(order_flipped) == 2 and order_flipped[0] != order_flipped[1]:
        result.spans_possible_half_boundary = True
        result.reason += (
            "; WARNING the teams' x-ordering reverses between the first and "
            "second half of this clip -- possibly a switch of ends, possibly "
            "sustained one-way pressure. Not disambiguated (no half metadata)."
        )
    return result
