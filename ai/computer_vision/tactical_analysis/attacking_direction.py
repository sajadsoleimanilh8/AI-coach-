"""Which way is each team attacking?

REPLACES a hardcoded global. `shot_heuristics.detect_shots()` and
`runner._compute_passing_vision_inputs()` both assumed `left_to_right` for
every team for the whole match. That is wrong for one of the two teams by
construction -- they attack opposite goals -- and wrong for both after the
half-time switch of ends.

WHAT THIS CAN AND CANNOT DO
    It infers direction from where each team's tracked players actually
    are, in pitch metres. That requires a valid calibration, so on footage
    where `calibration.valid` is False for every frame (the current state
    on real broadcast clips -- see docs/pipeline_architecture.md 6.3) this
    returns `unknown` for both teams and the callers degrade rather than
    guess. That is the intended behaviour, not a gap to paper over.

    PER-HALF DETECTION IS NOT IMPLEMENTED, deliberately. Nothing in the
    schema records a half boundary: `Match` has no period/kickoff columns
    and an uploaded clip carries no timeline metadata, so there is no
    evidence from which to locate a switch of ends. Inferring one from a
    mid-clip flip in mean position would be indistinguishable from a
    sustained period of one-way pressure, which is a normal thing for
    football to do. The honest scope is therefore ONE direction per team
    per clip, reported with the sample it was measured from; a clip that
    genuinely spans half-time will be wrong for part of its length, and
    `spans_possible_half_boundary` flags when the evidence looks like that
    may have happened so a caller can refuse the reading.

METHOD
    heuristic_proxy, always. A team's mean x is a proxy for which end it
    defends, not a measurement of intent.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ai.computer_vision.tactical_analysis.constants import PITCH_LENGTH_M

LEFT_TO_RIGHT = "left_to_right"
RIGHT_TO_LEFT = "right_to_left"
UNKNOWN = "unknown"

#: Minimum gap between the two teams' mean x, in metres, before the
#: ordering is treated as real. Two teams that genuinely defend opposite
#: ends separate by tens of metres in their mean position over any decent
#: sample; a gap of a few metres is noise, and calling it either way would
#: invert half the forward-pass counts in the match.
MIN_TEAM_SEPARATION_M = 5.0

#: Minimum positioned points per team before the mean is trusted at all.
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
    """Infers each team's attacking direction from tracked pitch positions.

    The team whose players sit on average nearer x=0 is defending the left
    goal and therefore attacking left-to-right. Only points with a real
    `pitch_x_m` count, so frames where calibration was invalid contribute
    nothing rather than contributing a fabricated coordinate.
    """
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
        result.by_team = dict.fromkeys(xs_by_team, UNKNOWN)
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
        result.by_team = dict.fromkeys(xs_by_team, UNKNOWN)
        result.reason = (
            f"teams' mean x differ by only {separation:.1f} m "
            f"(<{MIN_TEAM_SEPARATION_M} m); ordering is not distinguishable from noise"
        )
        return result

    # The team sitting nearer x=0 defends the left goal, so it attacks
    # towards x=PITCH_LENGTH_M.
    result.by_team = dict.fromkeys(xs_by_team, UNKNOWN)
    result.by_team[left_team] = LEFT_TO_RIGHT
    result.by_team[right_team] = RIGHT_TO_LEFT
    result.reason = (
        f"mean x: {left_team}={means[left_team]:.1f} m, {right_team}={means[right_team]:.1f} m "
        f"on a {PITCH_LENGTH_M:.0f} m pitch (separation {separation:.1f} m)"
    )

    # Half-boundary smell test: if the two teams' ordering reverses between
    # the first and second half of the CLIP, either the ends switched or
    # play swung heavily. Either way the single direction reported here is
    # not safe to use, so say so rather than picking one.
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
