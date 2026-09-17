"""Transparent what-if simulation over metrics the pipeline really computed.

THIS IS NOT REINFORCEMENT LEARNING, and it is not a learned model of any
kind. There is no policy, no reward, no rollout and no training step. It is
a deterministic recomputation: take the real tracked positions this match
produced, apply one explicitly-parameterised geometric or kinematic change
to them, and run the SAME scoring functions the pipeline runs
(`compute_compactness`, `detect_formation`, `compute_formation_stability`,
`compute_weak_zones`) over the modified input.

Why that constraint matters: a simulator that emits plausible-looking
numbers from its own model would be indistinguishable, in the API response,
from measured output. Here every simulated value is the output of the same
function that produced the baseline, over inputs that differ in exactly one
declared way -- so `SimulatedMetric` can name the real input it came from
and the parameter that changed, and a reader can check it.

WHAT IT CANNOT DO
    It does not predict match outcomes, goals, or xG. Moving players 10%
    closer together tells you what the compactness metric would read; it
    does not tell you the team would concede less. The causal claim is not
    available from this data and is not made -- see `caveats` on every
    result.

    It inherits every upstream limitation. If `calibration.valid` was False
    there are no pitch coordinates, the baseline metrics are already
    gated to None, and the simulation returns None for them too rather
    than inventing a coordinate space to move players around in.

METHOD
    Every simulated metric carries `method="heuristic_proxy"` regardless of
    what the underlying metric function reports. The recomputation is
    exact, but the INTERVENTION -- "what if this team were 10% more
    compact" -- is a hypothetical, and a hypothetical recomputed exactly is
    still a hypothetical. Reporting `deterministic` here would claim more
    than is true.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai.computer_vision.tactical_analysis.formation_detection import detect_formation
from ai.team_intelligence.formation_stability.team_shape import (
    compute_compactness,
    compute_formation_stability,
)
from ai.team_intelligence.weak_zone_detection.weak_zones import compute_weak_zones
from backend.database.models import MetricConfidence, MetricMethod

SIMULATION_METHOD = MetricMethod.heuristic_proxy

#: Interventions this engine accepts. Anything else is rejected loudly --
#: silently ignoring an unknown knob would report the baseline as if it
#: were a simulation result.
INTERVENTION_KINDS = (
    "compactness",        # scale distance from the team centroid
    "transition_speed",   # scale per-frame displacement (and thus speed)
    "remove_player",      # take one tracked player out of the shape
    "swap_player",        # give player A player B's tracked positions
)


class SimulationError(ValueError):
    """Raised for an intervention that cannot be honoured. Never downgraded
    to a silent no-op."""


@dataclass
class Intervention:
    """One declared change. `pct` is a percentage delta: +10.0 means "10%
    more compact"/"10% faster", -25.0 means the reverse."""

    kind: str
    team_id: str | None = None
    player_id: int | None = None
    other_player_id: int | None = None
    pct: float | None = None

    def describe(self) -> str:
        if self.kind == "compactness":
            return f"team {self.team_id} compactness {self.pct:+.1f}%"
        if self.kind == "transition_speed":
            return f"team {self.team_id} transition speed {self.pct:+.1f}%"
        if self.kind == "remove_player":
            return f"remove player {self.player_id} from team {self.team_id}"
        if self.kind == "swap_player":
            return f"player {self.player_id} takes player {self.other_player_id}'s tracked movement"
        return f"{self.kind}({self.team_id},{self.player_id},{self.pct})"

    def validate(self) -> None:
        if self.kind not in INTERVENTION_KINDS:
            raise SimulationError(
                f"unknown intervention kind {self.kind!r}; supported: {list(INTERVENTION_KINDS)}")
        if self.kind in ("compactness", "transition_speed"):
            if self.pct is None:
                raise SimulationError(f"{self.kind} requires `pct`")
            if self.pct <= -100.0:
                raise SimulationError(f"{self.kind} pct must be > -100 (got {self.pct})")
        if self.kind == "remove_player" and self.player_id is None:
            raise SimulationError("remove_player requires `player_id`")
        if self.kind == "swap_player" and (self.player_id is None or self.other_player_id is None):
            raise SimulationError("swap_player requires `player_id` and `other_player_id`")


@dataclass
class SimulatedMetric:
    """One metric, before and after, with its provenance."""

    metric_name: str
    team_id: str
    baseline_value: Any
    simulated_value: Any
    delta: float | None
    derived_from: str          # the real input this was recomputed from
    parameter_changed: str     # the declared intervention
    recomputed_by: str         # the actual function that produced it
    method: MetricMethod = SIMULATION_METHOD
    confidence: MetricConfidence = MetricConfidence.normal
    source_metric_id: str | None = None
    source_method: MetricMethod | None = None
    source_confidence: MetricConfidence | None = None
    baseline_sub_scores: dict = field(default_factory=dict)
    simulated_sub_scores: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "team_id": self.team_id,
            "baseline_value": self.baseline_value,
            "simulated_value": self.simulated_value,
            "delta": self.delta,
            "derived_from": self.derived_from,
            "parameter_changed": self.parameter_changed,
            "recomputed_by": self.recomputed_by,
            "method": self.method.value,
            "confidence": self.confidence.value,
            "source_metric_id": self.source_metric_id,
            "source_method": self.source_method.value if self.source_method else None,
            "source_confidence": self.source_confidence.value if self.source_confidence else None,
            "baseline_sub_scores": self.baseline_sub_scores,
            "simulated_sub_scores": self.simulated_sub_scores,
        }


@dataclass
class SimulationResult:
    interventions: list[str]
    metrics: list[SimulatedMetric]
    caveats: list[str]
    unavailable: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "interventions": self.interventions,
            "metrics": [m.as_dict() for m in self.metrics],
            "caveats": self.caveats,
            "unavailable": self.unavailable,
            "is_reinforcement_learning": False,
            "method": SIMULATION_METHOD.value,
        }


# ----------------------------------------------------------------------
# Position transforms -- the only place the input is altered
# ----------------------------------------------------------------------

def _positioned(points) -> list:
    return [p for p in points if getattr(p, "pitch_x_m", None) is not None]


def _centroid(team_traj: dict) -> tuple[float, float] | None:
    pts = [(p.pitch_x_m, p.pitch_y_m) for pts_ in team_traj.values() for p in _positioned(pts_)]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


@dataclass
class _Pt:
    """A minimal stand-in carrying only what the scorers read. The real
    TrackingPoint is not mutated -- the baseline must stay measurable after
    a simulation runs."""
    player_id: int
    frame_id: int
    team_id: str | None
    pitch_x_m: float | None
    pitch_y_m: float | None
    speed: float | None = None


def _copy_traj(team_traj: dict) -> dict:
    return {
        pid: [_Pt(p.player_id, p.frame_id, p.team_id, p.pitch_x_m, p.pitch_y_m,
                  getattr(p, "speed", None)) for p in pts]
        for pid, pts in team_traj.items()
    }


def _apply(team_traj: dict, iv: Intervention) -> dict:
    """Returns a modified COPY of one team's trajectories."""
    traj = _copy_traj(team_traj)

    if iv.kind == "compactness":
        c = _centroid(team_traj)
        if c is None:
            return traj
        # +10% compactness => every player 10% closer to the centroid. The
        # convex hull shrinks accordingly, which is exactly what
        # compute_compactness measures, so the recomputed score moves for
        # the same geometric reason the real one would.
        scale = 1.0 - (iv.pct / 100.0)
        for pts in traj.values():
            for p in pts:
                if p.pitch_x_m is None:
                    continue
                p.pitch_x_m = c[0] + (p.pitch_x_m - c[0]) * scale
                p.pitch_y_m = c[1] + (p.pitch_y_m - c[1]) * scale

    elif iv.kind == "transition_speed":
        # Scale each player's frame-to-frame displacement about their own
        # first positioned point: the shape is preserved, the distance
        # covered between frames is not. Formation STABILITY reads
        # between-frame movement, so that is the metric this moves.
        factor = 1.0 + (iv.pct / 100.0)
        for pts in traj.values():
            positioned = _positioned(pts)
            if not positioned:
                continue
            ox, oy = positioned[0].pitch_x_m, positioned[0].pitch_y_m
            for p in pts:
                if p.pitch_x_m is None:
                    continue
                p.pitch_x_m = ox + (p.pitch_x_m - ox) * factor
                p.pitch_y_m = oy + (p.pitch_y_m - oy) * factor
                if p.speed is not None:
                    p.speed = p.speed * factor

    elif iv.kind == "remove_player":
        if iv.player_id not in traj:
            raise SimulationError(
                f"player {iv.player_id} is not tracked in team {iv.team_id}; "
                f"tracked: {sorted(traj)}")
        traj.pop(iv.player_id)

    elif iv.kind == "swap_player":
        if iv.player_id not in traj or iv.other_player_id not in traj:
            raise SimulationError(
                f"swap_player needs both players tracked in team {iv.team_id}; "
                f"tracked: {sorted(traj)}")
        # A really takes B's movement: this is the "what if we played
        # someone with this player's actual movement profile here" case,
        # built from B's REAL tracked positions rather than a synthesised
        # profile.
        donor = traj[iv.other_player_id]
        traj[iv.player_id] = [_Pt(iv.player_id, p.frame_id, p.team_id,
                                  p.pitch_x_m, p.pitch_y_m, p.speed) for p in donor]

    return traj


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------

def _positions(team_traj: dict) -> list[tuple[float, float]]:
    return [(p.pitch_x_m, p.pitch_y_m) for pts in team_traj.values() for p in _positioned(pts)]


def _mean_per_player(team_traj: dict) -> list[tuple[float, float]]:
    out = []
    for pts in team_traj.values():
        v = _positioned(pts)
        if v:
            out.append((sum(p.pitch_x_m for p in v) / len(v),
                        sum(p.pitch_y_m for p in v) / len(v)))
    return out


def _by_frame(team_traj: dict) -> list[list[tuple[float, float]]]:
    frames: dict[int, list[tuple[float, float]]] = {}
    for pts in team_traj.values():
        for p in _positioned(pts):
            frames.setdefault(p.frame_id, []).append((p.pitch_x_m, p.pitch_y_m))
    return [frames[f] for f in sorted(frames)]


def _score(team_traj: dict, opponent_traj: dict, confidence: float,
           attacking_direction: str | None) -> dict[str, dict]:
    """Runs the real scoring functions. Same functions, same arguments the
    pipeline uses -- this is what makes a simulated number comparable to a
    measured one."""
    kwargs = {}
    if attacking_direction in ("left_to_right", "right_to_left"):
        kwargs["attacking_direction"] = attacking_direction
    opp = _positions(opponent_traj) if opponent_traj else None
    return {
        "formation": detect_formation(
            player_positions=_mean_per_player(team_traj),
            team_assignment_confidence=confidence, **kwargs),
        "compactness_score": compute_compactness(
            _positions(team_traj), team_assignment_confidence=confidence),
        "formation_stability_score": compute_formation_stability(
            _by_frame(team_traj), team_assignment_confidence=confidence),
        "weak_zone_map": compute_weak_zones(
            _positions(team_traj), team_assignment_confidence=confidence,
            opponent_positions_m=opp or None),
    }


_RECOMPUTED_BY = {
    "formation": "ai.computer_vision.tactical_analysis.formation_detection.detect_formation",
    "compactness_score": "ai.team_intelligence.formation_stability.team_shape.compute_compactness",
    "formation_stability_score": "ai.team_intelligence.formation_stability.team_shape.compute_formation_stability",
    "weak_zone_map": "ai.team_intelligence.weak_zone_detection.weak_zones.compute_weak_zones",
}

#: Which metrics an intervention can actually move. Reporting a metric the
#: intervention cannot affect invites reading noise as signal.
_AFFECTED = {
    "compactness": ("compactness_score", "formation", "weak_zone_map"),
    "transition_speed": ("formation_stability_score", "compactness_score"),
    "remove_player": ("compactness_score", "formation", "formation_stability_score", "weak_zone_map"),
    "swap_player": ("compactness_score", "formation", "formation_stability_score", "weak_zone_map"),
}


def simulate(
    trajectories_by_team: dict[str, dict],
    interventions: list[Intervention],
    team_assignment_confidence: float,
    directions=None,
    baseline_provenance: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> SimulationResult:
    """Recomputes team metrics under the given interventions.

    Args:
        trajectories_by_team: {team_id: {player_id: [TrackingPoint, ...]}},
            i.e. the output of runner._split_by_team().
        interventions: the declared changes. All are applied to the same
            baseline, then scored once -- not chained.
        team_assignment_confidence: passed straight through to the real
            scorers so their gates behave exactly as in the pipeline.
        directions: optional AttackingDirectionResult, for formation
            orientation.
    """
    for iv in interventions:
        iv.validate()

    caveats = [
        "Not reinforcement learning and not a learned model: every value below is "
        "the output of the same scoring function used on the real match, over "
        "positions altered in exactly the declared way.",
        "A metric moving does not mean the outcome would change. No causal or "
        "outcome claim (goals, xG, result) is made or supported by this data.",
    ]
    metrics: list[SimulatedMetric] = []
    unavailable: list[str] = []

    if not trajectories_by_team:
        unavailable.append(
            "no team-split trajectories were available, so there is no measured "
            "baseline to simulate from (usual cause: calibration invalid, so no "
            "pitch coordinates exist)")
        return SimulationResult([iv.describe() for iv in interventions], [], caveats, unavailable)

    for iv in interventions:
        target = iv.team_id
        if target is None and iv.kind in ("swap_player", "remove_player"):
            target = next((t for t, tr in trajectories_by_team.items()
                           if iv.player_id in tr), None)
        if target not in trajectories_by_team:
            raise SimulationError(
                f"intervention targets team {iv.team_id!r}, which has no tracked "
                f"players; available: {sorted(trajectories_by_team)}")

        own = trajectories_by_team[target]
        opponent = {pid: pts for t, tr in trajectories_by_team.items() if t != target
                    for pid, pts in tr.items()}
        direction = directions.for_team(target) if directions is not None else None

        baseline = _score(own, opponent, team_assignment_confidence, direction)
        modified = _apply(own, iv)
        simulated = _score(modified, opponent, team_assignment_confidence, direction)

        for name in _AFFECTED[iv.kind]:
            b, s = baseline[name], simulated[name]
            bv, sv = b.get("value"), s.get("value")
            delta = None
            if isinstance(bv, (int, float)) and isinstance(sv, (int, float)):
                delta = round(float(sv) - float(bv), 4)

            if bv is None:
                unavailable.append(
                    f"{name} for team {target}: baseline is unavailable "
                    f"({b.get('confidence')}), so no simulated value is reported")

            source = (baseline_provenance or {}).get(str(target), {}).get(name, {})
            simulated_confidence = (
                MetricConfidence(s.get("confidence", MetricConfidence.normal.value))
                if bv is not None else MetricConfidence.low_upstream_confidence
            )
            metrics.append(SimulatedMetric(
                metric_name=name,
                team_id=str(target),
                baseline_value=bv,
                # An unavailable baseline yields an unavailable simulation.
                # Reporting a number here would be the one thing this
                # engine exists to avoid.
                simulated_value=sv if bv is not None else None,
                delta=delta if bv is not None else None,
                derived_from=(
                    f"real tracked pitch positions of {len(own)} player(s) on team "
                    f"{target} ({sum(len(_positioned(p)) for p in own.values())} positioned samples)"),
                parameter_changed=iv.describe(),
                recomputed_by=_RECOMPUTED_BY[name],
                confidence=simulated_confidence,
                source_metric_id=source.get("metric_id"),
                source_method=(MetricMethod(source["method"]) if source.get("method") else None),
                source_confidence=(MetricConfidence(source["confidence"])
                                   if source.get("confidence") else None),
                baseline_sub_scores=b.get("sub_scores", {}),
                simulated_sub_scores=s.get("sub_scores", {}) if bv is not None else {},
            ))

    if any(iv.kind == "transition_speed" for iv in interventions):
        caveats.append(
            "transition_speed scales frame-to-frame displacement about each player's "
            "own starting point. It changes distance covered and shape stability; it "
            "does not model fatigue, acceleration limits, or whether a player could "
            "actually run that fast.")
    if any(iv.kind == "swap_player" for iv in interventions):
        caveats.append(
            "swap_player replays one tracked player's REAL positions in another's "
            "place. It carries over the donor's movement in the donor's match "
            "context -- it does not adapt that movement to the new role.")

    return SimulationResult([iv.describe() for iv in interventions], metrics, caveats, unavailable)
