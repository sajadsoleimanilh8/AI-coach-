"""
Goalkeeper / referee inference -- HEURISTIC, never a model prediction.

WHY THIS EXISTS
    The old combined detector was 4-class (0=ball, 1=goalkeeper, 2=player,
    3=referee). The `player` dataset that replaced it is single-class
    (`nc: 1, names: [player]`) -- the dataset owner does not label officials
    or keepers separately. Phase 1 recorded the decision (2026-08-12,
    docs/pipeline_architecture.md): keep player-only detection and recover
    the two roles downstream by heuristic. It was documented but never
    implemented, and the doc's own carried risk -- "until implemented,
    referees will be clustered as ordinary players and pollute team
    assignment" -- was live. This module implements it.

WHAT IS AND IS NOT CLAIMED
    Everything here is MetricMethod.heuristic_proxy. There is no trained
    goalkeeper or referee classifier in this repo and this module is not a
    substitute for one; it is an inference from kit colour and position.
    RoleAssignment.method is set accordingly and must be carried through to
    any row written from it. Presenting these as detections would be a
    false capability claim.

THE TWO HEURISTICS

    REFEREE -- a kit-colour outlier.
        assign_teams() clusters outfield crops into two kits. An official
        wears neither, so their crops sit far from BOTH centroids. The test
        is relative to this video's own population (median + k*MAD of the
        distance-to-nearest-centroid), never an absolute colour distance,
        because kit palettes and lighting differ per match and an absolute
        threshold would not transfer.

        Note what this deliberately does NOT do: it does not treat a low
        team-assignment confidence as evidence of a referee. A track that
        flip-flops between the two kits is ALSO unassigned, and is a
        player with bad crops, not an official. Only distance-from-both
        separates the two, which is why TrackKitStats exposes it.

    GOALKEEPER -- a kit outlier who is also deep and isolated.
        A keeper wears a different kit from their outfield team, so they
        are a kit outlier too. What separates them from a referee is
        position: inside their own penalty area and spatially isolated
        from the pack. That test needs PITCH COORDINATES, which need a
        valid calibration.

        So when calibration is invalid, this module does NOT guess. It
        returns role=unknown with a stated reason rather than falling back
        to a pixel-space approximation of "deep", because which part of the
        IMAGE is a penalty area depends entirely on camera angle. On the
        broadcast footage currently in storage/uploads the calibration
        model does not produce a valid homography at all (see
        scripts/validate_auto_calibration.py), so goalkeeper inference
        genuinely does not fire there -- that is the honest outcome and it
        is reported as `unknown`, not as `player`.

CALIBRATION STATUS
    REFEREE_KIT_OUTLIER_MADS and the isolation/depth thresholds below are
    UNCALIBRATED starting points, the same caveat as
    SCANS_PER_MINUTE_TARGET and the team-assignment constants in
    constants.py. They are geometric/statistical reasoning, not values
    fitted to labelled officials. Do not present per-role output as
    validated; the mechanism is sound, the thresholds are not yet tuned.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from enum import Enum

from ai.computer_vision.tactical_analysis.constants import (
    PENALTY_AREA_DEPTH_M,
    PENALTY_AREA_WIDTH_M,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
)

logger = logging.getLogger(__name__)

# How many MADs above the population median distance-to-nearest-kit-centroid
# a track must sit to count as wearing neither kit. 3 MADs is the
# conventional robust-outlier cut and, with ~22 outfield players against
# 1-3 officials, keeps the officials outside the bulk without catching
# ordinary players whose crops were merely noisy.
REFEREE_KIT_OUTLIER_MADS = 3.0

# A goalkeeper spends the overwhelming majority of open play in their own
# defensive third. Requiring the MEDIAN position (not any position) inside
# the penalty area makes this robust to the keeper coming out for a corner.
GOALKEEPER_MAX_DEPTH_M = PENALTY_AREA_DEPTH_M

# Minimum median distance to the nearest other tracked player. A keeper
# standing on their line is the most isolated player on the pitch; an
# outfield defender in the same area is surrounded.
GOALKEEPER_MIN_ISOLATION_M = 12.0

# Below this many observed frames a track has too little evidence for any
# role claim -- reported as `unknown`, not defaulted to `player`.
MIN_TRACK_OBSERVATIONS = 5


class Role(str, Enum):
    player = "player"
    goalkeeper = "goalkeeper"
    referee = "referee"
    unknown = "unknown"


@dataclass
class RoleAssignment:
    player_id: int
    role: Role
    #: 0-1. NOT a model probability -- a normalised strength-of-evidence for
    #: the heuristic that fired. Named `confidence` to match the rest of the
    #: codebase's contract, but see this module's docstring.
    confidence: float
    reason: str
    #: Always "heuristic_proxy". Present as a field, rather than left to the
    #: caller to remember, so it cannot be written to the DB as anything
    #: else by accident.
    method: str = "heuristic_proxy"


def infer_roles(
    team_result,
    trajectories: dict[int, list] | None = None,
    calibration_valid: bool = False,
) -> dict[int, RoleAssignment]:
    """
    Assigns a Role to every tracked outfield player_id.

    Args:
        team_result: TeamAssignmentResult from
            team_assignment.assign_teams_with_stats().
        trajectories: {player_id: [TrackingPoint, ...]} from
            enrich_with_pitch_coordinates(). Required for GOALKEEPER
            inference (needs pitch positions); referee inference works
            without it.
        calibration_valid: whether pitch coordinates are trustworthy. When
            False, goalkeeper inference is skipped entirely rather than
            approximated -- see the module docstring.

    Returns:
        {player_id: RoleAssignment} for every track in team_result.
    """
    out: dict[int, RoleAssignment] = {}
    if not team_result.tracks:
        return out

    med, mad = team_result.dist_median, team_result.dist_mad
    # A degenerate MAD (every track equidistant) makes the outlier test
    # meaningless -- no role claim rather than an arbitrary one.
    outlier_cut = med + REFEREE_KIT_OUTLIER_MADS * mad if mad > 1e-9 else float("inf")

    pitch_stats = _pitch_stats(trajectories) if (trajectories and calibration_valid) else {}

    for pid, kit in team_result.tracks.items():
        if kit.n_crops < MIN_TRACK_OBSERVATIONS:
            out[pid] = RoleAssignment(
                pid, Role.unknown, 0.0,
                f"only {kit.n_crops} usable crops (< {MIN_TRACK_OBSERVATIONS})")
            continue

        is_kit_outlier = kit.dist_to_nearest_centroid > outlier_cut

        if not is_kit_outlier:
            # Matches one of the two kits -> an outfield player. This is the
            # only positive `player` claim made here, and it is the one with
            # real evidence behind it.
            out[pid] = RoleAssignment(
                pid, Role.player,
                confidence=min(1.0, kit.mean_margin * kit.vote_fraction),
                reason=f"kit matches team {kit.team_id or 'cluster'} "
                       f"(dist {kit.dist_to_nearest_centroid:.3f} <= cut {outlier_cut:.3f})")
            continue

        # Kit outlier: referee or goalkeeper. Position decides, and position
        # needs calibration.
        stats = pitch_stats.get(pid)
        if stats is None:
            out[pid] = RoleAssignment(
                pid, Role.unknown, 0.0,
                "kit matches neither team, but goalkeeper/referee cannot be "
                "separated without valid pitch coordinates"
                + ("" if calibration_valid else " (calibration invalid)"))
            continue

        depth_m, isolation_m, in_box = stats
        if in_box and isolation_m >= GOALKEEPER_MIN_ISOLATION_M:
            # Strength of evidence: how far past the isolation floor, capped.
            conf = min(1.0, (isolation_m - GOALKEEPER_MIN_ISOLATION_M) / 10.0 + 0.5)
            out[pid] = RoleAssignment(
                pid, Role.goalkeeper, conf,
                f"kit outlier, median position inside a penalty area "
                f"({depth_m:.1f} m from goal line), isolated by {isolation_m:.1f} m")
        else:
            conf = min(1.0, (kit.dist_to_nearest_centroid - outlier_cut) / max(mad, 1e-6) / 3.0)
            out[pid] = RoleAssignment(
                pid, Role.referee, max(0.3, conf),
                f"kit matches neither team (dist {kit.dist_to_nearest_centroid:.3f} > "
                f"cut {outlier_cut:.3f}) and is not positioned as a keeper "
                f"(in_box={in_box}, isolation {isolation_m:.1f} m)")

    n_ref = sum(1 for r in out.values() if r.role is Role.referee)
    n_gk = sum(1 for r in out.values() if r.role is Role.goalkeeper)
    n_unk = sum(1 for r in out.values() if r.role is Role.unknown)
    logger.info("role inference (heuristic_proxy): %d tracks -> %d player, "
                "%d goalkeeper, %d referee, %d unknown",
                len(out), len(out) - n_ref - n_gk - n_unk, n_gk, n_ref, n_unk)
    return out


def _pitch_stats(trajectories: dict[int, list]) -> dict[int, tuple[float, float, bool]]:
    """
    Per track: (median distance from the nearer goal line in metres,
    median distance to the nearest other player in metres, median position
    is inside a penalty area).

    Medians throughout: a keeper who comes up for a corner, or a player who
    briefly runs through the box, must not flip the classification.
    """
    # Per-frame positions, keyed by frame, so isolation is measured between
    # players who were actually on screen at the same time.
    by_frame: dict[int, list[tuple[int, float, float]]] = {}
    for pid, points in trajectories.items():
        for p in points:
            if p.pitch_x_m is None or p.pitch_y_m is None:
                continue
            by_frame.setdefault(p.frame_id, []).append((pid, p.pitch_x_m, p.pitch_y_m))

    depths: dict[int, list[float]] = {}
    isolations: dict[int, list[float]] = {}
    in_box_votes: dict[int, list[bool]] = {}

    half_box_w = PENALTY_AREA_WIDTH_M / 2.0
    cy = PITCH_WIDTH_M / 2.0

    for _frame, entries in by_frame.items():
        for pid, x, y in entries:
            depth = min(x, PITCH_LENGTH_M - x)          # distance to nearer goal line
            depths.setdefault(pid, []).append(depth)
            in_box_votes.setdefault(pid, []).append(
                depth <= PENALTY_AREA_DEPTH_M and abs(y - cy) <= half_box_w)
            others = [(ox, oy) for opid, ox, oy in entries if opid != pid]
            if others:
                isolations.setdefault(pid, []).append(
                    min(((ox - x) ** 2 + (oy - y) ** 2) ** 0.5 for ox, oy in others))

    out: dict[int, tuple[float, float, bool]] = {}
    for pid, d in depths.items():
        iso = isolations.get(pid)
        votes = in_box_votes.get(pid, [])
        out[pid] = (
            statistics.median(d),
            statistics.median(iso) if iso else float("inf"),
            (sum(votes) / len(votes)) > 0.5 if votes else False,
        )
    return out
