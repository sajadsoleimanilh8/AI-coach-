"""
Goalkeeper / referee inference -- HEURISTIC, never a model prediction.
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

REFEREE_KIT_OUTLIER_MADS = 3.0

GOALKEEPER_MAX_DEPTH_M = PENALTY_AREA_DEPTH_M

GOALKEEPER_MIN_ISOLATION_M = 12.0

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
    confidence: float
    reason: str
    method: str = "heuristic_proxy"


def infer_roles(
    team_result,
    trajectories: dict[int, list] | None = None,
    calibration_valid: bool = False,
) -> dict[int, RoleAssignment]:
    """
    Assigns a Role to every tracked outfield player_id.
    """
    out: dict[int, RoleAssignment] = {}
    if not team_result.tracks:
        return out

    med, mad = team_result.dist_median, team_result.dist_mad
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
            out[pid] = RoleAssignment(
                pid, Role.player,
                confidence=min(1.0, kit.mean_margin * kit.vote_fraction),
                reason=f"kit matches team {kit.team_id or 'cluster'} "
                       f"(dist {kit.dist_to_nearest_centroid:.3f} <= cut {outlier_cut:.3f})")
            continue

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
    """
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
            depth = min(x, PITCH_LENGTH_M - x)
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
