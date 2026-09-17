"""
Weak-Zone Detection module.
Divides pitch into 6x4 coarse grid and computes defensive-player occupancy density.
Analysis Logic Design v3 & Implementation Spec §1.2.
"""

from __future__ import annotations

import numpy as np

from ai.common.metrics import MetricResult, metric_result
from ai.computer_vision.tactical_analysis.constants import (
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    SCHEMA_VERSION,
    TEAM_ASSIGNMENT_CONFIDENCE_MIN,
)


def compute_weak_zones(
    player_positions_m: list[tuple[float, float]],
    grid_x: int = 6,
    grid_y: int = 4,
    team_assignment_confidence: float = 0.0,
    opponent_positions_m: list[tuple[float, float]] | None = None,
) -> MetricResult:
    """
    Computes occupancy density per grid zone (grid_x x grid_y).
    Zones below bottom quartile coverage threshold are flagged as weak.

    `opponent_positions_m` (added Phase 3) turns "a zone we do not occupy"
    into "a zone we do not occupy AND the opposition does". Own-team
    density alone cannot distinguish a genuine weakness from empty space
    nobody is contesting: every team leaves its opposite corner empty all
    match, and flagging that as a weak zone is noise. Exposure is only
    reported when the caller supplies the opposing team's positions --
    passing None keeps the original own-density-only behaviour rather than
    inventing an opponent.

    Returns TeamMetric dict with metric_name="weak_zone_map".
    """
    if team_assignment_confidence < TEAM_ASSIGNMENT_CONFIDENCE_MIN or len(player_positions_m) == 0:
        # Generate default grid with None/low_upstream_confidence
        sub_scores = {f"zone_{x+1}_{y+1}": 0.0 for x in range(grid_x) for y in range(grid_y)}
        return metric_result(
            "weak_zone_map",
            None,
            method="heuristic_proxy",
            confidence="low_upstream_confidence" if team_assignment_confidence < TEAM_ASSIGNMENT_CONFIDENCE_MIN else "low_sample",
            sample_size=len(player_positions_m),
            sub_scores=sub_scores,
            schema_version=SCHEMA_VERSION,
        )

    dx = PITCH_LENGTH_M / grid_x
    dy = PITCH_WIDTH_M / grid_y

    counts = np.zeros((grid_x, grid_y), dtype=np.float64)

    for x, y in player_positions_m:
        gx = int(min(grid_x - 1, max(0, x // dx)))
        gy = int(min(grid_y - 1, max(0, y // dy)))
        counts[gx, gy] += 1.0

    total_pts = len(player_positions_m)
    density = counts / total_pts

    sub_scores = {}
    for x in range(grid_x):
        for y in range(grid_y):
            key = f"zone_{x+1}_{y+1}"
            sub_scores[key] = round(float(density[x, y]), 3)

    # Average occupancy per zone as baseline
    q25 = float(np.percentile(density, 25))
    under_occupied = density <= q25
    weak_zone_count = int(under_occupied.sum())

    if opponent_positions_m:
        opp_counts = np.zeros((grid_x, grid_y), dtype=np.float64)
        for x, y in opponent_positions_m:
            gx = int(min(grid_x - 1, max(0, x // dx)))
            gy = int(min(grid_y - 1, max(0, y // dy)))
            opp_counts[gx, gy] += 1.0
        opp_density = opp_counts / len(opponent_positions_m)

        # Exposed = we are under-occupying it AND the opposition is there
        # at more than their own average rate. Both halves are required:
        # the first alone flags dead space, the second alone flags any zone
        # the opposition happens to favour even when we are well set there.
        opp_mean = float(opp_density.mean())
        exposed = under_occupied & (opp_density > opp_mean)
        exposure = float((opp_density[exposed]).sum())

        for x in range(grid_x):
            for y in range(grid_y):
                sub_scores[f"opp_density_{x+1}_{y+1}"] = round(float(opp_density[x, y]), 3)
        sub_scores["exposed_zone_count"] = int(exposed.sum())
        sub_scores["opponent_share_in_exposed_zones"] = round(exposure, 3)
        sub_scores["under_occupied_zone_count"] = weak_zone_count
        sub_scores["opponent_sample_size"] = len(opponent_positions_m)

        # The headline value becomes the CONTESTED weakness count. Zones we
        # leave empty that nobody attacks are no longer counted as weak.
        value = float(int(exposed.sum()))
        basis = "own_density_and_opponent_presence"
    else:
        value = float(weak_zone_count)
        basis = "own_density_only"
        sub_scores["under_occupied_zone_count"] = weak_zone_count

    sub_scores["basis"] = basis

    return metric_result(
        "weak_zone_map",
        value,
        method="heuristic_proxy",
        confidence="normal",
        sample_size=total_pts,
        sub_scores=sub_scores,
        schema_version=SCHEMA_VERSION,
    )
