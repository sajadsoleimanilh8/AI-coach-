"""
Roboflow ``SoccerPitchConfiguration`` vertex order -> our PITCH_KEYPOINTS_32.

Source of the Roboflow order:
    https://github.com/roboflow/sports  ::  sports/configs/soccer.py
    (``SoccerPitchConfiguration.vertices``, 32 entries, centimetres, on a
    120 m x 70 m pitch). Fetched and transcribed 2026-08-30.

The two schemes enumerate the same 32 pitch landmarks in the same order, so
ROBOFLOW_TO_OURS is the identity. That is worth stating explicitly, because
the index is the ONLY linkage between a predicted keypoint and a pitch
metre coordinate (see pitch_keypoints.keypoints_to_correspondences), so a
silent off-by-one produces plausible-looking garbage homographies.

Two caveats, both verified rather than assumed:

1. The pitch DIMENSIONS differ. Roboflow declares 120 m x 70 m with a
   20.15 m penalty box; we use the IFAB-standard 105 m x 68 m with a 16.5 m
   box. Absolute metres therefore never match. What must match is the
   SEMANTICS of each slot, so ``roboflow_vertices_m()`` rebuilds the
   Roboflow ordering on OUR dimensions and the test compares that.

2. Slots 10, 11, 18 and 19 are the one place where Roboflow's declared
   vertex list disagrees with its own annotations. The list puts them on
   the penalty-box front line at goal-area width (y = centre +- 9.16 m);
   we put them where the penalty arc meets that same line
   (y = centre +- 7.31 m), 1.85 m away. Measurement on the actual labels
   (datasets/processed/field_datasets/2, a Roboflow
   football-field-detection export) shows the ANNOTATIONS sit on the arc,
   i.e. they follow our definition, not the published coordinate list:
   fitting a homography through our table gives a 0.39 m median residual at
   slot 10, where the list's definition would force 1.85 m. Dropping the
   four slots at inference also lowered end-to-end valid-frame rate
   (34%/23%/48% vs 35%/35%/61% on test1/2/3), confirming they carry real
   signal under our definition. So they stay mapped, and
   ROBOFLOW_CONFIG_DIVERGENT records the list-vs-annotation gap so a future
   upstream change is caught by the test rather than silently absorbed.
"""

from __future__ import annotations

from ai.computer_vision.tactical_analysis.constants import (
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
)
from ai.computer_vision.tactical_analysis.pitch_keypoints import (
    CENTRE_CIRCLE_RADIUS_M,
    GOAL_AREA_DEPTH_M,
    PENALTY_AREA_DEPTH_M,
    PENALTY_SPOT_DIST_M,
)

#: Roboflow SoccerPitchConfiguration defaults, centimetres.
ROBOFLOW_PITCH_CM = {
    "width": 7000,
    "length": 12000,
    "penalty_box_width": 4100,
    "penalty_box_length": 2015,
    "goal_box_width": 1832,
    "goal_box_length": 550,
    "centre_circle_radius": 915,
    "penalty_spot_distance": 1100,
}

#: Slots where Roboflow's published vertex list and its annotations disagree;
#: see the module docstring. Value is the y-offset (m) the list uses, ours
#: uses the penalty-arc intersection instead.
ROBOFLOW_CONFIG_DIVERGENT = {10, 11, 18, 19}

#: Roboflow vertex index -> our PITCH_KEYPOINTS_32 index.
ROBOFLOW_TO_OURS: list[int] = list(range(32))


def roboflow_vertices_m(
    length: float = PITCH_LENGTH_M,
    width: float = PITCH_WIDTH_M,
    penalty_box_width: float = 40.32,
    penalty_box_length: float = PENALTY_AREA_DEPTH_M,
    goal_box_width: float = 18.32,
    goal_box_length: float = GOAL_AREA_DEPTH_M,
    centre_circle_radius: float = CENTRE_CIRCLE_RADIUS_M,
    penalty_spot_distance: float = PENALTY_SPOT_DIST_M,
) -> list[tuple[float, float]]:
    """
    ``SoccerPitchConfiguration.vertices``, transcribed verbatim in order but
    parameterised by pitch dimensions so it can be evaluated on OUR pitch.
    Defaults are our dimensions, which is what the mapping test needs.
    """
    L, W = length, width
    pb_w, pb_l = penalty_box_width, penalty_box_length
    gb_w, gb_l = goal_box_width, goal_box_length
    r, spot = centre_circle_radius, penalty_spot_distance
    return [
        (0.0, 0.0),                          # 1
        (0.0, (W - pb_w) / 2),               # 2
        (0.0, (W - gb_w) / 2),               # 3
        (0.0, (W + gb_w) / 2),               # 4
        (0.0, (W + pb_w) / 2),               # 5
        (0.0, W),                            # 6
        (gb_l, (W - gb_w) / 2),              # 7
        (gb_l, (W + gb_w) / 2),              # 8
        (spot, W / 2),                       # 9
        (pb_l, (W - pb_w) / 2),              # 10
        (pb_l, (W - gb_w) / 2),              # 11  <- divergent slot
        (pb_l, (W + gb_w) / 2),              # 12  <- divergent slot
        (pb_l, (W + pb_w) / 2),              # 13
        (L / 2, 0.0),                        # 14
        (L / 2, W / 2 - r),                  # 15
        (L / 2, W / 2 + r),                  # 16
        (L / 2, W),                          # 17
        (L - pb_l, (W - pb_w) / 2),          # 18
        (L - pb_l, (W - gb_w) / 2),          # 19  <- divergent slot
        (L - pb_l, (W + gb_w) / 2),          # 20  <- divergent slot
        (L - pb_l, (W + pb_w) / 2),          # 21
        (L - spot, W / 2),                   # 22
        (L - gb_l, (W - gb_w) / 2),          # 23
        (L - gb_l, (W + gb_w) / 2),          # 24
        (L, 0.0),                            # 25
        (L, (W - pb_w) / 2),                 # 26
        (L, (W - gb_w) / 2),                 # 27
        (L, (W + gb_w) / 2),                 # 28
        (L, (W + pb_w) / 2),                 # 29
        (L, W),                              # 30
        (L / 2 - r, W / 2),                  # 31
        (L / 2 + r, W / 2),                  # 32
    ]


def remap(values: list, to_ours: list[int] = ROBOFLOW_TO_OURS) -> list:
    """
    Reorders a 32-long per-keypoint sequence from Roboflow slot order into
    our index order. ``out[to_ours[i]] = values[i]``.
    """
    if len(values) != len(to_ours):
        raise ValueError(f"expected {len(to_ours)} values, got {len(values)}")
    out: list = [None] * len(to_ours)
    for src, dst in enumerate(to_ours):
        out[dst] = values[src]
    return out
