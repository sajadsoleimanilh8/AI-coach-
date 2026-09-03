"""The Roboflow-vertex -> our-index mapping must stay coordinate-exact."""

import math

from ai.computer_vision.tactical_analysis._roboflow_pitch_map import (
    ROBOFLOW_CONFIG_DIVERGENT,
    ROBOFLOW_TO_OURS,
    remap,
    roboflow_vertices_m,
)
from ai.computer_vision.tactical_analysis.pitch_keypoints import PITCH_KEYPOINTS_32

TOLERANCE_M = 0.5
#: How far Roboflow's published list places slots 10/11/18/19 from where we
#: (and its own annotators) place them. See _roboflow_pitch_map's docstring.
DIVERGENCE_M = 1.85


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def test_mapping_is_a_permutation_of_all_32_indices():
    assert sorted(ROBOFLOW_TO_OURS) == list(range(32))


def test_every_mapped_point_agrees_within_half_a_metre():
    """Roboflow's vertex ORDER, evaluated on our pitch, must reproduce our
    table. The four documented divergent slots are checked separately."""
    vertices = roboflow_vertices_m()
    for rf_idx, our_idx in enumerate(ROBOFLOW_TO_OURS):
        if our_idx in ROBOFLOW_CONFIG_DIVERGENT:
            continue
        d = _distance(vertices[rf_idx], PITCH_KEYPOINTS_32[our_idx])
        assert d < TOLERANCE_M, (
            f"roboflow slot {rf_idx} -> our index {our_idx} is {d:.3f} m apart: "
            f"{vertices[rf_idx]} vs {PITCH_KEYPOINTS_32[our_idx]}")


def test_divergent_slots_differ_by_exactly_the_documented_amount():
    """Guards the one known list-vs-annotation gap. If upstream ever fixes
    its coordinate list, this fails and the docstring must be revisited."""
    vertices = roboflow_vertices_m()
    for our_idx in sorted(ROBOFLOW_CONFIG_DIVERGENT):
        rf_idx = ROBOFLOW_TO_OURS.index(our_idx)
        d = _distance(vertices[rf_idx], PITCH_KEYPOINTS_32[our_idx])
        assert abs(d - DIVERGENCE_M) < 0.05, (
            f"slot {rf_idx} divergence is {d:.3f} m, expected {DIVERGENCE_M} m")


def test_remap_moves_values_into_our_index_order():
    values = [f"rf{i}" for i in range(32)]
    out = remap(values)
    for rf_idx, our_idx in enumerate(ROBOFLOW_TO_OURS):
        assert out[our_idx] == values[rf_idx]
