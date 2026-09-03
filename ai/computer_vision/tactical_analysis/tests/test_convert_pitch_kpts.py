"""The SoccerNet -> our-32 converter must preserve point IDENTITY.

The index is the only linkage between a predicted keypoint and a pitch
metre coordinate, so these tests round-trip a synthetic label through the
converter and check that each of our indices came back carrying the pixel
that belongs to its own landmark.
"""

import math

import pytest

from ai.computer_vision.tactical_analysis.pitch_keypoints import PITCH_KEYPOINTS_32
from scripts.convert_pitch_kpts import (
    MATCH_TOLERANCE_M,
    MIN_VISIBLE,
    SOCCERNET_57_WORLD_M,
    SOCCERNET_TO_OURS,
    convert_row,
    derive_mapping,
)

WIDTH, HEIGHT = 960, 540


def test_every_one_of_our_32_points_has_a_soccernet_slot():
    assert sorted(SOCCERNET_TO_OURS) == list(range(32))
    assert len(set(SOCCERNET_TO_OURS.values())) == 32


def test_mapped_slots_agree_on_pitch_coordinates():
    for ours, slot in SOCCERNET_TO_OURS.items():
        ox, oy = PITCH_KEYPOINTS_32[ours]
        sx, sy = SOCCERNET_57_WORLD_M[slot]
        d = math.hypot(ox - sx, oy - sy)
        assert d < MATCH_TOLERANCE_M, (
            f"our {ours} -> soccernet {slot} is {d:.3f} m apart")


def test_derivation_rejects_a_table_that_is_missing_our_landmarks():
    """If upstream ever changes the scheme, the converter must refuse to
    guess rather than silently emit a shifted mapping."""
    with pytest.raises(ValueError):
        derive_mapping(world=[(0.0, 0.0)])


def _synthetic_row():
    """A 57-long SoccerNet row where slot s sits at a pixel that encodes s,
    so the converter's output can be traced back to the slot it came from."""
    row = [None] * 57
    for slot in range(57):
        row[slot] = [10.0 + slot * 4.0, 20.0 + slot * 3.0]
    return row


def test_round_trip_puts_each_slot_at_our_matching_index():
    row = _synthetic_row()
    line, n_visible = convert_row(row, WIDTH, HEIGHT)
    assert n_visible == 32
    parts = line.split()
    assert parts[0] == "0"
    kpts = parts[5:]
    assert len(kpts) == 96, "32 keypoints x (x, y, v)"

    for ours in range(32):
        x, y, v = kpts[3 * ours], kpts[3 * ours + 1], kpts[3 * ours + 2]
        assert v == "2"
        slot = SOCCERNET_TO_OURS[ours]
        expected_x, expected_y = row[slot]
        assert float(x) == pytest.approx(expected_x / WIDTH, abs=1e-5)
        assert float(y) == pytest.approx(expected_y / HEIGHT, abs=1e-5)


def test_absent_landmarks_become_invisible_not_shifted():
    """A missing slot must blank ITS OWN index, never shift its neighbours
    up -- the classic way a keypoint scheme silently corrupts."""
    row = _synthetic_row()
    dropped = {0, 5, 13}
    for ours in dropped:
        row[SOCCERNET_TO_OURS[ours]] = None

    line, n_visible = convert_row(row, WIDTH, HEIGHT)
    assert n_visible == 32 - len(dropped)
    kpts = line.split()[5:]
    for ours in range(32):
        v = kpts[3 * ours + 2]
        if ours in dropped:
            assert v == "0" and kpts[3 * ours] == "0"
        else:
            assert v == "2"
            slot = SOCCERNET_TO_OURS[ours]
            assert float(kpts[3 * ours]) == pytest.approx(
                row[slot][0] / WIDTH, abs=1e-5)


def test_off_frame_landmarks_are_not_marked_visible():
    row = [None] * 57
    for ours in range(MIN_VISIBLE):
        row[SOCCERNET_TO_OURS[ours]] = [100.0, 100.0]
    row[SOCCERNET_TO_OURS[10]] = [-500.0, 100.0]      # far left of frame
    row[SOCCERNET_TO_OURS[11]] = [100.0, HEIGHT * 3]  # far below frame

    line, n_visible = convert_row(row, WIDTH, HEIGHT)
    assert n_visible == MIN_VISIBLE
    kpts = line.split()[5:]
    assert kpts[3 * 10 + 2] == "0"
    assert kpts[3 * 11 + 2] == "0"


def test_too_sparse_a_frame_is_dropped_entirely():
    row = [None] * 57
    for ours in range(MIN_VISIBLE - 1):
        row[SOCCERNET_TO_OURS[ours]] = [100.0, 100.0]
    assert convert_row(row, WIDTH, HEIGHT) is None


def test_box_covers_the_visible_landmarks():
    row = [None] * 57
    for ours, px in zip(range(MIN_VISIBLE),
                        [(96.0, 54.0), (864.0, 54.0), (96.0, 486.0),
                         (864.0, 486.0), (480.0, 270.0), (480.0, 54.0)]):
        row[SOCCERNET_TO_OURS[ours]] = list(px)
    line, _ = convert_row(row, WIDTH, HEIGHT)
    cx, cy, bw, bh = (float(v) for v in line.split()[1:5])
    assert cx == pytest.approx(0.5, abs=1e-3)
    assert cy == pytest.approx(0.5, abs=1e-3)
    assert bw == pytest.approx(0.8, abs=1e-3)
    assert bh == pytest.approx(0.8, abs=1e-3)
