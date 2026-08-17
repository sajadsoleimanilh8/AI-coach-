"""
Sanity checks for pose.py's shoulder_line_angle().
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from pose import (
    MIN_LANDMARK_VISIBILITY,
    estimate_body_orientation,
    shoulder_line_angle,
)


def test_facing_camera_horizontal_shoulders():
    print("=" * 70)
    print("TEST 1: shoulders level (player facing camera) -> ~0 or ~180 deg")
    print("=" * 70)
    result = shoulder_line_angle(
        left_shoulder_xy=(0.3, 0.5), right_shoulder_xy=(0.6, 0.5),
        left_visibility=0.95, right_visibility=0.92,
    )
    print(f"  orientation_deg={result.orientation_deg:.2f}  confidence={result.confidence:.2f}")
    assert result.orientation_deg is not None
    assert abs(result.orientation_deg - 0.0) < 1e-6
    assert abs(result.confidence - 0.92) < 1e-9
    print("  PASS")


def test_side_on_shoulders_vertical():
    print()
    print("=" * 70)
    print("TEST 2: shoulders stacked vertically (player side-on) -> ~90 deg")
    print("=" * 70)
    result = shoulder_line_angle(
        left_shoulder_xy=(0.45, 0.3), right_shoulder_xy=(0.45, 0.6),
        left_visibility=0.9, right_visibility=0.9,
    )
    print(f"  orientation_deg={result.orientation_deg:.2f}")
    assert abs(result.orientation_deg - 90.0) < 1e-6
    print("  PASS")


def test_low_visibility_returns_unmeasured_not_fake_angle():
    print()
    print("=" * 70)
    print("TEST 3: below-threshold visibility -> orientation_deg=None, not a guessed angle")
    print("=" * 70)
    result = shoulder_line_angle(
        left_shoulder_xy=(0.3, 0.5), right_shoulder_xy=(0.6, 0.5),
        left_visibility=0.95, right_visibility=0.2,
    )
    print(f"  orientation_deg={result.orientation_deg}  confidence={result.confidence:.2f}")
    assert result.orientation_deg is None, "low-visibility landmark must not produce a confident-looking angle"
    assert result.confidence < MIN_LANDMARK_VISIBILITY
    print("  PASS")


def test_degenerate_identical_points_returns_unmeasured():
    print()
    print("=" * 70)
    print("TEST 4: both landmarks at the identical point -> None, not atan2(0,0)=0")
    print("=" * 70)
    result = shoulder_line_angle(
        left_shoulder_xy=(0.5, 0.5), right_shoulder_xy=(0.5, 0.5),
        left_visibility=0.9, right_visibility=0.9,
    )
    print(f"  orientation_deg={result.orientation_deg}")
    assert result.orientation_deg is None, (
        "degenerate zero-length shoulder vector must not silently read as 'facing along x-axis'"
    )
    print("  PASS")


def test_angle_wraps_to_0_360_range():
    print()
    print("=" * 70)
    print("TEST 5: angle is always reported in [0, 360), not negative")
    print("=" * 70)
    result = shoulder_line_angle(
        left_shoulder_xy=(0.6, 0.5), right_shoulder_xy=(0.3, 0.3),
        left_visibility=0.9, right_visibility=0.9,
    )
    print(f"  orientation_deg={result.orientation_deg:.2f}")
    assert 0.0 <= result.orientation_deg < 360.0
    print("  PASS")


def _legacy_mediapipe_solutions_available() -> bool:
    import mediapipe as mp
    return hasattr(mp, "solutions")


def test_estimate_body_orientation_on_blank_image():
    print()
    print("=" * 70)
    print("TEST 6: estimate_body_orientation() on a blank crop -> no landmarks, no crash")
    print("=" * 70)
    if not _legacy_mediapipe_solutions_available():
        print("  SKIPPED -- this mediapipe install has no `solutions` API (see pose.py's "
              "_load_legacy_pose_solution() docstring for the known-gotcha explanation). "
              "The angle math this really tests (TESTS 1-5) already passed independent of "
              "this; only the MediaPipe-calling wrapper itself is unverified here.")
        return
    blank_crop = np.zeros((128, 64, 3), dtype=np.uint8)
    result = estimate_body_orientation(blank_crop)
    print(f"  orientation_deg={result.orientation_deg}  confidence={result.confidence:.2f}")
    assert result.orientation_deg is None
    assert result.confidence == 0.0
    print("  PASS (a blank/unreadable crop reports 'not measured', not a fabricated angle)")


def test_estimate_body_orientation_on_empty_crop():
    print()
    print("=" * 70)
    print("TEST 7: zero-size crop (e.g. a degenerate bbox) doesn't crash")
    print("=" * 70)
    empty_crop = np.zeros((0, 0, 3), dtype=np.uint8)
    result = estimate_body_orientation(empty_crop)
    assert result.orientation_deg is None
    assert result.confidence == 0.0
    print("  PASS (this path returns before touching mediapipe at all, so it runs regardless)")


if __name__ == "__main__":
    test_facing_camera_horizontal_shoulders()
    test_side_on_shoulders_vertical()
    test_low_visibility_returns_unmeasured_not_fake_angle()
    test_degenerate_identical_points_returns_unmeasured()
    test_angle_wraps_to_0_360_range()
    test_estimate_body_orientation_on_blank_image()
    test_estimate_body_orientation_on_empty_crop()
    print()
    print("ALL TESTS PASSED")
