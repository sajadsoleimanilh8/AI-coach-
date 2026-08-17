"""
Sanity checks for tracker.py + trajectory.py, using synthetic detections
(no real video or trained YOLO checkpoint needed -- mirrors the approach in
tactical_analysis/tests/test_homography.py).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tactical_analysis"))

import numpy as np

from tracker import TrackedDetection, track_detections
from trajectory import enrich_with_pitch_coordinates, total_distance_covered, sprint_count


def make_synthetic_detection_frames(n_frames: int, start_xy=(100.0, 200.0), step_xy=(10.0, 0.0)):
    """One player walking in a straight line, one bbox per frame, no gaps."""
    frames = []
    x, y = start_xy
    dx, dy = step_xy
    w, h = 40.0, 90.0
    for f in range(n_frames):
        frames.append({
            "frame": f,
            "detections": [
                {"class": "player", "confidence": 0.9, "bbox": [x, y, x + w, y + h]},
            ],
        })
        x += dx
        y += dy
    return frames


def test_stable_id_across_frames():
    print("=" * 70)
    print("TEST 1: ByteTrack assigns ONE stable player_id across frames")
    print("=" * 70)
    frames = make_synthetic_detection_frames(n_frames=10)
    tracked = track_detections(frames, classes=["player"])

    ids_seen = set()
    for frame_dets in tracked:
        for det in frame_dets:
            ids_seen.add(det.player_id)

    print(f"  distinct player_id values across 10 frames: {ids_seen}")
    assert len(ids_seen) == 1, f"expected exactly 1 stable ID, got {len(ids_seen)}: {ids_seen}"
    print("  PASS")


def test_foot_point_is_bottom_center():
    print()
    print("=" * 70)
    print("TEST 2: foot_point() returns bottom-center, not box center")
    print("=" * 70)
    det = TrackedDetection(
        frame_number=0, player_id=1, class_name="player",
        team_id=None, team_assignment_confidence=None,
        x=100.0, y=200.0, width=40.0, height=90.0, confidence=0.9,
    )
    fx, fy = det.foot_point()
    expected = (100.0 + 20.0, 200.0 + 90.0)
    print(f"  foot_point = {(fx, fy)}  expected = {expected}")
    assert (fx, fy) == expected
    print("  PASS")


def test_speed_matches_known_constant_velocity():
    print()
    print("=" * 70)
    print("TEST 3: known constant pixel velocity -> correct pitch speed")
    print("=" * 70)
    H = np.array([
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 1.0],
    ])

    fps = 25.0
    step_px = 10.0
    frames = make_synthetic_detection_frames(n_frames=5, step_xy=(step_px, 0.0))
    tracked = track_detections(frames, classes=["player"])

    trajectories = enrich_with_pitch_coordinates(
        tracked, match_id="test-match", H=H, homography_confidence=0.95, fps=fps,
    )
    (player_id, traj), = trajectories.items()

    expected_speed = (step_px * 0.1) / (1.0 / fps)
    speeds = [p.speed for p in traj if p.speed is not None]
    print(f"  computed speeds (Kalman warm-up transient expected early): {[round(s, 3) for s in speeds]}")
    print(f"  expected steady-state speed: {expected_speed}")
    assert abs(speeds[-1] - expected_speed) / expected_speed < 0.10, (
        f"last speed {speeds[-1]} hasn't converged within 10% of {expected_speed}"
    )
    print("  PASS (converges to within 10% of true speed by the last frame)")

    print()
    print("  Distance/sprint aggregate sanity:")
    dist = total_distance_covered(traj)
    print(f"    total_distance_covered = {dist:.3f}m (~4.0m expected, some smoothing)")
    assert abs(dist - 4.0) / 4.0 < 0.15, f"total distance {dist} too far from expected ~4.0m"
    sprints = sprint_count(traj, sprint_threshold_ms=7.0)
    print(f"    sprint_count(>7 m/s)   = {sprints} (all samples are ~25 m/s, all should count)")
    assert sprints == len(speeds)
    print("  PASS")


def test_low_homography_confidence_drops_pitch_coords():
    print()
    print("=" * 70)
    print("TEST 4: low homography_confidence -> pitch coords/speed dropped, not fake")
    print("=" * 70)
    H = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 1.0]])
    frames = make_synthetic_detection_frames(n_frames=5)
    tracked = track_detections(frames, classes=["player"])

    trajectories = enrich_with_pitch_coordinates(
        tracked, match_id="test-match", H=H, homography_confidence=0.2, fps=25.0,
    )
    (player_id, traj), = trajectories.items()
    print(f"  all pitch_x_m are None: {all(p.pitch_x_m is None for p in traj)}")
    print(f"  all speed are None:     {all(p.speed is None for p in traj)}")
    assert all(p.pitch_x_m is None for p in traj)
    assert all(p.speed is None for p in traj)
    print("  PASS (no silently-wrong speed computed on unusable positions)")


if __name__ == "__main__":
    test_stable_id_across_frames()
    test_foot_point_is_bottom_center()
    test_speed_matches_known_constant_velocity()
    test_low_homography_confidence_drops_pitch_coords()
    print()
    print("ALL TESTS PASSED")
