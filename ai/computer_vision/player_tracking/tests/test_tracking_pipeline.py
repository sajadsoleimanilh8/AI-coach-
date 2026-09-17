"""
Sanity checks for tracker.py + trajectory.py, using synthetic detections
(no real video or trained YOLO checkpoint needed -- mirrors the approach in
tactical_analysis/tests/test_homography.py).

Covers:
  1. track_detections(): a synthetic "already-detected" player moving in a
     straight line gets a single stable player_id across frames (this is
     the entire point of ByteTrack -- ID must not change frame to frame).
  2. TrackedDetection.foot_point(): returns bottom-center, not box center.
  3. trajectory.enrich_with_pitch_coordinates(): a player moving at a known
     constant real-world speed produces the expected speed_m/s within
     tolerance, and distance/acceleration behave sanely.
  4. Low homography_confidence correctly drops pitch coordinates instead of
     silently computing garbage speed on unusable positions.

Run: python3 tests/test_tracking_pipeline.py
"""

from __future__ import annotations

import numpy as np

from ai.computer_vision.player_tracking.tracker import TrackedDetection, track_detections
from ai.computer_vision.player_tracking.trajectory import (
    MAX_SPEED_GAP_FRAMES,
    enrich_with_pitch_coordinates,
    sprint_count,
    total_distance_covered,
)


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
    expected = (100.0 + 20.0, 200.0 + 90.0)  # x + w/2, y + h (bottom, not center)
    print(f"  foot_point = {(fx, fy)}  expected = {expected}")
    assert (fx, fy) == expected
    print("  PASS")


def test_speed_matches_known_constant_velocity():
    print()
    print("=" * 70)
    print("TEST 3: known constant pixel velocity -> correct pitch speed")
    print("=" * 70)
    # Identity-like homography: 10 pixels == 1 pitch meter, no rotation,
    # so we can hand-verify the expected speed without relying on
    # tactical_analysis's own fitting code (keeps this test independent).
    H = np.array([
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 1.0],
    ])

    fps = 25.0
    step_px = 10.0  # per frame, x-direction only
    frames = make_synthetic_detection_frames(n_frames=5, step_xy=(step_px, 0.0))
    tracked = track_detections(frames, classes=["player"])

    trajectories = enrich_with_pitch_coordinates(
        tracked, match_id="test-match", H=H, homography_confidence=0.95, fps=fps,
    )
    (player_id, traj), = trajectories.items()

    # NOTE: the fixed BYTETracker (bytetrack.py) returns Kalman-filtered
    # positions (STrack.tlbr comes from the filter's state estimate, not
    # the raw measurement) -- this is correct, standard tracker behavior
    # (it smooths detection jitter), but it means speed on the first couple
    # of frames has a filter warm-up transient before converging to the
    # true constant velocity, rather than matching it exactly frame 1.
    # What matters downstream (ACWR, sprint counts) is convergence, not
    # frame-1 exactness, so check that here.
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
        tracked, match_id="test-match", H=H, homography_confidence=0.2, fps=25.0,  # below 0.6 threshold
    )
    (player_id, traj), = trajectories.items()
    print(f"  all pitch_x_m are None: {all(p.pitch_x_m is None for p in traj)}")
    print(f"  all speed are None:     {all(p.speed is None for p in traj)}")
    assert all(p.pitch_x_m is None for p in traj)
    assert all(p.speed is None for p in traj)
    print("  PASS (no silently-wrong speed computed on unusable positions)")


class _Cal:
    """Duck-type of frame_data.CalibrationState, so these tests stay free of
    the video pipeline's imports (same approach as the synthetic H above)."""

    def __init__(self, H, valid=True, confidence=0.9):
        self.H = H
        self.valid = valid
        self.confidence = confidence


def test_per_frame_homography_moves_the_same_pixel():
    print()
    print("=" * 70)
    print("TEST 5: two frames, two different valid H -> different pitch coords")
    print("=" * 70)
    # Same scale, different translation: exactly what a panning camera does
    # to the mapping between a fixed pixel and the pitch.
    H_a = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 1.0]])
    H_b = np.array([[0.1, 0.0, 5.0], [0.0, 0.1, 0.0], [0.0, 0.0, 1.0]])

    # One player, NOT moving in pixels, across two frames.
    frames = make_synthetic_detection_frames(n_frames=2, step_xy=(0.0, 0.0))
    tracked = track_detections(frames, classes=["player"])

    trajectories = enrich_with_pitch_coordinates(
        tracked, match_id="test-match", H=None, homography_confidence=0.0, fps=25.0,
        calibration_by_frame={0: _Cal(H_a), 1: _Cal(H_b)},
    )
    (player_id, traj), = trajectories.items()
    assert len(traj) == 2, f"expected 2 points, got {len(traj)}"

    x0, x1 = traj[0].pitch_x_m, traj[1].pitch_x_m
    print(f"  frame 0 (H_a) pitch_x_m = {x0}")
    print(f"  frame 1 (H_b) pitch_x_m = {x1}")
    assert x0 is not None and x1 is not None
    # H_b adds a 5m x-offset, so the SAME pixel must land 5m further along.
    assert abs((x1 - x0) - 5.0) < 1e-6, (
        f"per-frame H not applied: same pixel moved {x1 - x0}m, expected 5.0m"
    )
    print("  PASS (each frame projected through its own matrix)")


def test_per_frame_invalid_calibration_yields_none_not_carried_h():
    print()
    print("=" * 70)
    print("TEST 6: a frame with invalid calibration -> None, not a carried H")
    print("=" * 70)
    H = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 1.0]])
    frames = make_synthetic_detection_frames(n_frames=3, step_xy=(0.0, 0.0))
    tracked = track_detections(frames, classes=["player"])

    trajectories = enrich_with_pitch_coordinates(
        tracked, match_id="test-match", H=None, homography_confidence=0.0, fps=25.0,
        calibration_by_frame={
            0: _Cal(H, valid=True),
            1: _Cal(H, valid=False, confidence=0.55),  # H present but NOT valid
            2: _Cal(H, valid=True),
        },
    )
    (player_id, traj), = trajectories.items()
    by_frame = {p.frame_id: p for p in traj}
    print(f"  frame 0 pitch_x_m = {by_frame[0].pitch_x_m}")
    print(f"  frame 1 pitch_x_m = {by_frame[1].pitch_x_m} (invalid calibration)")
    print(f"  frame 2 pitch_x_m = {by_frame[2].pitch_x_m}")
    assert by_frame[0].pitch_x_m is not None
    assert by_frame[1].pitch_x_m is None, "invalid frame was projected anyway"
    assert by_frame[1].speed is None
    assert by_frame[2].pitch_x_m is not None
    # The per-frame confidence must be recorded, not a match-wide one.
    assert by_frame[1].homography_confidence == 0.55
    print("  PASS (gap is an honest None; per-frame confidence recorded)")


def test_speed_uses_frame_gap_not_one_frame_interval():
    print()
    print("=" * 70)
    print("TEST 7: speed across a calibration gap divides by the REAL elapsed time")
    print("=" * 70)
    H = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 1.0]])
    fps = 25.0
    step_px = 10.0  # per frame, x-direction only
    frames = make_synthetic_detection_frames(n_frames=3, step_xy=(step_px, 0.0))
    tracked = track_detections(frames, classes=["player"])

    # Frame 1 has no valid calibration, so frames 0 and 2 are 2 frames apart.
    trajectories = enrich_with_pitch_coordinates(
        tracked, match_id="test-match", H=None, homography_confidence=0.0, fps=fps,
        calibration_by_frame={
            0: _Cal(H, valid=True),
            1: _Cal(H, valid=False),
            2: _Cal(H, valid=True),
        },
    )
    (player_id, traj), = trajectories.items()
    by_frame = {p.frame_id: p for p in traj}
    speed = by_frame[2].speed
    # Two frames of travel: distance is ~2 * step, elapsed is 2/fps, so the
    # speed must match the one-frame-interval speed -- NOT double it.
    expected = (step_px * 0.1) / (1.0 / fps)
    print(f"  speed at frame 2 = {speed} m/s, one-interval speed = {expected} m/s")
    assert speed is not None
    assert abs(speed - expected) / expected < 0.15, (
        f"speed {speed} not within 15% of {expected} -- gap divided by the "
        f"wrong dt would report ~{expected * 2}"
    )
    print("  PASS (multi-frame displacement divided by multi-frame dt)")


def test_speed_not_emitted_across_a_long_gap():
    print()
    print("=" * 70)
    print("TEST 8: gap longer than MAX_SPEED_GAP_FRAMES -> no kinematics at all")
    print("=" * 70)
    H = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 1.0]])
    n = MAX_SPEED_GAP_FRAMES + 3
    frames = make_synthetic_detection_frames(n_frames=n, step_xy=(10.0, 0.0))
    tracked = track_detections(frames, classes=["player"])

    cal = {i: _Cal(H, valid=(i == 0 or i == n - 1)) for i in range(n)}
    trajectories = enrich_with_pitch_coordinates(
        tracked, match_id="test-match", H=None, homography_confidence=0.0, fps=25.0,
        calibration_by_frame=cal,
    )
    (player_id, traj), = trajectories.items()
    by_frame = {p.frame_id: p for p in traj}
    last = by_frame[n - 1]
    print(f"  gap = {n - 1} frames (max is {MAX_SPEED_GAP_FRAMES})")
    print(f"  last frame: pitch_x_m={last.pitch_x_m}, speed={last.speed}, "
          f"distance={last.distance}")
    assert last.pitch_x_m is not None, "position is still known and should be kept"
    assert last.speed is None, "speed integrated across too long a gap"
    assert last.distance is None
    print("  PASS (position kept, kinematics withheld)")


if __name__ == "__main__":
    test_stable_id_across_frames()
    test_foot_point_is_bottom_center()
    test_speed_matches_known_constant_velocity()
    test_low_homography_confidence_drops_pitch_coords()
    test_per_frame_homography_moves_the_same_pixel()
    test_per_frame_invalid_calibration_yields_none_not_carried_h()
    test_speed_uses_frame_gap_not_one_frame_interval()
    test_speed_not_emitted_across_a_long_gap()
    print()
    print("ALL TESTS PASSED")
