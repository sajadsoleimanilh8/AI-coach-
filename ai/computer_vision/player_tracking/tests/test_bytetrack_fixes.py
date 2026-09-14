"""
Regression tests for the two bugs fixed in bytetrack.py.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bytetrack import (
    REMOVED_TRACK_HISTORY,
    BYTETracker,
    STrack,
    linear_assignment,
)


def _det(x1, y1, x2, y2, score, class_id=0):
    return [x1, y1, x2, y2, score, class_id]


def test_no_crash_on_single_hit_then_disappear():
    """
    Regression test for BUG 1: a track that gets created (activate()) and
    is NOT matched on the very next frame (never gets update() called)
    must not crash once it ages past track_buffer in lost_stracks -- this
    used to raise AttributeError because frame_id was never set in
    """
    print("=" * 70)
    print("TEST 1: single-hit track that immediately disappears (BUG 1)")
    print("=" * 70)

    tracker = BYTETracker(track_thresh=0.5, track_buffer=3, match_thresh=0.8)

    tracker.update(np.array([
        _det(100, 100, 140, 190, 0.9),
        _det(500, 500, 540, 590, 0.9),
    ]))

    for frame in range(2, 7):
        x_shift = (frame - 1) * 5
        tracker.update(np.array([
            _det(100 + x_shift, 100, 140 + x_shift, 190, 0.9),
        ]))
        print(f"  frame {frame}: update() completed without crashing")

    print("  PASS -- no AttributeError across track_buffer expiry")


def test_confirmation_delay_filters_transient_false_positive():
    """
    Regression test for BUG 2: a detection that appears for exactly one
    frame and is never matched again must NOT be emitted with a
    player_id -- it should sit as 'unconfirmed' and then get discarded,
    not leak into the tracked output.
    """
    print()
    print("=" * 70)
    print("TEST 2: transient single-frame false positive is not confirmed (BUG 2)")
    print("=" * 70)

    tracker = BYTETracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8)

    tracker.update(np.array([_det(100, 100, 140, 190, 0.9)]))

    output_2 = tracker.update(np.array([
        _det(105, 100, 145, 190, 0.9),
        _det(800, 800, 840, 890, 0.9),
    ]))
    ids_frame_2 = {t.track_id for t in output_2}
    print(f"  frame 2 output track_ids: {ids_frame_2}")
    assert len(ids_frame_2) == 1, f"expected exactly 1 confirmed track, got {ids_frame_2}"

    output_3 = tracker.update(np.array([
        _det(110, 100, 150, 190, 0.9),
    ]))
    ids_frame_3 = {t.track_id for t in output_3}
    print(f"  frame 3 output track_ids: {ids_frame_3}")
    assert len(ids_frame_3) == 1
    print("  PASS -- transient detection never got a confirmed player_id")


def test_genuine_new_track_confirms_after_second_hit():
    """A real new player (present on 2+ consecutive frames) should still
    end up tracked -- the confirmation delay must not eat real tracks."""
    print()
    print("=" * 70)
    print("TEST 3: a real (persistent) new track gets confirmed by frame 2")
    print("=" * 70)

    tracker = BYTETracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
    tracker.update(np.array([_det(100, 100, 140, 190, 0.9)]))

    tracker.update(np.array([
        _det(105, 100, 145, 190, 0.9),
        _det(300, 300, 340, 390, 0.9),
    ]))

    output_3 = tracker.update(np.array([
        _det(110, 100, 150, 190, 0.9),
        _det(305, 300, 345, 390, 0.9),
    ]))
    print(f"  frame 3 confirmed track count: {len(output_3)}")
    assert len(output_3) == 2, "genuine persistent second player should be confirmed by frame 3"
    print("  PASS -- real tracks aren't blocked by the confirmation delay")


if __name__ == "__main__":
    STrack.reset_id_counter()
    test_no_crash_on_single_hit_then_disappear()
    STrack.reset_id_counter()
    test_confirmation_delay_filters_transient_false_positive()
    STrack.reset_id_counter()
    test_genuine_new_track_confirms_after_second_hit()
    print()
    print("ALL TESTS PASSED")


def test_above_threshold_pair_cannot_steal_a_detection():
    """A forbidden pair must not consume a row/column in the assignment.

    scipy's linear_sum_assignment minimises total cost over the WHOLE
    matrix. Filtering by threshold only afterwards lets the optimiser pick
    an over-threshold pair -- which is then discarded -- while that choice
    has already pushed a detection onto the wrong track. Here t0 is the
    rightful owner of d0 (cost 0.10) but the unmasked optimum is
    t0->d1 (0.95, discarded) + t1->d0 (0.12), handing d0 to t1: an identity
    switch manufactured entirely by the matcher.
    """
    cost = np.array([[0.10, 0.95],
                     [0.12, 0.99]])
    matches, unmatched_a, _unmatched_b = linear_assignment(cost, thresh=0.8)

    assert matches.tolist() == [[0, 0]], "d0 must go to its cheapest valid track"
    assert 1 in unmatched_a.tolist(), "t1 has no valid detection and must be unmatched"


def test_assignment_is_still_globally_optimal_among_valid_pairs():
    """Masking must not degrade into greedy matching."""
    cost = np.array([[0.10, 0.20],
                     [0.15, 0.11]])
    matches, _ua, _ub = linear_assignment(cost, thresh=0.8)
    assert sorted(matches.tolist()) == [[0, 0], [1, 1]]


def test_every_index_is_accounted_for_exactly_once():
    """Rectangular matrices must not drop or duplicate an index."""
    cost = np.array([[0.1, 0.9, 0.9],
                     [0.9, 0.9, 0.2]])
    matches, unmatched_a, unmatched_b = linear_assignment(cost, thresh=0.8)
    rows = [m[0] for m in matches.tolist()] + unmatched_a.tolist()
    cols = [m[1] for m in matches.tolist()] + unmatched_b.tolist()
    assert sorted(rows) == [0, 1]
    assert sorted(cols) == [0, 1, 2]


def test_predict_moves_the_box_used_for_association():
    """predict() must actually steer matching.

    The tracker calls predict() on the whole pool and then associates on
    IoU against `tlbr`. If `tlbr` reported the last measured box instead of
    the filter's estimate, the prediction would be dead code and a track
    would be matched against where its player was, not where it is -- the
    difference that loses a fast-moving player through an occlusion.
    """
    from bytetrack import KalmanFilter

    kf = KalmanFilter()
    track = STrack(np.array([100.0, 100.0, 20.0, 40.0]), 0.9, 0)
    track.activate(kf, 1)
    for frame in range(2, 6):
        moving_right = STrack(np.array([100.0 + 10 * (frame - 1), 100.0, 20.0, 40.0]), 0.9, 0)
        track.predict()
        track.update(moving_right, frame)

    before = track.tlbr.copy()
    track.predict()
    after = track.tlbr

    assert after[0] > before[0], "a rightward-moving track must predict rightward"
    centre_x = after[0] + (after[2] - after[0]) / 2
    assert abs(centre_x - track.mean[0]) < 1e-6, "tlbr must follow the Kalman mean"


def test_lost_tracks_age_out_across_frames_with_no_detections():
    """A frame with nothing in it must still age the lost tracks.

    Broadcast football goes blank for a second or more constantly -- a cut to
    the crowd, a replay wipe. Taking an early return on those frames without
    expiring anything let a track outlive track_buffer by an arbitrary margin
    and then re-claim whoever appeared next, turning a dead identity into a
    silent ID switch.
    """
    tracker = BYTETracker(track_thresh=0.5, track_buffer=5, match_thresh=0.8)

    for frame in range(4):
        tracker.update(np.array([[100.0, 100.0, 120.0, 180.0, 0.9, 0.0]]))
    survivor = {t.track_id for t in tracker.tracked_stracks}
    assert survivor, "a steadily-detected box should hold an id"

    for _ in range(40):
        tracker.update(np.empty((0, 6)))

    assert tracker.lost_stracks == [], "nothing may still be alive 40 frames past a buffer of 5"

    # An unrelated box far away must get a NEW id, not the expired one.
    tracker.update(np.array([[600.0, 400.0, 620.0, 480.0, 0.9, 0.0]]))
    tracker.update(np.array([[600.0, 400.0, 620.0, 480.0, 0.9, 0.0]]))
    reborn = {t.track_id for t in tracker.tracked_stracks}
    assert reborn.isdisjoint(survivor), "an expired identity must not be re-used"


def test_removed_track_history_stays_bounded():
    """removed_stracks is walked once per frame, so it cannot grow forever."""
    tracker = BYTETracker(track_thresh=0.5, track_buffer=3, match_thresh=0.8)
    for frame in range(300):
        x = 50.0 * (frame % 20)
        tracker.update(np.array([[x, 100.0, x + 18.0, 160.0, 0.9, 0.0]]))
    assert len(tracker.removed_stracks) <= REMOVED_TRACK_HISTORY
