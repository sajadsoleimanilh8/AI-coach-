"""
Regression tests for the two bugs fixed in bytetrack.py.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bytetrack import BYTETracker, STrack


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
