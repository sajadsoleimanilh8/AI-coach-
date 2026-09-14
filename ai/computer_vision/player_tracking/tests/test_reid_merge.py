"""
reid_merge.py's guarantees, on synthetic tracks.

These are the properties the rest of the pipeline relies on and that a
tuning change must not quietly break: a merge never crosses teams, never
crosses a camera cut, never puts one id twice in one frame, and always
produces the same ids for the same input.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai.computer_vision.player_tracking.reid_merge import (
    MIN_TRACK_FRAMES,
    merge_reidentified_tracks,
)
from ai.computer_vision.player_tracking.tracker import TrackedDetection


def _det(frame: int, track_id: int, x: float, y: float,
         w: float = 20.0, h: float = 50.0) -> TrackedDetection:
    return TrackedDetection(
        frame_number=frame, player_id=track_id, class_name="player",
        team_id=None, team_assignment_confidence=None,
        x=x, y=y, width=w, height=h, confidence=0.9,
    )


def _frames(n: int) -> list[list[TrackedDetection]]:
    return [[] for _ in range(n)]


@pytest.fixture
def two_fragments():
    """One player walking right, tracked as id 1 then (after a gap) id 2."""
    frames = _frames(40)
    for f in range(0, 15):
        frames[f].append(_det(f, 1, 100.0 + 4 * f, 200.0))
    for f in range(20, 40):
        frames[f].append(_det(f, 2, 100.0 + 4 * f, 200.0))
    return frames


def test_no_video_means_no_merge(tmp_path, two_fragments):
    """Without decodable frames there is no colour evidence, so the pass must
    do nothing rather than merge on motion alone."""
    result = merge_reidentified_tracks(str(tmp_path / "missing.mp4"),
                                       two_fragments, fps=25.0)
    assert result.merges == 0
    assert {d.player_id for dets in two_fragments for d in dets} == {1, 2}


def test_one_id_per_frame_after_merge(two_fragments, monkeypatch):
    _force_merge(monkeypatch, teams={1: "team-home", 2: "team-home"})
    merge_reidentified_tracks("unused.mp4", two_fragments, fps=25.0)
    for detections in two_fragments:
        ids = [d.player_id for d in detections]
        assert len(ids) == len(set(ids))


def test_merge_keeps_the_lowest_id(two_fragments, monkeypatch):
    _force_merge(monkeypatch, teams={1: "team-home", 2: "team-home"})
    result = merge_reidentified_tracks("unused.mp4", two_fragments, fps=25.0)
    assert result.merges == 1
    assert {d.player_id for dets in two_fragments for d in dets} == {1}


def test_never_merges_across_teams(two_fragments, monkeypatch):
    _force_merge(monkeypatch, teams={1: "team-home", 2: "team-away"})
    result = merge_reidentified_tracks("unused.mp4", two_fragments, fps=25.0)
    assert result.merges == 0
    assert result.rejected_cross_team >= 1
    assert {d.player_id for dets in two_fragments for d in dets} == {1, 2}


def test_never_merges_across_a_hard_cut(two_fragments, monkeypatch):
    _force_merge(monkeypatch, teams={1: "team-home", 2: "team-home"}, cuts={17})
    result = merge_reidentified_tracks("unused.mp4", two_fragments, fps=25.0)
    assert result.merges == 0
    assert result.rejected_across_cut >= 1


def test_deterministic_across_runs(monkeypatch):
    def build():
        frames = _frames(40)
        for f in range(0, 15):
            frames[f].append(_det(f, 1, 100.0 + 4 * f, 200.0))
            frames[f].append(_det(f, 3, 600.0 - 4 * f, 400.0))
        for f in range(20, 40):
            frames[f].append(_det(f, 2, 100.0 + 4 * f, 200.0))
            frames[f].append(_det(f, 4, 600.0 - 4 * f, 400.0))
        return frames

    teams = {i: "team-home" for i in (1, 2, 3, 4)}
    first, second = build(), build()
    _force_merge(monkeypatch, teams=teams)
    merge_reidentified_tracks("unused.mp4", first, fps=25.0)
    merge_reidentified_tracks("unused.mp4", second, fps=25.0)

    as_ids = lambda fs: [[d.player_id for d in dets] for dets in fs]  # noqa: E731
    assert as_ids(first) == as_ids(second)


def test_ghost_tracks_are_suppressed(monkeypatch):
    frames = _frames(30)
    for f in range(30):
        frames[f].append(_det(f, 1, 100.0, 200.0))
    frames[10].append(_det(10, 99, 500.0, 300.0))  # one-frame ghost

    _force_merge(monkeypatch, teams={1: "team-home", 99: "team-home"})
    result = merge_reidentified_tracks("unused.mp4", frames, fps=25.0)

    assert result.ghosts_before == 1
    assert result.ghosts_suppressed == 1
    assert {d.player_id for dets in frames for d in dets} == {1}


def test_ghost_suppression_can_be_disabled(monkeypatch):
    frames = _frames(30)
    for f in range(30):
        frames[f].append(_det(f, 1, 100.0, 200.0))
    frames[10].append(_det(10, 99, 500.0, 300.0))

    _force_merge(monkeypatch, teams={1: "team-home", 99: "team-home"})
    result = merge_reidentified_tracks("unused.mp4", frames, fps=25.0,
                                       min_track_frames=0)
    assert result.ghosts_suppressed == 0
    assert 99 in {d.player_id for dets in frames for d in dets}


def test_min_track_frames_matches_what_a_trajectory_needs():
    """Guards the rationale in MIN_TRACK_FRAMES' comment: below three frames
    trajectory.py cannot produce an acceleration."""
    assert MIN_TRACK_FRAMES == 3


def _force_merge(monkeypatch, teams: dict[int, str], cuts: set[int] | None = None):
    """Replace the video-decoding half of the pass with fixed evidence.

    Everything these tests care about -- the team gate, the cut gate, id
    assignment, determinism -- is decided after the decode, so stubbing it
    lets the properties be tested on synthetic geometry with no fixture
    video. Identical histograms mean the colour gate always passes, which
    isolates the gate under test.
    """
    from ai.computer_vision.player_tracking import reid_merge

    histogram = np.ones((reid_merge.HIST_HUE_BINS, reid_merge.HIST_SAT_BINS),
                        dtype=np.float32)
    histogram /= histogram.sum()

    def fake_scan(video_path, frames, tracks):
        return (set(cuts or ()),
                {track_id: histogram.copy() for track_id in tracks},
                {track_id: [np.zeros(3)] for track_id in tracks})

    monkeypatch.setattr(reid_merge, "_scan_video", fake_scan)
    monkeypatch.setattr(reid_merge, "_resolve_teams",
                        lambda tracks, features: dict(teams))


def test_ghost_diagnostic_survives_suppression_being_disabled(monkeypatch):
    """ghosts_before must still report what the TRACKER produced.

    Measuring the pipeline with suppression off is the main reason to pass
    min_track_frames=0, so counting ghosts against a threshold of 0 -- and
    therefore reporting none -- would blind exactly that measurement.
    """
    frames = _frames(30)
    for f in range(30):
        frames[f].append(_det(f, 1, 100.0, 200.0))
    frames[10].append(_det(10, 99, 500.0, 300.0))  # one-frame ghost

    _force_merge(monkeypatch, teams={1: "team-home", 99: "team-home"})
    result = merge_reidentified_tracks("unused.mp4", frames, fps=25.0,
                                       min_track_frames=0)

    assert result.ghosts_before == 1, "the tracker still emitted a ghost"
    assert result.ghosts_suppressed == 0, "but nothing may be dropped"


def test_ghosts_suppressed_never_exceeds_ghosts_before(monkeypatch):
    """A custom threshold must keep the two counters coherent."""
    frames = _frames(30)
    for f in range(30):
        frames[f].append(_det(f, 1, 100.0, 200.0))
    for f in (10, 11, 12):
        frames[f].append(_det(f, 99, 500.0, 300.0))  # 3 frames: ghost only at >3

    _force_merge(monkeypatch, teams={1: "team-home", 99: "team-home"})
    result = merge_reidentified_tracks("unused.mp4", frames, fps=25.0,
                                       min_track_frames=5)

    assert result.ghosts_suppressed <= result.ghosts_before


def test_a_lone_ghost_track_is_still_suppressed(monkeypatch):
    """Suppression must not depend on how many OTHER tracks happen to exist."""
    frames = _frames(30)
    frames[5].append(_det(5, 7, 100.0, 200.0))

    _force_merge(monkeypatch, teams={7: "team-home"})
    result = merge_reidentified_tracks("unused.mp4", frames, fps=25.0)

    assert result.ghosts_suppressed == 1
    assert all(not dets for dets in frames)


def test_one_crop_one_vote_regardless_of_crop_size():
    """A track's colour signature must not be dominated by its biggest crop.

    Two crops of the same kit at different distances, plus one crop of a
    different colour. If crops were accumulated as raw pixel counts the huge
    near crop would swamp the other two; with per-crop normalisation each
    contributes a third.
    """
    from ai.computer_vision.player_tracking.reid_merge import _jersey_histogram

    def pixels(hue: float, count: int) -> np.ndarray:
        return np.column_stack([
            np.full(count, hue, dtype=np.float64),
            np.full(count, 200.0, dtype=np.float64),
            np.full(count, 200.0, dtype=np.float64),
        ])

    near = _jersey_histogram(pixels(20.0, 5000))
    far = _jersey_histogram(pixels(20.0, 50))
    other = _jersey_histogram(pixels(120.0, 50))

    assert near.sum() == pytest.approx(1.0)
    assert far.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(near, far, atol=1e-6)

    signature = near + far + other
    signature /= signature.sum()
    # the odd colour still holds its third, rather than being rounded away
    assert signature[other > 0].sum() == pytest.approx(1 / 3, abs=1e-6)
