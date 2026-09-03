"""
Appearance re-association: repairs identities that ByteTrack dropped.

ByteTrack associates by motion and IoU alone. When a player is occluded,
leaves the frame, or is missed by the detector for longer than the tracker's
buffer, the track dies and the same player comes back wearing a brand-new
id. Everything downstream that keys on player_id -- trajectory.py's
distance/speed integration, the per-player intelligence scores -- then sees
two half-players instead of one.

This module is a POST-PASS over the materialised frame list. It does not
touch ByteTrack and it does not touch track_video()'s streaming interface:
it takes the same `list[list[TrackedDetection]]` that team_assignment.py
takes, and rewrites `det.player_id` in place, exactly as
assign_teams_with_stats() rewrites `det.team_id` in place.

HOW A MERGE IS DECIDED
Every track that dies is held for HOLD_FRAMES frames along with its jersey
histogram, its last box, and its last velocity. A track born inside that
window is merged into it only when ALL of these hold:

  * same team          -- the hard constraint. Tracks whose team could not
                          be resolved (too few usable crops, and every
                          goalkeeper, whose kit is deliberately unlike
                          either outfield kit) are never merged at all.
  * no hard cut between -- see hard_cut_frames(). Across a shot boundary,
                          position and velocity are meaningless and eleven
                          identically-dressed players are mutually
                          indistinguishable, so association is SUSPENDED
                          rather than guessed.
  * position agrees    -- the newborn's first box is within MAX_PREDICTION_
                          DISTANCE_H box-heights of where the dead track was
                          heading under constant velocity.
  * size agrees        -- box heights within HEIGHT_RATIO_RANGE, which is
                          what stops a near player being merged into a far
                          one.
  * colour agrees      -- hue/saturation histogram correlation at least
                          MIN_HISTOGRAM_CORRELATION.

Ties are broken by track id and the assignment is one-to-one and greedy on
ascending cost, so the same video always produces the same ids.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np

from ai.computer_vision.player_tracking.tracker import TrackedDetection
from ai.computer_vision.tactical_analysis.team_assignment import (
    GOALKEEPER_CLASS,
    OUTFIELD_CLASS,
    crop_to_feature,
    resolve_track_teams,
    torso_hsv_pixels,
)

PLAYER_CLASSES = (OUTFIELD_CLASS, GOALKEEPER_CLASS)

# --- association gates ------------------------------------------------------
HOLD_SECONDS = 1.8                 # how long a dead track stays re-associable
MAX_PREDICTION_DISTANCE_H = 2.0    # box-heights between prediction and newborn
HEIGHT_RATIO_RANGE = (0.70, 1.45)
MIN_HISTOGRAM_CORRELATION = 0.50   # cv2.HISTCMP_CORREL over the H/S histogram
VELOCITY_SAMPLE = 5                # frames averaged into the velocity estimate
VELOCITY_MAX_PX = 40.0             # per-frame clamp, so a jittery final box
                                   # cannot fling the prediction across the pitch
APPEARANCE_CROPS = 12              # crops sampled per track for its histogram

# --- jersey histogram -------------------------------------------------------
HIST_HUE_BINS = 24
HIST_SAT_BINS = 8

# --- hard-cut detection -----------------------------------------------------
# A shot boundary changes the whole frame's colour distribution in one frame.
# A whip pan, which the optical-flow signal in auto_calibration.py's
# CameraMotionDetector also reports as CameraMotion.cut, does not -- content
# leaves the frame gradually. Comparing consecutive HSV histograms separates
# the two, which matters because a pan must NOT suspend association (the
# players are still there) while a real cut must.
CUT_HIST_DISTANCE = 0.50           # Bhattacharyya distance between frames
CUT_PROBE_SIZE = (320, 180)


@dataclass
class _Track:
    track_id: int
    frames: list[int] = field(default_factory=list)
    boxes: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)
    is_goalkeeper: bool = False

    @property
    def first(self) -> int:
        return self.frames[0]

    @property
    def last(self) -> int:
        return self.frames[-1]

    def centre(self, frame: int) -> tuple[float, float]:
        x, y, w, h = self.boxes[frame]
        return (x + w / 2.0, y + h / 2.0)

    def height(self, frame: int) -> float:
        return self.boxes[frame][3]

    def velocity(self) -> tuple[float, float]:
        tail = self.frames[-VELOCITY_SAMPLE:]
        if len(tail) < 2:
            return (0.0, 0.0)
        (x0, y0), (x1, y1) = self.centre(tail[0]), self.centre(tail[-1])
        dt = tail[-1] - tail[0]
        vx, vy = (x1 - x0) / dt, (y1 - y0) / dt
        speed = float(np.hypot(vx, vy))
        if speed > VELOCITY_MAX_PX:
            vx, vy = vx * VELOCITY_MAX_PX / speed, vy * VELOCITY_MAX_PX / speed
        return (vx, vy)


@dataclass
class MergeResult:
    """What the post-pass did, for logging and for scripts/eval_tracking.py."""

    tracks_before: int = 0
    tracks_after: int = 0
    merges: int = 0
    chains: int = 0
    hard_cuts: int = 0
    rejected_cross_team: int = 0
    rejected_across_cut: int = 0
    unresolved_team_tracks: int = 0
    merged_pairs: list[tuple[int, int]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "tracks_before": self.tracks_before,
            "tracks_after": self.tracks_after,
            "merges": self.merges,
            "chains": self.chains,
            "hard_cuts": self.hard_cuts,
            "rejected_cross_team": self.rejected_cross_team,
            "rejected_across_cut": self.rejected_across_cut,
            "unresolved_team_tracks": self.unresolved_team_tracks,
        }


def _summarise(frames: list[list[TrackedDetection]]) -> dict[int, _Track]:
    tracks: dict[int, _Track] = {}
    for frame_number, detections in enumerate(frames):
        for det in detections:
            if det.class_name not in PLAYER_CLASSES:
                continue
            track = tracks.setdefault(det.player_id, _Track(det.player_id))
            if frame_number not in track.boxes:
                track.frames.append(frame_number)
            track.boxes[frame_number] = (det.x, det.y, det.width, det.height)
            if det.class_name == GOALKEEPER_CLASS:
                track.is_goalkeeper = True
    for track in tracks.values():
        track.frames.sort()
    return tracks


def _frame_histogram(frame_bgr: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame_bgr, CUT_PROBE_SIZE)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def _jersey_histogram(pixels: np.ndarray) -> np.ndarray:
    """Normalised hue/saturation histogram of one crop's jersey pixels."""
    hs = pixels[:, :2].astype(np.float32)
    hist, _, _ = np.histogram2d(
        hs[:, 0], hs[:, 1],
        bins=(HIST_HUE_BINS, HIST_SAT_BINS),
        range=((0.0, 180.0), (0.0, 256.0)),
    )
    return hist.astype(np.float32)


def hard_cut_frames(video_path: str, n_frames: int) -> set[int]:
    """Frame numbers at which the shot changed.

    A frame f is a cut when the HSV histogram of f differs from f-1 by more
    than CUT_HIST_DISTANCE. See this module's constants for why the
    optical-flow signal alone is not used.
    """
    cuts: set[int] = set()
    if not os.path.exists(video_path):
        return cuts
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return cuts
    try:
        previous = None
        for frame_number in range(n_frames):
            ok, raw = cap.read()
            if not ok or raw is None:
                break
            hist = _frame_histogram(raw)
            if previous is not None:
                distance = cv2.compareHist(previous, hist, cv2.HISTCMP_BHATTACHARYYA)
                if distance >= CUT_HIST_DISTANCE:
                    cuts.add(frame_number)
            previous = hist
    finally:
        cap.release()
    return cuts


def _sample_frames_for(track: _Track) -> list[int]:
    picks = track.frames
    if len(picks) <= APPEARANCE_CROPS:
        return list(picks)
    idx = np.linspace(0, len(picks) - 1, APPEARANCE_CROPS).round().astype(int)
    return [picks[i] for i in sorted(set(idx.tolist()))]


def _scan_video(video_path: str,
                frames: list[list[TrackedDetection]],
                tracks: dict[int, _Track]) -> tuple[set[int], dict[int, np.ndarray], dict[int, list[np.ndarray]]]:
    """One decode pass: hard cuts, per-track jersey histogram, per-track
    team-clustering features.

    Both readings come off the same masked torso pixels
    (team_assignment.torso_hsv_pixels), so the colour the merge gate sees is
    the colour the team gate sees.
    """
    wanted: dict[int, list[int]] = {}
    for track_id, track in tracks.items():
        for frame_number in _sample_frames_for(track):
            wanted.setdefault(frame_number, []).append(track_id)

    cuts: set[int] = set()
    histograms: dict[int, np.ndarray] = {}
    team_features: dict[int, list[np.ndarray]] = {}

    if not os.path.exists(video_path):
        return cuts, histograms, team_features
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return cuts, histograms, team_features

    try:
        previous = None
        for frame_number in range(len(frames)):
            ok, raw = cap.read()
            if not ok or raw is None:
                break

            hist = _frame_histogram(raw)
            if previous is not None:
                if cv2.compareHist(previous, hist, cv2.HISTCMP_BHATTACHARYYA) >= CUT_HIST_DISTANCE:
                    cuts.add(frame_number)
            previous = hist

            if frame_number not in wanted:
                continue
            frame_h, frame_w = raw.shape[:2]
            for track_id in wanted[frame_number]:
                x, y, w, h = tracks[track_id].boxes[frame_number]
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(frame_w, int(x + w)), min(frame_h, int(y + h))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = raw[y1:y2, x1:x2]
                pixels = torso_hsv_pixels(crop)
                if pixels is None:
                    continue
                accumulated = histograms.get(track_id)
                jersey = _jersey_histogram(pixels)
                histograms[track_id] = jersey if accumulated is None else accumulated + jersey
                feature = crop_to_feature(crop)
                if feature is not None:
                    team_features.setdefault(track_id, []).append(feature)
    finally:
        cap.release()

    for track_id, hist in histograms.items():
        total = float(hist.sum())
        if total > 0:
            histograms[track_id] = hist / total

    return cuts, histograms, team_features


def _resolve_teams(tracks: dict[int, _Track],
                   team_features: dict[int, list[np.ndarray]]) -> dict[int, str]:
    """Per-track team label for the "same team" gate.

    Goalkeepers are excluded from the clustering population exactly as
    team_assignment.py excludes them -- a keeper's kit matches neither
    outfield cluster, so letting keepers vote would drag a centroid. They
    therefore come back unlabelled, and an unlabelled track is never merged.
    """
    outfield = {track_id: feats for track_id, feats in team_features.items()
                if not tracks[track_id].is_goalkeeper}
    return resolve_track_teams(outfield)


def _candidate_merges(tracks: dict[int, _Track],
                      histograms: dict[int, np.ndarray],
                      teams: dict[int, str],
                      cuts: set[int],
                      hold_frames: int,
                      result: MergeResult) -> list[tuple[float, int, int]]:
    sorted_cuts = sorted(cuts)

    def cut_between(a_last: int, b_first: int) -> bool:
        return any(a_last < c <= b_first for c in sorted_cuts)

    candidates: list[tuple[float, int, int]] = []
    for a_id in sorted(tracks):
        a = tracks[a_id]
        a_team = teams.get(a_id)
        if a_team is None:
            continue
        a_hist = histograms.get(a_id)
        if a_hist is None:
            continue
        a_h = a.height(a.last)
        if a_h <= 0:
            continue
        vx, vy = a.velocity()
        ax, ay = a.centre(a.last)

        for b_id in sorted(tracks):
            if b_id == a_id:
                continue
            b = tracks[b_id]
            gap = b.first - a.last
            if gap < 1 or gap > hold_frames:
                continue

            b_team = teams.get(b_id)
            if b_team is None:
                continue
            if b_team != a_team:
                result.rejected_cross_team += 1
                continue
            if cut_between(a.last, b.first):
                result.rejected_across_cut += 1
                continue

            b_h = b.height(b.first)
            if b_h <= 0:
                continue
            if not (HEIGHT_RATIO_RANGE[0] <= b_h / a_h <= HEIGHT_RATIO_RANGE[1]):
                continue

            bx, by = b.centre(b.first)
            distance = float(np.hypot(bx - (ax + vx * gap), by - (ay + vy * gap))) / a_h
            if distance > MAX_PREDICTION_DISTANCE_H:
                continue

            b_hist = histograms.get(b_id)
            if b_hist is None:
                continue
            correlation = float(cv2.compareHist(a_hist, b_hist, cv2.HISTCMP_CORREL))
            if correlation < MIN_HISTOGRAM_CORRELATION:
                continue

            cost = distance / MAX_PREDICTION_DISTANCE_H + (1.0 - correlation)
            candidates.append((cost, a_id, b_id))

    candidates.sort(key=lambda c: (round(c[0], 9), c[1], c[2]))
    return candidates


def merge_reidentified_tracks(video_path: str,
                              frames: list[list[TrackedDetection]],
                              fps: float = 25.0,
                              hold_seconds: float = HOLD_SECONDS) -> MergeResult:
    """
    Merge re-appearing players back onto their original id, IN PLACE.

    `frames` is the materialised output of tracker.track_video(). Every
    merged detection's player_id is rewritten to the lowest id in its chain,
    so ids stay stable and a re-run over the same video reproduces them
    exactly.
    """
    result = MergeResult()
    tracks = _summarise(frames)
    result.tracks_before = len(tracks)
    result.tracks_after = len(tracks)
    if len(tracks) < 2:
        return result

    cuts, histograms, team_features = _scan_video(video_path, frames, tracks)
    result.hard_cuts = len(cuts)

    teams = _resolve_teams(tracks, team_features)
    result.unresolved_team_tracks = len(tracks) - len(teams)

    hold_frames = max(1, int(round(hold_seconds * fps)))
    candidates = _candidate_merges(tracks, histograms, teams, cuts, hold_frames, result)

    parent = {track_id: track_id for track_id in tracks}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    used_dead: set[int] = set()
    used_born: set[int] = set()
    for _cost, a_id, b_id in candidates:
        if a_id in used_dead or b_id in used_born:
            continue
        used_dead.add(a_id)
        used_born.add(b_id)
        root_a, root_b = find(a_id), find(b_id)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)
        result.merges += 1
        result.merged_pairs.append((a_id, b_id))

    canonical = {track_id: find(track_id) for track_id in tracks}
    members_per_root: dict[int, int] = {}
    for root in canonical.values():
        members_per_root[root] = members_per_root.get(root, 0) + 1
    result.chains = sum(1 for count in members_per_root.values() if count > 1)
    result.tracks_after = len(members_per_root)

    if result.merges:
        for detections in frames:
            for det in detections:
                if det.class_name in PLAYER_CLASSES:
                    det.player_id = canonical.get(det.player_id, det.player_id)

    return result


def merged_track_ids(frames: Iterable[list[TrackedDetection]]) -> set[int]:
    """Convenience for callers that want the surviving id set."""
    return {det.player_id for detections in frames for det in detections
            if det.class_name in PLAYER_CLASSES}
