"""
Jersey-color team assignment.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import cv2
import numpy as np

from ai.computer_vision.player_tracking.tracker import TrackedDetection
from ai.computer_vision.tactical_analysis.constants import (
    GRASS_HUE_RANGE,
    GRASS_MIN_SATURATION,
    GRASS_MIN_VALUE,
    MIN_SAMPLE_EVENTS,
    SKIN_HUE_RANGES,
    SKIN_MIN_VALUE,
    SKIN_SATURATION_RANGE,
    TEAM_ASSIGNMENT_CONFIDENCE_MIN,
    TEAM_ASSIGNMENT_CROP_TOP_FRACTION,
    TEAM_ASSIGNMENT_KMEANS_MAX_ITERS,
    TEAM_ASSIGNMENT_MIN_USABLE_PIXELS,
    TEAM_ASSIGNMENT_SAMPLE_STRIDE,
)

OUTFIELD_CLASS = "player"
GOALKEEPER_CLASS = "goalkeeper"
TEAM_LABELS = ("team-home", "team-away")



def _grass_mask(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue_lo, hue_hi = GRASS_HUE_RANGE
    return (h >= hue_lo) & (h <= hue_hi) & (s >= GRASS_MIN_SATURATION) & (v >= GRASS_MIN_VALUE)


def _skin_mask(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat_lo, sat_hi = SKIN_SATURATION_RANGE
    in_sat_val = (s >= sat_lo) & (s <= sat_hi) & (v >= SKIN_MIN_VALUE)
    hue_match = np.zeros(h.shape, dtype=bool)
    for lo, hi in SKIN_HUE_RANGES:
        hue_match |= (h >= lo) & (h <= hi)
    return hue_match & in_sat_val


def torso_hsv_pixels(crop_bgr: np.ndarray | None) -> np.ndarray | None:
    """
    The jersey pixels of one player crop, as an (N, 3) HSV array, or None
    if the crop is unusable.

    This is the shared front half of every jersey-colour reading in the
    codebase: take the top TEAM_ASSIGNMENT_CROP_TOP_FRACTION of the box
    (torso, not shorts/socks/grass), convert to HSV, and drop the pixels
    that are pitch or skin rather than kit. crop_to_feature() reduces the
    result to a 3-D clustering feature; player_tracking/reid_merge.py
    builds a hue/saturation histogram from the same pixels. Keeping the
    masking here means the two readings can never drift apart.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    top_h = max(1, int(round(crop_bgr.shape[0] * TEAM_ASSIGNMENT_CROP_TOP_FRACTION)))
    torso = crop_bgr[:top_h, :, :]
    if torso.size == 0:
        return None

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV).astype(np.float64)
    keep = ~(_grass_mask(hsv) | _skin_mask(hsv))
    if int(keep.sum()) < TEAM_ASSIGNMENT_MIN_USABLE_PIXELS:
        return None
    return hsv[keep]


def crop_to_feature(crop_bgr: np.ndarray | None) -> np.ndarray | None:
    """
    One BGR crop in, one 3-D feature vector out (or None if unusable) --
    same "caller owns the crop, this function owns the pure per-crop math"
    split as ai/computer_vision/pose_estimation/pose.py's
    estimate_body_orientation().

    Feature = [cos(2*hue), sin(2*hue), mean_saturation/255]. Hue is encoded
    on the unit circle rather than averaged directly because OpenCV hue is
    0-179 and wraps -- a red jersey's pixels can straddle H=~0 and H=~179,
    and a naive arithmetic mean of those would land near H=90 (green/cyan),
    which is exactly wrong. cos/sin averaging is the standard circular-mean
    fix, used consistently here for both per-crop feature extraction and
    (via the same encoding) the k-means distance/clustering math below.

    Value/brightness is deliberately excluded from the feature: it's
    dominated by pitch lighting/shadow (which half of the pitch is
    sunlit, floodlit vs. not), not jersey identity, and including it would
    risk the population clustering splitting on lighting conditions rather
    than team.
    """
    pixels = torso_hsv_pixels(crop_bgr)
    if pixels is None:
        return None

    h = pixels[:, 0]
    s = pixels[:, 1]
    theta = h * (2.0 * math.pi / 180.0)
    cos_h = float(np.mean(np.cos(theta)))
    sin_h = float(np.mean(np.sin(theta)))
    mean_sat = float(np.mean(s) / 255.0)
    return np.array([cos_h, sin_h, mean_sat], dtype=np.float64)



def _iter_sampled_crops(video_path: str, frames: list[list[TrackedDetection]], stride: int):
    """Yields (player_id, class_name, crop_ndarray) for every stride-sampled
    frame's player/goalkeeper detections."""
    if stride <= 0 or not os.path.exists(video_path):
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    try:
        for frame_number, detections in enumerate(frames):
            ok, raw_frame = cap.read()
            if not ok or raw_frame is None:
                break
            if frame_number % stride != 0:
                continue

            frame_h, frame_w = raw_frame.shape[:2]
            for det in detections:
                if det.class_name not in (OUTFIELD_CLASS, GOALKEEPER_CLASS):
                    continue
                x1 = max(0, int(det.x))
                y1 = max(0, int(det.y))
                x2 = min(frame_w, int(det.x + det.width))
                y2 = min(frame_h, int(det.y + det.height))
                if x2 <= x1 or y2 <= y1:
                    continue
                yield det.player_id, det.class_name, raw_frame[y1:y2, x1:x2]
    finally:
        cap.release()



def _kmeans2(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Standard Lloyd's algorithm, k=2, Euclidean distance in the 3-D feature
    space from crop_to_feature().
    """
    n = features.shape[0]
    mean = features.mean(axis=0)
    d_from_mean = np.linalg.norm(features - mean, axis=1)
    c1_idx = int(np.argmax(d_from_mean))
    d_from_c1 = np.linalg.norm(features - features[c1_idx], axis=1)
    c2_idx = int(np.argmax(d_from_c1))
    centroids = np.stack([features[c1_idx], features[c2_idx]]).astype(np.float64)

    labels = np.full(n, -1, dtype=np.int64)
    for _ in range(TEAM_ASSIGNMENT_KMEANS_MAX_ITERS):
        d0 = np.linalg.norm(features - centroids[0], axis=1)
        d1 = np.linalg.norm(features - centroids[1], axis=1)
        new_labels = (d1 < d0).astype(np.int64)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in (0, 1):
            members = features[labels == k]
            if len(members) > 0:
                centroids[k] = members.mean(axis=0)
    return labels, centroids


def _margins(features: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Per-sample cluster-separation margin -- the hand-rolled analogue of a
    silhouette score (the inter/intra-cluster distance ratio this task
    doc suggested as the no-scikit-learn alternative):
    (dist_to_OTHER_centroid - dist_to_OWN_centroid) / (dist_to_other + dist_to_own),
    """
    d0 = np.linalg.norm(features - centroids[0], axis=1)
    d1 = np.linalg.norm(features - centroids[1], axis=1)
    own = np.where(labels == 0, d0, d1)
    other = np.where(labels == 0, d1, d0)
    denom = own + other
    margin = np.divide(other - own, denom, out=np.zeros_like(denom), where=denom > 0)
    return np.clip(margin, 0.0, 1.0)


def _cluster_team_labels(centroids: np.ndarray) -> dict[int, str]:
    """
    Maps the two arbitrary cluster indices to stable "team-home"/
    "team-away" symbolic labels, ordered by centroid hue angle ascending
    so the same two jersey colors always map to the same labels across
    runs. See this module's docstring for why these labels are symbolic,
    """
    angles = [math.atan2(c[1], c[0]) for c in centroids]
    order = sorted(range(2), key=lambda i: angles[i])
    return {order[0]: TEAM_LABELS[0], order[1]: TEAM_LABELS[1]}



def resolve_track_teams(features_by_track: dict[int, list[np.ndarray]],
                        confidence_min: float = TEAM_ASSIGNMENT_CONFIDENCE_MIN) -> dict[int, str]:
    """
    Team label per track, from already-extracted crop_to_feature() vectors.

    The same k=2 clustering and the same majority-vote-weighted-by-margin
    confidence that assign_teams_with_stats() applies, but taking features
    the caller already has rather than decoding the video again, and
    returning only the tracks that clear `confidence_min`.

    player_tracking/reid_merge.py uses this as its "same team" gate: it has
    to know teams BEFORE it rewrites track ids, whereas
    assign_teams_with_stats() runs afterwards and is keyed on the final ids.
    It passes a lower `confidence_min` than the default on purpose -- see
    that module's _resolve_teams() for why a label good enough to SEPARATE
    two tracks is a weaker thing than one good enough to REPORT.
    """
    stacked: list[np.ndarray] = []
    owners: list[int] = []
    for track_id, feats in sorted(features_by_track.items()):
        for feat in feats:
            stacked.append(feat)
            owners.append(track_id)

    if len(stacked) < MIN_SAMPLE_EVENTS:
        return {}

    features = np.stack(stacked)
    owner_ids = np.array(owners)
    labels, centroids = _kmeans2(features)
    margins = _margins(features, centroids, labels)
    cluster_to_team = _cluster_team_labels(centroids)

    resolved: dict[int, str] = {}
    for track_id in np.unique(owner_ids):
        track_id = int(track_id)
        mask = owner_ids == track_id
        pid_labels = labels[mask]
        counts = np.bincount(pid_labels, minlength=2)
        majority = int(np.argmax(counts))
        vote_fraction = float(counts[majority]) / float(mask.sum())
        majority_mask = pid_labels == majority
        mean_margin = float(margins[mask][majority_mask].mean()) if np.any(majority_mask) else 0.0
        if mean_margin * vote_fraction >= confidence_min:
            resolved[track_id] = cluster_to_team[majority]
    return resolved


@dataclass
class TrackKitStats:
    """Per-track jersey-clustering evidence, for role inference."""

    player_id: int
    n_crops: int
    dist_to_nearest_centroid: float
    mean_margin: float
    vote_fraction: float
    team_id: str | None


@dataclass
class TeamAssignmentResult:
    confidence: float
    tracks: dict[int, TrackKitStats]
    dist_median: float
    dist_mad: float


def assign_teams(video_path: str, frames: list[list[TrackedDetection]]) -> float:
    """Backwards-compatible wrapper returning only the confidence scalar."""
    return assign_teams_with_stats(video_path, frames).confidence


def assign_teams_with_stats(video_path: str,
                            frames: list[list[TrackedDetection]]) -> TeamAssignmentResult:
    """
    Mutates det.team_id / det.team_assignment_confidence IN PLACE on every
    player/goalkeeper TrackedDetection across ALL frames (not just the
    stride-sampled ones used to build the clustering population) -- see
    the in-place-mutation precedent of
    """
    outfield_features: list[np.ndarray] = []
    outfield_track_ids: list[int] = []

    for player_id, class_name, crop in _iter_sampled_crops(video_path, frames, TEAM_ASSIGNMENT_SAMPLE_STRIDE):
        if class_name == GOALKEEPER_CLASS:
            continue
        feat = crop_to_feature(crop)
        if feat is None:
            continue
        outfield_features.append(feat)
        outfield_track_ids.append(player_id)

    if len(outfield_features) < MIN_SAMPLE_EVENTS:
        _mutate_all(frames, track_team={}, track_confidence={})
        return TeamAssignmentResult(confidence=0.0, tracks={},
                                    dist_median=0.0, dist_mad=0.0)

    features = np.stack(outfield_features)
    track_ids = np.array(outfield_track_ids)

    labels, centroids = _kmeans2(features)
    margins = _margins(features, centroids, labels)
    cluster_to_team = _cluster_team_labels(centroids)
    match_confidence = float(np.mean(margins))

    d0_all = np.linalg.norm(features - centroids[0], axis=1)
    d1_all = np.linalg.norm(features - centroids[1], axis=1)
    d_nearest = np.minimum(d0_all, d1_all)

    track_team: dict[int, str] = {}
    track_confidence: dict[int, float] = {}
    track_stats: dict[int, TrackKitStats] = {}
    for pid in np.unique(track_ids):
        pid = int(pid)
        mask = track_ids == pid
        pid_labels = labels[mask]
        pid_margins = margins[mask]

        counts = np.bincount(pid_labels, minlength=2)
        majority_label = int(np.argmax(counts))
        vote_fraction = float(counts[majority_label]) / float(mask.sum())

        majority_mask = pid_labels == majority_label
        mean_margin = float(pid_margins[majority_mask].mean()) if np.any(majority_mask) else 0.0

        track_confidence[pid] = mean_margin * vote_fraction
        track_team[pid] = cluster_to_team[majority_label]

        assigned = (cluster_to_team[majority_label]
                    if track_confidence[pid] >= TEAM_ASSIGNMENT_CONFIDENCE_MIN else None)
        track_stats[pid] = TrackKitStats(
            player_id=pid,
            n_crops=int(mask.sum()),
            dist_to_nearest_centroid=float(d_nearest[mask].mean()),
            mean_margin=mean_margin,
            vote_fraction=vote_fraction,
            team_id=assigned,
        )

    _mutate_all(frames, track_team, track_confidence)

    per_track = np.array([s.dist_to_nearest_centroid for s in track_stats.values()])
    med = float(np.median(per_track))
    mad = float(np.median(np.abs(per_track - med)))
    return TeamAssignmentResult(confidence=match_confidence, tracks=track_stats,
                                dist_median=med, dist_mad=mad)


def _mutate_all(
    frames: list[list[TrackedDetection]],
    track_team: dict[int, str],
    track_confidence: dict[int, float],
) -> None:
    for detections in frames:
        for det in detections:
            if det.class_name == GOALKEEPER_CLASS:
                det.team_id = None
                det.team_assignment_confidence = None
            elif det.class_name == OUTFIELD_CLASS:
                conf = track_confidence.get(det.player_id)
                det.team_assignment_confidence = conf
                if conf is not None and conf >= TEAM_ASSIGNMENT_CONFIDENCE_MIN:
                    det.team_id = track_team[det.player_id]
                else:
                    det.team_id = None
