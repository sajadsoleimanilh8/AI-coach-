"""
Jersey-color team assignment.

Fills in the one gap that used to force TEAM_ASSIGNMENT_CONFIDENCE = 0.0 on
every pipeline run (see backend/pipeline/runner.py's former module
docstring): formation, press_resistance_score, defensive_positioning_score,
weak_zone_map, and pressing_intensity_score are all correctly *implemented*
already -- they just never received a usable team_id to split players by.

Approach: classical CV, not a trained model -- no new checkpoint/asset
dependency for a competition timeline. Sample player/goalkeeper crops
across the video, extract a jersey-color feature per crop (HSV, masking out
grass and skin), pool outfield-player crops into one k=2 clustering fit for
the whole match, then assign each TRACK (not each frame) a team via
majority vote across that track's own sampled frames -- a track shouldn't
flip teams because of one bad crop.

Honesty rules this module follows (same register as backend/pipeline/
runner.py's own "never fake a value" rules):
  - "team-home"/"team-away" are stable, reproducible labels for "the two
    jersey clusters found in this match," not a resolved real-world
    home/away identity -- there is no signal in jersey color alone that
    reveals which physical team is "home." Same honesty register as
    player_id being a ByteTrack tracking ID, not a resolved player
    identity (see backend/api/player_intelligence.py's PLAYER_NAME_MAP fix
    comment).
  - A track's team_id is left None whenever its own confidence falls below
    TEAM_ASSIGNMENT_CONFIDENCE_MIN, even if the match-level fit overall was
    good. No forced guesses.
  - Goalkeepers are never assigned by color-proximity to an outfield
    cluster. This isn't a time-saving shortcut -- a keeper's kit is
    *specifically chosen* to look different from both outfield teams
    (kit-clash-avoidance), so "nearest outfield cluster by color" is a
    systematically anti-correlated signal for this class, not merely a
    noisy one. team_id stays None for every goalkeeper detection,
    unconditionally.

Calibration status: the HSV grass/skin bounds and the 3-D color-feature
design below are a defensible starting point, not numbers tuned against
real broadcast footage -- same caveat as SCANS_PER_MINUTE_TARGET/
IDEAL_SQUARENESS_DEG in constants.py. Treat the relative behavior (well-
separated kits cluster confidently, similar kits don't) as sound; treat the
exact HSV cutoffs as a first pass.
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


# ======================================================================
# Per-crop color feature
# ======================================================================

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
    theta = h * (2.0 * math.pi / 180.0)  # OpenCV hue period is 180, not 360
    cos_h = float(np.mean(np.cos(theta)))
    sin_h = float(np.mean(np.sin(theta)))
    mean_sat = float(np.mean(s) / 255.0)
    return np.array([cos_h, sin_h, mean_sat], dtype=np.float64)


# ======================================================================
# Video-crop sampling (mirrors backend/pipeline/runner.py's
# _estimate_orientations(): track_video()'s own YOLO pass never exposes
# decoded pixel arrays, only box metadata, so this does its own second
# sequential cv2.VideoCapture read, staying frame-synced by reading -- not
# seeking -- every frame).
# ======================================================================

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
                break  # end of video, or an unrecoverable decode failure
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
                    continue  # degenerate bbox
                yield det.player_id, det.class_name, raw_frame[y1:y2, x1:x2]
    finally:
        cap.release()


# ======================================================================
# k=2 clustering (hand-rolled, plain numpy -- no scikit-learn dependency;
# not installed in this environment, and adding it just for a k=2 fit
# would mean coordinating a new dependency across two separate
# requirements files for no real accuracy gain over ~20 lines of Lloyd's
# algorithm).
# ======================================================================

def _kmeans2(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Standard Lloyd's algorithm, k=2, Euclidean distance in the 3-D feature
    space from crop_to_feature().

    Deterministic, seedless initialization (stronger than "a fixed RNG
    seed," which would still tie reproducibility to numpy's RNG
    implementation across versions): centroid 1 = the pooled point
    farthest from the population mean; centroid 2 = the point farthest
    from centroid 1. The same video always produces the same clustering.

    Returns (labels, centroids): labels is an (N,) int array of 0/1,
    centroids is a (2, 3) array.
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
    clipped to [0, 1]. 1.0 = sitting right on its own centroid, 0.0 = as
    close (or closer) to the other cluster as its own -- i.e. not
    confidently assigned, clipped rather than left negative since a
    negative margin and a zero margin both mean the same thing for our
    purposes here: "don't trust this assignment."
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
    not a resolved real-world home/away identity.
    """
    angles = [math.atan2(c[1], c[0]) for c in centroids]
    order = sorted(range(2), key=lambda i: angles[i])
    return {order[0]: TEAM_LABELS[0], order[1]: TEAM_LABELS[1]}


# ======================================================================
# Entry point
# ======================================================================

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
    """Per-track jersey-clustering evidence, for role inference.

    Exposed because "which kit does this track match" and "which team is
    this track on" are different questions, and the second one throws away
    the evidence the first one needs. A referee matches NEITHER kit; a
    track that flip-flops between the two matches both badly. Both come
    out of assign_teams() as team_id=None, and only these numbers tell
    them apart.
    """

    player_id: int
    n_crops: int
    #: Mean Euclidean distance, in crop_to_feature() space, to the NEARER
    #: of the two team centroids. Large = the kit resembles neither team.
    dist_to_nearest_centroid: float
    mean_margin: float
    vote_fraction: float
    team_id: str | None


@dataclass
class TeamAssignmentResult:
    confidence: float
    tracks: dict[int, TrackKitStats]
    #: Population median/MAD of dist_to_nearest_centroid, so a consumer can
    #: judge "unusually far" relative to THIS video rather than against an
    #: absolute threshold that would not transfer between kits or lighting.
    dist_median: float
    dist_mad: float


def assign_teams(video_path: str, frames: list[list[TrackedDetection]]) -> float:
    """Backwards-compatible wrapper returning only the confidence scalar.

    Kept as the primary entry point because seven downstream call sites and
    the existing test suite use this signature. Use
    assign_teams_with_stats() when the per-track clustering evidence is
    needed (role inference)."""
    return assign_teams_with_stats(video_path, frames).confidence


def assign_teams_with_stats(video_path: str,
                            frames: list[list[TrackedDetection]]) -> TeamAssignmentResult:
    """
    Mutates det.team_id / det.team_assignment_confidence IN PLACE on every
    player/goalkeeper TrackedDetection across ALL frames (not just the
    stride-sampled ones used to build the clustering population) -- see
    the in-place-mutation precedent of
    ai/computer_vision/player_tracking/trajectory.py's
    attach_body_orientation(). Each frame's TrackedDetection owns its own
    copy of these fields, so a track-level decision made from the sampled
    subset must be explicitly propagated across every frame that track
    appears in, including frames never cropped -- skipping that second
    pass would silently leave non-sampled frames' rows null even for a
    confidently-assigned track.

    Call this BEFORE building pitch trajectories: enrich_with_pitch_coordinates()
    copies det.team_id onto TrackingPoint verbatim, so team_id must already
    be final by the time that runs. Does not need fps -- sampling is
    frame-number-indexed (TEAM_ASSIGNMENT_SAMPLE_STRIDE), not time-indexed,
    same as POSE_SAMPLE_STRIDE.

    Returns the match-level confidence scalar (mean per-crop margin across
    the whole clustering population) -- this REPLACES the old hardcoded
    TEAM_ASSIGNMENT_CONFIDENCE constant in backend/pipeline/runner.py, fed
    into the same 7 downstream call sites (detect_formation,
    compute_compactness, compute_formation_stability, compute_weak_zones,
    compute_pressing_intensity, score_press_resistance,
    score_defensive_positioning) unchanged.
    """
    outfield_features: list[np.ndarray] = []
    outfield_track_ids: list[int] = []

    for player_id, class_name, crop in _iter_sampled_crops(video_path, frames, TEAM_ASSIGNMENT_SAMPLE_STRIDE):
        if class_name == GOALKEEPER_CLASS:
            continue  # never fed into the outfield k=2 fit -- see module docstring
        feat = crop_to_feature(crop)
        if feat is None:
            continue
        outfield_features.append(feat)
        outfield_track_ids.append(player_id)

    if len(outfield_features) < MIN_SAMPLE_EVENTS:
        # Not enough usable crops to cluster at all (short clip, heavy
        # occlusion, grass/skin masking left almost nothing) -- leave
        # everyone unassigned rather than fit k-means on a near-empty,
        # unreliable population. _mutate_all() below still runs so
        # goalkeepers correctly get team_id=None too.
        _mutate_all(frames, track_team={}, track_confidence={})
        return TeamAssignmentResult(confidence=0.0, tracks={},
                                    dist_median=0.0, dist_mad=0.0)

    features = np.stack(outfield_features)
    track_ids = np.array(outfield_track_ids)

    labels, centroids = _kmeans2(features)
    margins = _margins(features, centroids, labels)
    cluster_to_team = _cluster_team_labels(centroids)
    match_confidence = float(np.mean(margins))

    # Distance to the NEARER centroid, per crop. This is the kit-outlier
    # signal role inference needs: the margin above is scale-free (a ratio)
    # and so cannot distinguish "equidistant because it sits between the
    # two kits" from "equidistant because it is far from both".
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
        majority_label = int(np.argmax(counts))  # ties broken toward index 0 (numpy argmax convention)
        vote_fraction = float(counts[majority_label]) / float(mask.sum())

        majority_mask = pid_labels == majority_label
        mean_margin = float(pid_margins[majority_mask].mean()) if np.any(majority_mask) else 0.0

        # Confidence combines both signals on purpose: mean margin alone
        # can look fine even for a track that flip-flopped ~50/50 between
        # clusters, since each individual crop can be well-separated from
        # whichever cluster it happened to land near that frame. Requiring
        # vote_fraction too catches that case.
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

    # Median + MAD rather than mean + std: with 22 outfield players and 1-3
    # officials, the officials ARE the outliers this statistic is used to
    # find, and they would inflate a standard deviation enough to hide
    # themselves.
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
                conf = track_confidence.get(det.player_id)  # None if this track had zero usable sampled crops
                det.team_assignment_confidence = conf
                if conf is not None and conf >= TEAM_ASSIGNMENT_CONFIDENCE_MIN:
                    det.team_id = track_team[det.player_id]
                else:
                    det.team_id = None
            # ball / referee detections are untouched -- team_id was
            # already None from track_video() and stays that way.
