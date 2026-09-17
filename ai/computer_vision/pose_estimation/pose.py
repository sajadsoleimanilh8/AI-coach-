"""
MediaPipe Pose on player crops -> shoulder-line body orientation.

Mirrors tactical_analysis/homography.py's pattern: every result carries
its own confidence, never just a bare angle. Per docs/data analysis.md
§0.2's precedent (homography_confidence, team_assignment_confidence), a
body-orientation reading that "exists" but is low-confidence must be
distinguishable from one that doesn't exist at all -- see
backend/database/models.py::PlayerTracking.body_orientation_confidence.

Landmark indices (MediaPipe Pose, BlazePose topology):
    11 = left shoulder
    12 = right shoulder
We deliberately use only these two -- hips (23/24) would also work and are
more robust to arm movement, but shoulders are more reliably visible on
tight broadcast player crops where the lower body is often cropped out or
occluded by other players.

This module does NOT decide sampling stride or which frames to run on --
that's an orchestration decision made by backend/pipeline/runner.py
(POSE_SAMPLE_STRIDE), kept separate so this module stays testable in
isolation with synthetic landmark coordinates, the same way
tactical_analysis/homography.py's tests don't need a real broadcast frame.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import cv2
import numpy as np

# Landmark indices, named instead of magic numbers at every call site.
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

# Below this, MediaPipe itself considers the landmark too unreliable to
# trust (BlazePose's own convention) -- treat as "not visible", not as a
# noisy-but-usable reading.
MIN_LANDMARK_VISIBILITY = 0.5


def _load_legacy_pose_solution():
    """
    Loads mediapipe.solutions.pose, with a specific, actionable error
    instead of a bare AttributeError if it isn't available.

    KNOWN GOTCHA (found while wiring this up): some `pip install
    mediapipe` wheels -- depending on platform/Python version -- do NOT
    ship the legacy `mp.solutions` API at all, only the newer Tasks API
    (`mediapipe.tasks.python.vision.PoseLandmarker`, which needs a
    separately-downloaded `.task` model asset). `requirements.txt` lists
    `mediapipe` unpinned, so which one you get is not guaranteed to be
    consistent across the team's machines. If this raises on your
    machine: either pin a mediapipe version confirmed to ship
    `solutions` (check with `python -c "import mediapipe;
    print(hasattr(mediapipe, 'solutions'))"` before relying on this
    module), or port this function to the Tasks API -- the angle math in
    shoulder_line_angle() doesn't change either way, only how you get
    (x, y, visibility) out of the model.
    """
    import mediapipe as mp

    if not hasattr(mp, "solutions"):
        raise RuntimeError(
            "This mediapipe install does not expose mediapipe.solutions.pose "
            "(legacy Solutions API). Installed version may only support the "
            "newer Tasks API. Pin a mediapipe version known to ship "
            "`solutions` (test with "
            "`python -c \"import mediapipe; print(hasattr(mediapipe,'solutions'))\"`), "
            "or port pose.py to mediapipe.tasks.python.vision.PoseLandmarker."
        )
    return mp.solutions.pose


@dataclass
class OrientationResult:
    orientation_deg: float | None  # 0-360, angle of the shoulder line vs. the image x-axis
    confidence: float              # 0-1, min visibility of the two shoulder landmarks (0.0 if neither visible)


def shoulder_line_angle(
    left_shoulder_xy: tuple[float, float],
    right_shoulder_xy: tuple[float, float],
    left_visibility: float,
    right_visibility: float,
) -> OrientationResult:
    """
    Pure function: given two shoulder landmark positions (already
    extracted from a MediaPipe result, in any consistent 2D coordinate
    space -- normalized image coords or pixel coords, doesn't matter for
    an angle) and their visibility scores, compute the shoulder-line
    orientation angle and its confidence.

    Split out from estimate_body_orientation() specifically so this can
    be unit-tested with synthetic coordinates -- no MediaPipe model, no
    real image needed (same reasoning as
    tactical_analysis/tests/test_homography.py's synthetic-geometry
    approach: the angle math is what can silently break, not whether
    MediaPipe itself works).

    Returns:
        OrientationResult with orientation_deg=None if either shoulder's
        visibility is below MIN_LANDMARK_VISIBILITY -- a low-visibility
        landmark's (x, y) is frequently a hallucinated extrapolation, not
        a real observation, so an angle computed from it would be
        confident-looking noise, not a low-confidence-but-real reading.
    """
    confidence = min(left_visibility, right_visibility)

    if left_visibility < MIN_LANDMARK_VISIBILITY or right_visibility < MIN_LANDMARK_VISIBILITY:
        return OrientationResult(orientation_deg=None, confidence=confidence)

    dx = right_shoulder_xy[0] - left_shoulder_xy[0]
    dy = right_shoulder_xy[1] - left_shoulder_xy[1]

    if dx == 0 and dy == 0:
        # Degenerate case: both landmarks reported at the identical point
        # (can happen on a very poor detection that still cleared the
        # visibility threshold). atan2(0, 0) is defined as 0.0 in Python,
        # which would silently read as "facing exactly along the x-axis"
        # -- a specific, wrong-looking fact. Report as unmeasured instead.
        return OrientationResult(orientation_deg=None, confidence=confidence)

    angle = math.degrees(math.atan2(dy, dx)) % 360.0
    return OrientationResult(orientation_deg=angle, confidence=confidence)


_local = threading.local()


def _pose_instance():
    """One MediaPipe Pose graph per thread, built once and reused.

    This function's predecessor constructed and tore down a Pose instance on
    every call. Its docstring said a persistent instance "would be faster but
    is not thread-safe across concurrent Celery workers", and deferred the
    decision until the stage had been latency-profiled. It has been now, and
    the numbers are not marginal:

        pose_estimation   58.47s of a 70.37s run   83.1% of total wall-clock
        construct-per-call   93.7 ms
        reused instance      18.5 ms   (5.1x)

    Constructing the graph, not running inference on it, was five sixths of
    the cost. `static_image_mode=True` means the graph carries no tracking
    state between calls, so reuse cannot leak one crop's result into the next
    -- the only real hazard was concurrency, and threading.local() answers
    that directly: each worker thread gets its own instance, so two threads
    never share one graph. A process-wide singleton would have been the
    unsafe version.

    Never closed on purpose. The instance lives as long as its thread, which
    for a Celery worker is the process; closing it after each stage would
    reintroduce the construction cost this exists to avoid.
    """
    pose = getattr(_local, "pose", None)
    if pose is None:
        mp_pose = _load_legacy_pose_solution()
        pose = mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
            model_complexity=0,
        )
        _local.pose = pose
    return pose


def estimate_body_orientation(frame_crop: np.ndarray) -> OrientationResult:
    """
    Run MediaPipe Pose on a single player crop and return shoulder-line
    orientation.

    Args:
        frame_crop: BGR image (as read by cv2), cropped tightly to one
            player's bounding box. Caller (runner.py) is responsible for
            the crop -- this function does one player, one frame, no
            batching, so it can be called per-(player, sampled-frame) pair.

    Returns:
        OrientationResult. orientation_deg=None, confidence=0.0 if no pose
        landmarks were detected at all in the crop (e.g. crop too small,
        player facing directly away with shoulders occluded, motion blur).
    """
    if frame_crop is None or frame_crop.size == 0:
        return OrientationResult(orientation_deg=None, confidence=0.0)

    # MediaPipe's Pose.process() expects RGB, but cv2 decodes BGR, so the crop
    # must be converted. Forgetting this fails quietly: BlazePose still finds a
    # person in a colour-swapped image often enough to look plausible, but
    # landmark visibility drops and readings fall under MIN_LANDMARK_VISIBILITY
    # and are discarded as unmeasured, instead of raising.
    rgb_crop = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2RGB)

    results = _pose_instance().process(rgb_crop)

    if not results.pose_landmarks:
        return OrientationResult(orientation_deg=None, confidence=0.0)

    lm = results.pose_landmarks.landmark
    left, right = lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER]

    return shoulder_line_angle(
        left_shoulder_xy=(left.x, left.y),
        right_shoulder_xy=(right.x, right.y),
        left_visibility=left.visibility,
        right_visibility=right.visibility,
    )
