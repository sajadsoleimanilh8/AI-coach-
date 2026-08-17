"""
MediaPipe Pose on player crops -> shoulder-line body orientation.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import cv2
import numpy as np

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

MIN_LANDMARK_VISIBILITY = 0.5


def _load_legacy_pose_solution():
    """
    Loads mediapipe.solutions.pose, with a specific, actionable error
    instead of a bare AttributeError if it isn't available.
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
    orientation_deg: float | None
    confidence: float


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
    """
    confidence = min(left_visibility, right_visibility)

    if left_visibility < MIN_LANDMARK_VISIBILITY or right_visibility < MIN_LANDMARK_VISIBILITY:
        return OrientationResult(orientation_deg=None, confidence=confidence)

    dx = right_shoulder_xy[0] - left_shoulder_xy[0]
    dy = right_shoulder_xy[1] - left_shoulder_xy[1]

    if dx == 0 and dy == 0:
        return OrientationResult(orientation_deg=None, confidence=confidence)

    angle = math.degrees(math.atan2(dy, dx)) % 360.0
    return OrientationResult(orientation_deg=angle, confidence=confidence)


_local = threading.local()


def _pose_instance():
    """One MediaPipe Pose graph per thread, built once and reused."""
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
    """
    if frame_crop is None or frame_crop.size == 0:
        return OrientationResult(orientation_deg=None, confidence=0.0)

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
