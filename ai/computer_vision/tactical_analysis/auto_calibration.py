"""
Automatic pitch calibration from the trained 32-keypoint pose model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ai.computer_vision.frame_data import (
    CalibrationSource,
    CalibrationState,
    CameraMotion,
    CameraState,
    FieldRegion,
)
from ai.computer_vision.tactical_analysis.constants import (
    CALIBRATION_FALLBACK_DECAY,
    CALIBRATION_MAX_FALLBACK_FRAMES,
    CALIBRATION_MAX_JUMP_M,
    CALIBRATION_SMOOTHING,
    CAMERA_CUT_SHIFT_PX,
    CAMERA_STATIC_SHIFT_PX,
    HOMOGRAPHY_CONFIDENCE_MIN,
    HOMOGRAPHY_RANSAC_THRESHOLD_M,
    MIN_CALIBRATION_POINTS,
)
from ai.computer_vision.tactical_analysis.homography import (
    HomographyResult,
    compute_homography,
)
from ai.computer_vision.tactical_analysis.pitch_keypoints import (
    keypoints_to_correspondences,
)

logger = logging.getLogger(__name__)


@dataclass
class AutoCalibrationResult:
    """Outcome of ONE attempt to calibrate a single frame."""

    homography: HomographyResult | None
    keypoints_px: list[tuple[float, float]] = field(default_factory=list)
    keypoint_indices: list[int] = field(default_factory=list)
    detection_confidence: float | None = None
    n_keypoints_visible: int = 0
    reason: str | None = None
    strategy: str | None = None

    @property
    def ok(self) -> bool:
        return self.homography is not None



def preprocess_frame(
    frame: np.ndarray, imgsz: int, mode: str
) -> tuple[np.ndarray, float, float]:
    """
    Prepares a frame for the pose model and returns (image, sx, sy), where
    multiplying a detected keypoint by (sx, sy) maps it back to ORIGINAL
    frame pixels.
    """
    if mode == "native":
        return frame, 1.0, 1.0
    if mode != "stretch_square":
        logger.warning(
            "unknown calibration preprocess mode %r, falling back to 'native'; "
            "expected 'stretch_square' or 'native'", mode)
        return frame, 1.0, 1.0

    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return frame, 1.0, 1.0
    square = cv2.resize(frame, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    return square, w / float(imgsz), h / float(imgsz)



def extract_keypoints(
    result: Any, min_confidence: float,
    scale: tuple[float, float] = (1.0, 1.0),
) -> tuple[dict[int, tuple[float, float]], dict[int, float], float | None]:
    """
    Pulls {index: (px, py)} and {index: confidence} out of one ultralytics
    pose Result.
    """
    boxes = getattr(result, "boxes", None)
    kp = getattr(result, "keypoints", None)
    if boxes is None or kp is None or len(boxes) == 0 or kp.data is None or len(kp.data) == 0:
        return {}, {}, None

    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
    best = int(np.argmax(confs))
    det_conf = float(confs[best])

    data = kp.data[best]
    data = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)

    sx, sy = scale
    keypoints: dict[int, tuple[float, float]] = {}
    confidences: dict[int, float] = {}
    for idx, row in enumerate(data):
        if len(row) >= 3:
            x, y, c = float(row[0]), float(row[1]), float(row[2])
        else:
            x, y, c = float(row[0]), float(row[1]), 1.0
        confidences[idx] = c
        if c < min_confidence:
            continue
        if x == 0.0 and y == 0.0:
            continue
        keypoints[idx] = (x * sx, y * sy)

    return keypoints, confidences, det_conf


def calibrate_from_keypoints(
    keypoints: dict[int, tuple[float, float]],
    confidences: dict[int, float] | None = None,
    min_confidence: float = 0.0,
    detection_confidence: float | None = None,
    strategy: str | None = None,
) -> AutoCalibrationResult:
    """
    Turns detected keypoints into a HomographyResult via the EXISTING
    compute_homography().
    """
    pixel_pts, pitch_pts, used = keypoints_to_correspondences(
        keypoints, min_confidence=min_confidence, confidences=confidences
    )

    if len(pixel_pts) < MIN_CALIBRATION_POINTS:
        return AutoCalibrationResult(
            homography=None,
            keypoints_px=pixel_pts,
            keypoint_indices=used,
            detection_confidence=detection_confidence,
            n_keypoints_visible=len(pixel_pts),
            reason=(f"only {len(pixel_pts)} usable keypoints, need "
                    f">= {MIN_CALIBRATION_POINTS}"),
            strategy=strategy,
        )

    method = cv2.RANSAC if len(pixel_pts) > MIN_CALIBRATION_POINTS else 0
    try:
        homography = compute_homography(
            np.array(pixel_pts), np.array(pitch_pts),
            method=method,
            ransac_reproj_threshold=HOMOGRAPHY_RANSAC_THRESHOLD_M,
        )
    except ValueError as exc:
        return AutoCalibrationResult(
            homography=None,
            keypoints_px=pixel_pts,
            keypoint_indices=used,
            detection_confidence=detection_confidence,
            n_keypoints_visible=len(pixel_pts),
            reason=f"homography fit failed: {exc}",
            strategy=strategy,
        )

    return AutoCalibrationResult(
        homography=homography,
        keypoints_px=pixel_pts,
        keypoint_indices=used,
        detection_confidence=detection_confidence,
        n_keypoints_visible=len(pixel_pts),
        strategy=strategy,
    )


def to_calibration_state(
    result: AutoCalibrationResult,
    source: CalibrationSource = CalibrationSource.model,
    field_region: FieldRegion | None = None,
    solved_on_frame: int | None = None,
    frame_size: tuple[int, int] | None = None,
) -> CalibrationState:
    """
    Wraps an AutoCalibrationResult in the shared CalibrationState and runs
    the validity gate ONCE (confidence + geometric consistency against the
    field model's pitch polygon + keypoint spread across the frame).
    """
    if not result.ok:
        return CalibrationState.unavailable(result.reason or "auto-calibration failed")

    h = result.homography

    gate_keypoints = result.keypoints_px
    if h.inlier_mask and len(h.inlier_mask) == len(result.keypoints_px):
        gate_keypoints = [p for p, keep in zip(result.keypoints_px, h.inlier_mask)
                          if keep]
    state = CalibrationState(
        H=h.H,
        confidence=h.confidence,
        reprojection_error_m=h.reprojection_error_m,
        n_points=h.n_points,
        source=source,
        solved_on_frame=solved_on_frame,
        n_inliers=h.n_inliers,
        inlier_ratio=h.inlier_ratio,
        solved_confidence=h.confidence,
        carried_frames=0,
    )
    return state.evaluate(field_region=field_region,
                          keypoints_px=gate_keypoints,
                          frame_size=frame_size)


def calibration_state_from_manual(
    H, record: dict, field_region: FieldRegion | None = None,
    solved_on_frame: int | None = None,
    frame_size: tuple[int, int] | None = None,
) -> CalibrationState:
    """
    The manual path's entry into the SAME representation.
    """
    if H is None:
        return CalibrationState.unavailable("manual calibration file had no homography")

    keypoints_px = [tuple(p["pixel"]) for p in record.get("points", []) if "pixel" in p]
    state = CalibrationState(
        H=np.asarray(H),
        confidence=float(record.get("confidence", 0.0)),
        reprojection_error_m=record.get("reprojection_error_m"),
        n_points=int(record.get("n_points", len(keypoints_px))),
        source=CalibrationSource.manual,
        solved_on_frame=solved_on_frame if solved_on_frame is not None
        else record.get("frame_number"),
    )
    return state.evaluate(field_region=field_region, keypoints_px=keypoints_px,
                          frame_size=frame_size)



class CameraMotionDetector:
    """
    Frame-to-frame camera shift, by sparse optical flow.
    """

    def __init__(
        self,
        static_threshold_px: float = CAMERA_STATIC_SHIFT_PX,
        cut_threshold_px: float = CAMERA_CUT_SHIFT_PX,
        max_corners: int = 200,
        downscale: float = 0.5,
    ) -> None:
        self.static_threshold_px = static_threshold_px
        self.cut_threshold_px = cut_threshold_px
        self.max_corners = max_corners
        self.downscale = downscale
        self._prev_gray: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_gray = None

    def update(self, frame: np.ndarray) -> CameraState:
        """Returns this frame's CameraState relative to the previous call."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.downscale != 1.0:
                gray = cv2.resize(gray, None, fx=self.downscale, fy=self.downscale)
        except cv2.error:
            return CameraState(motion=CameraMotion.unknown)

        prev = self._prev_gray
        self._prev_gray = gray

        if prev is None or prev.shape != gray.shape:
            return CameraState(motion=CameraMotion.unknown, recalibration_advised=True)

        pts = cv2.goodFeaturesToTrack(prev, maxCorners=self.max_corners,
                                      qualityLevel=0.01, minDistance=8)
        if pts is None or len(pts) < 8:
            return CameraState(motion=CameraMotion.unknown, recalibration_advised=False)

        nxt, status, _err = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None)
        if nxt is None or status is None:
            return CameraState(motion=CameraMotion.unknown)

        ok = status.reshape(-1).astype(bool)
        if ok.sum() < 8:
            return CameraState(motion=CameraMotion.cut, shift_px=None,
                               recalibration_advised=True)

        deltas = (nxt.reshape(-1, 2)[ok] - pts.reshape(-1, 2)[ok]) / self.downscale
        shift = float(np.median(np.linalg.norm(deltas, axis=1)))

        if shift >= self.cut_threshold_px:
            motion = CameraMotion.cut
        elif shift >= self.static_threshold_px:
            motion = CameraMotion.panning
        else:
            motion = CameraMotion.static

        return CameraState(
            motion=motion,
            shift_px=shift,
            recalibration_advised=motion is not CameraMotion.static,
        )



def homography_jump_m(
    H_old: np.ndarray, H_new: np.ndarray, frame_size: tuple[int, int]
) -> float:
    """
    How far, in PITCH METRES, the two homographies disagree about the same
    image.
    """
    w, h = frame_size
    probes = np.array([[w * 0.25, h * 0.25], [w * 0.75, h * 0.25],
                       [w * 0.25, h * 0.75], [w * 0.75, h * 0.75]],
                      dtype=np.float64).reshape(-1, 1, 2)
    try:
        old = np.asarray(H_old, dtype=np.float64)
        new = np.asarray(H_new, dtype=np.float64)
        if old.shape != (3, 3) or new.shape != (3, 3):
            return float("inf")
        if not (np.isfinite(old).all() and np.isfinite(new).all()):
            return float("inf")
        if abs(np.linalg.det(old)) < 1e-12 or abs(np.linalg.det(new)) < 1e-12:
            return float("inf")
        a = cv2.perspectiveTransform(probes, old).reshape(-1, 2)
        b = cv2.perspectiveTransform(probes, new).reshape(-1, 2)
    except (cv2.error, np.linalg.LinAlgError):
        return float("inf")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("inf")
    return float(np.median(np.linalg.norm(a - b, axis=1)))


def blend_homographies(
    H_old: np.ndarray, H_new: np.ndarray, frame_size: tuple[int, int],
    weight_new: float,
) -> np.ndarray | None:
    """
    Smooths toward `H_new` and returns a homography that is still a valid
    projective transform, or None when the blend cannot be formed.
    """
    weight_new = max(0.0, min(1.0, float(weight_new)))
    if weight_new >= 1.0:
        return np.asarray(H_new, dtype=np.float64)
    if weight_new <= 0.0:
        return np.asarray(H_old, dtype=np.float64)

    w, h = frame_size
    probes = np.array([[w * 0.25, h * 0.25], [w * 0.75, h * 0.25],
                       [w * 0.25, h * 0.75], [w * 0.75, h * 0.75]],
                      dtype=np.float64)
    try:
        old = np.asarray(H_old, dtype=np.float64)
        new = np.asarray(H_new, dtype=np.float64)
        if old.shape != (3, 3) or new.shape != (3, 3):
            return None
        if not (np.isfinite(old).all() and np.isfinite(new).all()):
            return None
        src = probes.reshape(-1, 1, 2)
        a = cv2.perspectiveTransform(src, old).reshape(-1, 2)
        b = cv2.perspectiveTransform(src, new).reshape(-1, 2)
    except (cv2.error, np.linalg.LinAlgError):
        return None
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return None

    blended = a * (1.0 - weight_new) + b * weight_new
    H, _mask = cv2.findHomography(probes, blended, method=0)
    if H is None or not np.isfinite(H).all():
        return None
    return H



class AutoCalibrator:
    """
    Persistent pose-model wrapper + temporal-stability state machine.
    """

    def __init__(
        self,
        model: Any | None = None,
        model_name: str = "calibration",
        kpt_conf_min: float | None = None,
        imgsz: int | None = None,
        conf: float | None = None,
        max_jump_m: float = CALIBRATION_MAX_JUMP_M,
        motion_detector: CameraMotionDetector | None = None,
        device: str | None = None,
        preprocess: str | None = None,
        kpt_conf_relaxed: float | None = None,
        max_fallback_frames: int = CALIBRATION_MAX_FALLBACK_FRAMES,
        fallback_decay: float = CALIBRATION_FALLBACK_DECAY,
        smoothing: float = CALIBRATION_SMOOTHING,
    ) -> None:
        from configs import registry

        spec = registry.get_model(model_name)
        inference = spec.inference
        self.kpt_conf_min = (kpt_conf_min if kpt_conf_min is not None
                             else float(inference.get("kpt_conf_min", 0.5)))
        self.imgsz = imgsz if imgsz is not None else int(inference.get("imgsz", 960))
        self.conf = conf if conf is not None else float(inference.get("conf", 0.3))
        self.preprocess = (preprocess if preprocess is not None
                           else str(inference.get("preprocess", "native")))
        self.kpt_conf_relaxed = (
            kpt_conf_relaxed if kpt_conf_relaxed is not None
            else float(inference.get("kpt_conf_relaxed", 0.35)))
        self.max_jump_m = max_jump_m
        self.max_fallback_frames = max_fallback_frames
        self.fallback_decay = fallback_decay
        self.smoothing = smoothing
        self.device = device

        if model is None:
            from ultralytics import YOLO
            model = YOLO(str(spec.require_checkpoint()))
        self.model = model

        self.motion = motion_detector or CameraMotionDetector()

        self.stable: CalibrationState | None = None
        self.last_camera: CameraState = CameraState()
        self.carried_frames = 0
        self.n_attempted = 0
        self.n_accepted = 0
        self.n_reused = 0
        self.n_rejected_jump = 0
        self.n_rejected_invalid = 0
        self.n_strict = 0
        self.n_relaxed = 0
        self.n_fallback_expired = 0
        self.n_smoothed = 0

    def calibrate_frame(
        self, frame: np.ndarray, field_region: FieldRegion | None = None,
        frame_id: int | None = None,
    ) -> AutoCalibrationResult:
        """
        Runs the model on one frame and fits a homography. No reuse, no
        jump rejection -- the plain "calibrate this image" entry point,
        used directly by tests and by the QA overlay.
        """
        self.n_attempted += 1
        image, sx, sy = preprocess_frame(frame, self.imgsz, self.preprocess)
        try:
            predict_kwargs = dict(conf=self.conf, imgsz=self.imgsz, verbose=False)
            if self.device is not None:
                predict_kwargs["device"] = self.device
            result = self.model.predict(image, **predict_kwargs)[0]
        except Exception as exc:  # noqa: BLE001 -- see docstring
            logger.warning("calibration model inference failed: %s", exc)
            return AutoCalibrationResult(homography=None,
                                         reason=f"inference failed: {exc}")

        floor = min(self.kpt_conf_min, self.kpt_conf_relaxed)
        keypoints, confidences, det_conf = extract_keypoints(
            result, floor, scale=(sx, sy))
        if not keypoints:
            return AutoCalibrationResult(
                homography=None, detection_confidence=det_conf,
                reason="no pitch keypoints above the visibility floor",
            )

        strict = calibrate_from_keypoints(
            keypoints, confidences=confidences,
            min_confidence=self.kpt_conf_min, detection_confidence=det_conf,
            strategy="strict",
        )
        if strict.ok or self.kpt_conf_relaxed >= self.kpt_conf_min:
            return strict

        relaxed = calibrate_from_keypoints(
            keypoints, confidences=confidences,
            min_confidence=self.kpt_conf_relaxed, detection_confidence=det_conf,
            strategy="relaxed",
        )
        if not relaxed.ok:
            return strict
        return relaxed

    def calibrate(
        self,
        frame: np.ndarray,
        frame_id: int,
        field_region: FieldRegion | None = None,
    ) -> tuple[CalibrationState, CameraState]:
        """
        The per-frame entry point the pipeline uses.
        """
        camera = self.motion.update(frame)
        self.last_camera = camera
        h_px, w_px = frame.shape[:2]

        if (self.stable is not None and self.stable.valid
                and camera.motion is CameraMotion.static):
            carried, camera_out = self._carry(camera, frame_id)
            if carried.valid:
                return carried, camera_out

        attempt = self.calibrate_frame(frame, field_region=field_region, frame_id=frame_id)
        state = to_calibration_state(
            attempt, source=CalibrationSource.model,
            field_region=field_region, solved_on_frame=frame_id,
            frame_size=(w_px, h_px),
        )

        if not state.valid:
            self.n_rejected_invalid += 1
            if self.stable is not None and camera.motion is not CameraMotion.cut:
                carried, camera_out = self._carry(camera, frame_id)
                if carried.valid:
                    return carried, camera_out
            return state, camera

        if self.stable is not None and self.stable.valid and self.stable.H is not None:
            jump = homography_jump_m(self.stable.H, state.H, (w_px, h_px))
            if jump > self.max_jump_m and camera.motion is not CameraMotion.cut:
                self.n_rejected_jump += 1
                logger.info(
                    "frame %s: rejecting recalibration, implausible jump %.1f m "
                    "(> %.1f m) from the prior stable calibration",
                    frame_id, jump, self.max_jump_m,
                )
                carried, camera_out = self._carry(camera, frame_id)
                if carried.valid:
                    return carried, camera_out

            elif camera.motion is not CameraMotion.cut and self.smoothing < 1.0:
                blended = blend_homographies(
                    self.stable.H, state.H, (w_px, h_px), self.smoothing)
                if blended is not None:
                    candidate = CalibrationState(
                        H=blended, confidence=state.confidence,
                        reprojection_error_m=state.reprojection_error_m,
                        n_points=state.n_points, source=CalibrationSource.model,
                        solved_on_frame=frame_id, n_inliers=state.n_inliers,
                        inlier_ratio=state.inlier_ratio,
                        solved_confidence=state.confidence,
                    ).evaluate(field_region=field_region,
                               keypoints_px=attempt.keypoints_px,
                               frame_size=(w_px, h_px))
                    if candidate.valid:
                        self.n_smoothed += 1
                        state = candidate

        self.n_accepted += 1
        if attempt.strategy == "relaxed":
            self.n_relaxed += 1
        else:
            self.n_strict += 1
        self.stable = state
        self.carried_frames = 0
        return state, camera

    def _carry(
        self, camera: CameraState, frame_id: int | None,
    ) -> tuple[CalibrationState, CameraState]:
        """
        Serves the last stable calibration for one more frame, with its
        confidence decayed, or returns an INVALID state once the fallback
        has expired.
        """
        if self.stable is None or not self.stable.valid:
            return CalibrationState.unavailable(
                "no previous valid calibration to carry forward"), camera

        carried_frames = self.carried_frames + 1
        solved_on = self.stable.solved_on_frame
        if carried_frames > self.max_fallback_frames:
            self.n_fallback_expired += 1
            self.stable = None
            self.carried_frames = 0
            return CalibrationState.unavailable(
                f"last valid calibration (solved on frame {solved_on}) expired "
                f"after {self.max_fallback_frames} carried frames"
            ), camera

        decayed = float(self.stable.confidence) * self.fallback_decay
        if decayed < HOMOGRAPHY_CONFIDENCE_MIN:
            self.n_fallback_expired += 1
            self.stable = None
            self.carried_frames = 0
            return CalibrationState.unavailable(
                f"carried calibration from frame {solved_on} decayed to "
                f"{decayed:.3f}, below HOMOGRAPHY_CONFIDENCE_MIN "
                f"{HOMOGRAPHY_CONFIDENCE_MIN}, after {carried_frames - 1} "
                "carried frames"
            ), camera

        self.carried_frames = carried_frames
        self.stable.confidence = decayed
        self.n_reused += 1
        return CalibrationState(
            H=self.stable.H,
            confidence=decayed,
            reprojection_error_m=self.stable.reprojection_error_m,
            n_points=self.stable.n_points,
            source=CalibrationSource.carried,
            valid=True,
            invalid_reason=None,
            solved_on_frame=self.stable.solved_on_frame,
            n_inliers=self.stable.n_inliers,
            inlier_ratio=self.stable.inlier_ratio,
            carried_frames=carried_frames,
            solved_confidence=self.stable.solved_confidence,
        ), camera

    def stats(self) -> dict:
        """Counters for the pipeline log / calibration history summary."""
        return {
            "attempted": self.n_attempted,
            "accepted": self.n_accepted,
            "accepted_strict": self.n_strict,
            "accepted_relaxed": self.n_relaxed,
            "reused": self.n_reused,
            "smoothed": self.n_smoothed,
            "fallback_expired": self.n_fallback_expired,
            "rejected_invalid": self.n_rejected_invalid,
            "rejected_jump": self.n_rejected_jump,
        }
