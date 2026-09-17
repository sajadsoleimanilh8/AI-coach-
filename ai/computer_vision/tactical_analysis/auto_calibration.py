"""
Automatic pitch calibration from the trained 32-keypoint pose model.

WHAT THIS REPLACES
    manual_calibration.py -- an operator clicking 17 named landmarks
    (CALIBRATION_POINT_ORDER) once per match -- was the ONLY way to obtain
    a homography. This module derives the same correspondences from the
    `calibration` pose model instead, so a video can be processed with no
    operator in the loop.

    manual_calibration.py is deliberately NOT deleted. It stays as the
    explicit override for footage this model handles badly, and as the QA
    reference. Both paths converge on the SAME two types --
    homography.HomographyResult and frame_data.CalibrationState -- so
    nothing downstream needs to know which one ran; only
    `CalibrationState.source` records it.

WHAT THE MATH IS NOT
    compute_homography() / homography_confidence() / pixel_to_pitch() are
    untouched and not reimplemented here. This module's only job is
    producing (pixel, pitch-metre) correspondences and deciding, over
    time, when to recompute them.

MEASURED BEHAVIOUR OF THE MODEL (2026-08-13, do not delete this note)
    The index -> pitch-metre table this module feeds (pitch_keypoints.py)
    is verified correct: fitting GROUND-TRUTH dataset keypoints through it
    gives a median reprojection error of 0.295 m across 255 training
    images, 99.6% under 1.0 m (`python -m scripts.validate_pitch_keypoints`).

    The MODEL, however, does not generalise off its training domain. On
    the in-domain test split it fires on 12/12 images with box confidence
    ~0.95. On the broadcast clips in storage/uploads it fires on roughly
    1 sampled frame in 6 at 1280x720 and 0 in 6 at 854x480, and where it
    does fire the resulting homography reprojects at 7-25 m -- far beyond
    anything usable.

    That is WHY this module is built the way it is: every result is gated
    on HOMOGRAPHY_CONFIDENCE_MIN and cross-checked against the field
    model's pitch region, a failed or rejected frame degrades to
    CalibrationState.unavailable() instead of raising, and the manual path
    remains available. An auto-calibration that silently returned a 20 m
    homography would corrupt every pitch-space metric downstream while
    looking like it worked.

WHY THE MODEL LOOKED WORSE THAN IT IS (2026-08-16, the note above amended)
    Most of that measured collapse was a TRAIN/INFERENCE PREPROCESSING
    MISMATCH, not the model.

    Every image in the calibration dataset is 960x960 -- a 16:9 broadcast
    frame resized to a square WITHOUT preserving aspect ratio (verify by
    eye: the centre circle in a training image is nearly round, where a
    real broadcast wide shot shows a markedly flattened ellipse). The model
    therefore only ever learned the vertically-stretched pitch.

    Inference did the opposite. Passing a 1280x720 frame to ultralytics at
    imgsz=960 LETTERBOXES it to 960x544 with padding, preserving aspect --
    so every broadcast frame arrived in a geometry the model had never seen
    once during training.

    Measured over 3300 frames of test.mp4, holding everything else fixed:
        letterboxed (what this module used to do)
            box detected above conf=0.30 on   200/3300 frames (6.1%)
            valid calibrations                  0/3300 frames (0.0%)
        stretched to 960x960 (what training did)
            box detected above conf=0.30 on  2123/3300 frames (64.3%)

    So `preprocess: stretch_square` in configs/models.yaml is not a tuning
    knob and not a workaround -- it is inference finally matching training.
    Keypoints are detected in the square image and mapped back to original
    frame pixels before any homography is fitted, so every coordinate this
    module hands downstream stays in native frame pixels exactly as before.

    The generalisation gap is REAL but smaller than it looked, and it has
    moved: detection is now the easy part, and what remains poor off-domain
    is keypoint IDENTITY -- see the next note.

WHAT REMAINS BROKEN (2026-08-16 -- READ THIS BEFORE TRUSTING A HOMOGRAPHY)
    Fixing preprocessing did NOT make this module produce correct pitch
    coordinates on out-of-domain broadcast footage. It made the model
    detect the pitch; the homographies it then supports are still wrong by
    metres, and the gates in this file CANNOT tell which ones.

    The evidence, and it is direct rather than inferred: frames the full
    gate chain accepted were rendered with the pitch outline, halfway line,
    penalty areas, goal areas and centre circle projected back through the
    accepted H and drawn over the image. On every sampled frame the
    projected lines miss the painted lines by metres -- including a frame
    with 8/8 RANSAC inliers and 0.31 m reprojection error, the best score
    this system can produce.

    WHY NO GATE HERE CAN CATCH THAT. Reprojection error, confidence, inlier
    count and inlier ratio are all SELF-REFERENTIAL: they ask whether the
    model's own keypoints agree with each other. When the model assigns the
    wrong pitch IDENTITY to a landmark -- calling a goal-area corner a
    penalty-area corner, say -- and does so consistently, those keypoints
    agree with each other perfectly while describing a pitch that is not
    the one in the picture. The residual is small BECAUSE the error is in
    the labels rather than the positions.

    Independent signals were tried and do not separate the cases either:
    the field-segmentation polygon saturates (projected-pitch vs detected-
    pitch IoU is ~0.97 on zoomed shots for both good and bad fits), and
    tightening the consensus gates reduces how many wrong fits get through
    without making the survivors right.

    CONSEQUENCE. The gates in this module are honest about what they can
    see, and they are not sufficient. Do not treat a valid=True from the
    automatic path on out-of-domain footage as a verified calibration.
    manual_calibration.py remains the trustworthy path, and the real fix is
    retraining the keypoint model on aspect-correct, domain-representative
    footage -- see the final report accompanying this change.

REPRODUCE ALL OF THE ABOVE
    python -m scripts.evaluate_calibration_video test.mp4 --frames 3300 \
        --compare-preprocess          # detection: letterbox vs stretch
    python -m scripts.evaluate_calibration_video test.mp4 --frames 3300 \
        --pipeline                    # end-to-end, incl. reuse breakdown
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
    """Outcome of ONE attempt to calibrate a single frame.

    `homography` is None whenever the attempt did not produce a matrix at
    all -- no detection, too few visible keypoints, or a degenerate fit.
    `reason` always explains a None, so a caller can log why calibration
    is unavailable instead of reporting a bare failure."""

    homography: HomographyResult | None
    keypoints_px: list[tuple[float, float]] = field(default_factory=list)
    keypoint_indices: list[int] = field(default_factory=list)
    detection_confidence: float | None = None
    n_keypoints_visible: int = 0
    reason: str | None = None
    #: Which rung of the fallback chain produced this -- "strict",
    #: "relaxed", or None when nothing was produced. Recorded so a run can
    #: report how much of its calibration came from the relaxed rung rather
    #: than leaving that indistinguishable from a clean strict solve.
    strategy: str | None = None

    @property
    def ok(self) -> bool:
        return self.homography is not None


# ----------------------------------------------------------------------
# Inference preprocessing
# ----------------------------------------------------------------------

def preprocess_frame(
    frame: np.ndarray, imgsz: int, mode: str
) -> tuple[np.ndarray, float, float]:
    """
    Prepares a frame for the pose model and returns (image, sx, sy), where
    multiplying a detected keypoint by (sx, sy) maps it back to ORIGINAL
    frame pixels.

    `mode` is read from configs/models.yaml (`inference.preprocess`):

      "stretch_square"  resize to imgsz x imgsz, ignoring aspect ratio.
                        This reproduces how the calibration dataset was
                        built -- see this module's docstring, which has the
                        3300-frame measurement showing the letterboxed path
                        detected a pitch on 6.1% of broadcast frames and
                        this one on 64.3%.

      "native"          hand the frame over untouched and let ultralytics
                        letterbox it. The previous behaviour, kept because
                        it is correct for any model actually trained on
                        aspect-preserved images, and because a future
                        retrained checkpoint should switch back to it by
                        changing one line of YAML rather than code.

    Any unrecognised mode falls back to "native" with a warning rather than
    raising: a typo in config must not take down a pipeline run, but it must
    also not silently look like the intended setting.
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


# ----------------------------------------------------------------------
# Keypoint extraction
# ----------------------------------------------------------------------

def extract_keypoints(
    result: Any, min_confidence: float,
    scale: tuple[float, float] = (1.0, 1.0),
) -> tuple[dict[int, tuple[float, float]], dict[int, float], float | None]:
    """
    Pulls {index: (px, py)} and {index: confidence} out of one ultralytics
    pose Result.

    `scale` is the (sx, sy) returned by preprocess_frame(), applied here so
    that every coordinate leaving this function is in ORIGINAL frame pixels
    regardless of what the model was fed. Defaults to (1, 1), the identity,
    so existing callers that pass a Result straight through are unaffected.

    Only the highest-confidence instance is used: models.yaml sets
    max_det=1 for this model because a frame contains exactly one pitch,
    but that is an inference setting a caller could override, so the
    choice is made explicitly here rather than assumed.

    Keypoints below `min_confidence` are dropped, not defaulted -- an
    unreliable landmark biases the fit while looking like extra evidence
    (see keypoints_to_correspondences()'s own note).
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
        # A pose row is (x, y, conf); some exports omit conf entirely, in
        # which case every detected landmark is treated as confident --
        # the downstream confidence gate still applies to the FIT.
        if len(row) >= 3:
            x, y, c = float(row[0]), float(row[1]), float(row[2])
        else:
            x, y, c = float(row[0]), float(row[1]), 1.0
        confidences[idx] = c
        if c < min_confidence:
            continue
        # (0, 0) is what ultralytics emits for an unlabelled/absent
        # landmark. A genuine corner flag never lands exactly on the
        # origin, so this is safe to treat as "not present".
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

    RANSAC is used as soon as there are more than the minimal 4 points: a
    single mislocalised landmark otherwise drags the entire fit, and the
    reported reprojection error would then describe a homography that is
    wrong everywhere rather than flagging the one bad point.

    The RANSAC inlier threshold comes from
    constants.HOMOGRAPHY_RANSAC_THRESHOLD_M. Note the unit -- cv2 measures that
    threshold in the DESTINATION space, which for this homography is pitch
    METRES, not pixels; 3 m is loose enough to admit badly mislocalised
    landmarks into the consensus set, which is the population RANSAC exists to
    exclude.
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
        # Collinear or duplicated points -- a real, expected outcome when
        # the model finds only landmarks along a single pitch line.
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

    This is the single place `valid` is decided for the auto path; the
    manual path goes through calibration_state_from_manual() below and
    lands on exactly the same evaluate() call.

    `frame_size` is (width, height); passing it enables the point-spread
    gate (HOMOGRAPHY_MIN_POINT_SPREAD), which is skipped when it is None
    because hull area needs a frame area to be a fraction of. calibrate()
    below passes it. NOT YET THREADED: backend/api/calibration_debug.py's
    call, which is a debug-overlay route left untouched here -- the spread
    gate therefore does not apply on that path.
    """
    if not result.ok:
        return CalibrationState.unavailable(result.reason or "auto-calibration failed")

    h = result.homography

    # The spread and geometry gates below must see the points that ACTUALLY
    # DETERMINED the matrix, which is the RANSAC consensus set -- not every
    # point that was offered. Measuring spread over the rejected outliers too
    # overstates coverage: a consensus bunched in one corner still scores
    # well-spread as long as some discarded outlier sat on the far side of
    # the frame, which is precisely the case the spread gate exists to catch.
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

    manual_calibration.load_calibration() returns (H, record); this turns
    that into the identical CalibrationState the auto path produces, so
    downstream code reads one shape and one `valid` flag regardless of
    which path ran (the requirement that motivated this module).
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


# ----------------------------------------------------------------------
# Camera motion
# ----------------------------------------------------------------------

class CameraMotionDetector:
    """
    Frame-to-frame camera shift, by sparse optical flow.

    Deliberately measures the shift of BACKGROUND features rather than
    trying to model the camera: the only question this needs to answer is
    "did the view change enough that the existing homography is stale",
    and a median feature displacement answers that directly.

    The median (not the mean) is what makes this work on football footage
    -- players moving across a static frame produce a minority of large
    displacements, and a mean would read those as camera motion. The
    median tracks the dominant, background motion instead.
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
            # First frame (or a resolution change): nothing to compare
            # against. `unknown` -- not `static` -- so a caller does not
            # mistake "not measured" for "measured and stable".
            return CameraState(motion=CameraMotion.unknown, recalibration_advised=True)

        pts = cv2.goodFeaturesToTrack(prev, maxCorners=self.max_corners,
                                      qualityLevel=0.01, minDistance=8)
        if pts is None or len(pts) < 8:
            # A featureless frame (fog, heavy blur, a plain graphic) --
            # honestly unknown rather than assumed static.
            return CameraState(motion=CameraMotion.unknown, recalibration_advised=False)

        nxt, status, _err = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None)
        if nxt is None or status is None:
            return CameraState(motion=CameraMotion.unknown)

        ok = status.reshape(-1).astype(bool)
        if ok.sum() < 8:
            # Most features lost between frames -- that itself is the
            # signature of a hard cut, not of a measurement failure.
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


# ----------------------------------------------------------------------
# Homography jump rejection
# ----------------------------------------------------------------------

def homography_jump_m(
    H_old: np.ndarray, H_new: np.ndarray, frame_size: tuple[int, int]
) -> float:
    """
    How far, in PITCH METRES, the two homographies disagree about the same
    image.

    Compares where four probe points (the image quarter-points) land under
    each matrix and returns the median displacement. This is the
    meaningful comparison; differencing the matrices entry-by-entry is not,
    because a homography is only defined up to scale and its entries have
    no common unit.

    Returns inf when either matrix is singular or either projection is
    non-finite, so an unusable new matrix is rejected outright rather than
    scoring some arbitrary finite distance. The singularity check is
    explicit because cv2.perspectiveTransform does NOT raise on a
    degenerate matrix -- an all-zero H returns finite coordinates, which
    happened to exceed the threshold in testing but only by luck, not by
    construction.
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
        # A homography must be invertible to describe a projective map at
        # all. `abs(det) < 1e-12` is singular-to-numerical-precision.
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

    WHY NOT (1-w)*H_old + w*H_new
        Because that is not a meaningful operation. A homography is defined
        only up to scale, so the two matrices may be scaled arbitrarily
        relative to each other and an entry-wise average of them describes
        neither camera -- it can be near-singular, or fold the image across
        its horizon, while looking like a perfectly ordinary 3x3.

    WHAT THIS DOES INSTEAD
        Interpolates where the two matrices SEND THE SAME FOUR IMAGE POINTS,
        which is the thing that has physical meaning and a unit (pitch
        metres), then refits a homography through the four blended
        correspondences. The result is a genuine projective map by
        construction, because it was fitted as one from four point pairs
        rather than assembled numerically.

    `weight_new` is CALIBRATION_SMOOTHING: 1.0 takes the new fit unchanged,
    0.0 keeps the old one. Values outside [0, 1] are clamped rather than
    extrapolated -- extrapolating past either endpoint invents a camera pose
    that neither fit measured.
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


# ----------------------------------------------------------------------
# The calibrator
# ----------------------------------------------------------------------

class AutoCalibrator:
    """
    Persistent pose-model wrapper + temporal-stability state machine.

    The model is loaded ONCE, in __init__, and reused for every frame --
    instantiating YOLO() inside a per-frame loop reloads weights from disk
    on every frame and was the specific inference-hygiene defect this
    class exists to avoid.

    Per-frame policy (calibrate()):
      static camera   -> reuse the last stable calibration, do not re-run
                         the model at all
      panning / cut   -> re-run and accept the result only if it is valid
                         AND does not jump implausibly from the last
                         stable one
      never solved    -> attempt on every frame until one succeeds

    A rejected recalculation does NOT discard the previous stable
    calibration; it carries it forward (source=carried). Throwing away a
    good calibration because one frame produced a bad one is strictly
    worse than keeping it.

    THE FALLBACK CHAIN, in order, deterministic and explainable:
      A. strict   -- keypoints at the configured visibility floor
                     (kpt_conf_min), fitted with RANSAC
      B. relaxed  -- retried at kpt_conf_relaxed, ONLY when A produced too
                     few correspondences to fit at all. Not tried when A
                     produced a fit that the gates rejected: adding weaker
                     keypoints to a set that already failed on geometry
                     makes the fit worse, not better, and retrying until
                     something passes is how a validity gate gets defeated.
                     A result from this rung is labelled strategy="relaxed"
                     all the way into the run stats, so the report can say
                     how much of the output leaned on it.
      C. carried  -- reuse the last stable calibration, with confidence
                     decayed per frame and a hard expiry at
                     CALIBRATION_MAX_FALLBACK_FRAMES
      D. otherwise CalibrationState.unavailable() -- valid=False with the
                     real reason attached

    WHAT C DOES NOT DO. It never resurrects an expired calibration, never
    raises a decayed confidence back up, and never marks a carried frame
    valid once either the frame budget or the confidence gate has run out.
    A stale homography served as fresh would put players at pitch
    coordinates nothing measured, which is exactly the class of silent
    corruption this module exists to prevent.
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
        # How the frame is shaped before inference -- see preprocess_frame().
        # Config-driven so that a retrained, aspect-preserving checkpoint
        # switches back by editing configs/models.yaml, not this file.
        self.preprocess = (preprocess if preprocess is not None
                           else str(inference.get("preprocess", "native")))
        # Rung B of the fallback chain. Defaults below kpt_conf_min; a value
        # >= kpt_conf_min makes rung B a no-op, which is a legitimate way to
        # disable it from config.
        self.kpt_conf_relaxed = (
            kpt_conf_relaxed if kpt_conf_relaxed is not None
            else float(inference.get("kpt_conf_relaxed", 0.35)))
        self.max_jump_m = max_jump_m
        self.max_fallback_frames = max_fallback_frames
        self.fallback_decay = fallback_decay
        self.smoothing = smoothing
        # Device plumbing only -- no effect on keypoint extraction, the
        # homography fit, or any gate below. None keeps ultralytics'
        # auto-selection (unchanged behaviour); run_pipeline() passes a
        # probed device (see backend/pipeline/device.py).
        self.device = device

        if model is None:
            from ultralytics import YOLO
            # require_checkpoint() raises a specific, actionable error
            # naming the trainer -- it never falls back to stock weights.
            model = YOLO(str(spec.require_checkpoint()))
        self.model = model

        self.motion = motion_detector or CameraMotionDetector()

        # Temporal state
        self.stable: CalibrationState | None = None
        self.last_camera: CameraState = CameraState()
        # Consecutive frames the current stable calibration has been carried
        # without being re-solved. Reset to 0 on every accepted solve.
        self.carried_frames = 0
        # Counters, surfaced in the pipeline log so "temporal stability is
        # working" is an observation rather than an assertion.
        self.n_attempted = 0
        self.n_accepted = 0
        self.n_reused = 0
        self.n_rejected_jump = 0
        self.n_rejected_invalid = 0
        # How the accepted solves were obtained, and how often the fallback
        # was leaned on. Reported so a run can state plainly whether its
        # valid frames came from detection or from reuse.
        self.n_strict = 0
        self.n_relaxed = 0
        self.n_fallback_expired = 0
        self.n_smoothed = 0

    # -- single-frame, no temporal logic ------------------------------
    def calibrate_frame(
        self, frame: np.ndarray, field_region: FieldRegion | None = None,
        frame_id: int | None = None,
    ) -> AutoCalibrationResult:
        """
        Runs the model on one frame and fits a homography. No reuse, no
        jump rejection -- the plain "calibrate this image" entry point,
        used directly by tests and by the QA overlay.

        Rungs A and B of the fallback chain live here (strict keypoints,
        then relaxed keypoints when strict produced too few to fit at all).
        Rungs C and D are temporal and live in calibrate().

        ONE inference serves both rungs. The model is run once and the two
        visibility floors are applied to the same keypoint set afterwards --
        re-running the model at a lower box threshold would return the same
        single instance anyway (max_det=1) at roughly 85 ms a frame.

        Never raises on inference failure: a corrupt frame or a CUDA
        hiccup returns a `reason`-carrying failure, because one bad frame
        must not end a pipeline run (see runner.py's degradation rules).
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

        # Extract once at the RELAXED floor, which is a superset; the strict
        # rung then filters the same dict rather than paying for a second
        # pass over the model output.
        floor = min(self.kpt_conf_min, self.kpt_conf_relaxed)
        keypoints, confidences, det_conf = extract_keypoints(
            result, floor, scale=(sx, sy))
        if not keypoints:
            return AutoCalibrationResult(
                homography=None, detection_confidence=det_conf,
                reason="no pitch keypoints above the visibility floor",
            )

        # -- rung A: strict ------------------------------------------------
        strict = calibrate_from_keypoints(
            keypoints, confidences=confidences,
            min_confidence=self.kpt_conf_min, detection_confidence=det_conf,
            strategy="strict",
        )
        if strict.ok or self.kpt_conf_relaxed >= self.kpt_conf_min:
            return strict

        # -- rung B: relaxed ----------------------------------------------
        # Reached ONLY because strict could not form a fit at all. If strict
        # produced a homography that the validity gates then rejected, that
        # rejection stands; see this class's docstring for why retrying with
        # weaker evidence in that case would be defeating the gate rather
        # than recovering from a shortage of correspondences.
        relaxed = calibrate_from_keypoints(
            keypoints, confidences=confidences,
            min_confidence=self.kpt_conf_relaxed, detection_confidence=det_conf,
            strategy="relaxed",
        )
        if not relaxed.ok:
            # Report the STRICT failure -- it is the one that describes the
            # configured behaviour; the relaxed retry finding nothing either
            # is not new information.
            return strict
        return relaxed

    # -- temporally stabilised ----------------------------------------
    def calibrate(
        self,
        frame: np.ndarray,
        frame_id: int,
        field_region: FieldRegion | None = None,
    ) -> tuple[CalibrationState, CameraState]:
        """
        The per-frame entry point the pipeline uses.

        Returns (calibration, camera_state). The calibration is always a
        usable object -- never None -- with `valid` False and a populated
        `invalid_reason` whenever this frame has no trustworthy geometry.
        """
        camera = self.motion.update(frame)
        self.last_camera = camera
        h_px, w_px = frame.shape[:2]

        # Static camera with an existing good calibration: reuse it. This
        # is the case that makes per-frame calibration affordable at all.
        # Still subject to the fallback budget -- a static camera is a
        # reason not to RE-SOLVE, not a licence to serve one solve forever.
        # When the budget runs out the frame falls through to a real solve
        # below rather than reporting nothing, which is the whole point of
        # expiring: go and get fresh evidence.
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
            # Rung C: keep serving the last good calibration rather than
            # dropping to nothing on a single bad frame -- but only while
            # the camera has not cut, which invalidates the old view
            # outright, and only within the fallback budget.
            if self.stable is not None and camera.motion is not CameraMotion.cut:
                carried, camera_out = self._carry(camera, frame_id)
                if carried.valid:
                    return carried, camera_out
                # Rung D: the fallback has expired. Report THIS frame's own
                # failure, not the expiry -- the model's reason is the more
                # useful of the two, and the expiry is already counted.
            return state, camera

        # Valid fit -- but is it a plausible successor to the last one?
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
                # Fallback expired and the new fit is the only evidence
                # left, so take it rather than reporting nothing.

            # Smooth toward the new fit instead of snapping to it, so that
            # frame-to-frame noise in the keypoints does not translate into
            # jitter in every pitch coordinate downstream. Blending happens
            # in pitch space and is refitted -- see blend_homographies().
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
                    # The smoothed matrix must clear the SAME gates as an
                    # unsmoothed one. If blending pushed it out of geometric
                    # plausibility, the unsmoothed fit is used unchanged
                    # rather than the smoothed one being waved through.
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

    # -- rung C: bounded, decaying reuse -------------------------------
    def _carry(
        self, camera: CameraState, frame_id: int | None,
    ) -> tuple[CalibrationState, CameraState]:
        """
        Serves the last stable calibration for one more frame, with its
        confidence decayed, or returns an INVALID state once the fallback
        has expired.

        Two independent expiries, whichever comes first:
          - the frame budget, CALIBRATION_MAX_FALLBACK_FRAMES;
          - the ordinary confidence gate, once decay has taken the carried
            confidence below HOMOGRAPHY_CONFIDENCE_MIN. A calibration that
            solved only marginally above the gate therefore expires sooner
            than a strong one, which is the intended behaviour: weaker
            evidence should not survive as long.

        The decayed confidence is what downstream consumers see, so a
        carried calibration is never reported as being as good as a fresh
        solve. Nothing here can raise a confidence or extend an expiry.
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
