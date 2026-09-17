"""
Persistent, registry-resolved wrappers around the four per-frame detection
models (ball, field, goalpost, and the shared player checkpoint).

TWO PROBLEMS THIS SOLVES

1. INFERENCE HYGIENE. Every model here is instantiated ONCE, when the
   bundle is built, and reused for every frame. `YOLO(path)` inside a
   per-frame loop re-reads the checkpoint from disk and rebuilds the graph
   on every frame; ai/computer_vision/pipeline.py's loop is the shape that
   must not be repeated (it loads once but hardcodes `yolov8n.pt`, a stock
   COCO checkpoint). Paths come from configs/registry.py -- nothing here
   accepts a hardcoded .pt path.

2. GRACEFUL DEGRADATION. Each detector answers one question, and "I could
   not answer" is a legal answer. A missing ball returns None; a missing
   pitch returns None; missing goalposts return []. No detector raises
   into run_pipeline() for a per-frame failure -- an empty frame, a blurred
   frame, an occluded ball and a CUDA hiccup are all normal operating
   conditions over a full match, not job-ending errors.

   The ONE thing that does raise is a missing checkpoint, at construction
   time, via registry.require_checkpoint(). That is deliberate and is the
   opposite failure mode from the old placeholder: a stock COCO checkpoint
   silently produced confident, meaningless detections. Failing loudly at
   startup with the trainer command in the message is strictly better than
   degrading to nonsense at frame 5,000.

WHAT IS NOT HERE
    Player detection + tracking stays in player_tracking/tracker.py
    (ultralytics' built-in ByteTrack, one pass over the video). This module
    does not introduce a second tracker; PlayerModel below only exposes the
    resolved checkpoint path that track_video() should be handed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ai.computer_vision.frame_data import (
    BallObservation,
    BallSource,
    FieldRegion,
    GoalpostDetection,
)

logger = logging.getLogger(__name__)


class _BaseDetector:
    """Shared construction: resolve the spec, load the weights once."""

    model_name: str = ""

    def __init__(self, model: Any | None = None, **overrides: Any) -> None:
        from configs import registry

        self.spec = registry.get_model(self.model_name)
        self.inference = {**self.spec.inference, **overrides}
        if model is None:
            from ultralytics import YOLO

            model = YOLO(str(self.spec.require_checkpoint()))
        self.model = model
        self.n_calls = 0
        self.n_failures = 0

    def _predict(self, frame: np.ndarray):
        """One inference call. Returns None instead of raising -- see the
        module docstring's degradation rule."""
        self.n_calls += 1
        try:
            kwargs = dict(
                conf=float(self.inference.get("conf", 0.25)),
                iou=float(self.inference.get("iou", 0.5)),
                imgsz=int(self.inference.get("imgsz", 640)),
                max_det=int(self.inference.get("max_det", 300)),
                verbose=False,
            )
            # models.yaml carries no inference `device`, so this is None
            # unless the caller resolved one (see DetectorBundle(device=...)).
            # Omitted entirely when None -> ultralytics auto-selects, exactly
            # as before.
            device = self.inference.get("device")
            if device is not None:
                kwargs["device"] = device
            return self.model.predict(frame, **kwargs)[0]
        except Exception as exc:  # noqa: BLE001 -- deliberate, see docstring
            self.n_failures += 1
            logger.warning("%s detector failed on a frame: %s", self.model_name, exc)
            return None

    def stats(self) -> dict:
        return {"calls": self.n_calls, "failures": self.n_failures}


class BallDetector(_BaseDetector):
    """Small-object ball detector.

    Returns the single highest-confidence ball, or None. models.yaml sets
    a deliberately LOW conf (0.10) for this model: a missed ball frame
    costs more than a false positive, because interpolate_ball_gaps() can
    bridge a short miss but nothing can invent a detection that was never
    made. Returning None rather than a zero-position placeholder is what
    lets that interpolation stay honest about which points are inferred.
    """

    model_name = "ball"

    def detect(self, frame: np.ndarray) -> BallObservation | None:
        result = self._predict(frame)
        if result is None or result.boxes is None or len(result.boxes) == 0:
            return None
        boxes = result.boxes
        confs = boxes.conf.cpu().numpy()
        best = int(np.argmax(confs))
        x1, y1, x2, y2 = boxes.xyxy[best].tolist()
        # Ball centre, not foot point: a ball is a sphere whose centroid is
        # its position. (Players use bottom-centre because the bbox centre
        # sits at chest height, off the z=0 pitch plane -- see
        # TrackedDetection.foot_point().) A ball in flight is genuinely off
        # that plane too, which no single anchor fixes; the centre is the
        # standard convention and the error is symmetric rather than biased.
        return BallObservation(
            pixel_x=(x1 + x2) / 2.0,
            pixel_y=(y1 + y2) / 2.0,
            confidence=float(confs[best]),
            source=BallSource.detected,
        )


class FieldDetector(_BaseDetector):
    """Pitch-region segmentation: *where is the pitch in this image*.

    Kept separate from calibration on purpose (see
    docs/pipeline_architecture.md). Its polygon is used to cross-check
    calibration keypoints geometrically -- a homography can reproject
    consistently among its own points and still be matched to the wrong
    part of the image, which only a second, independent view of "where the
    pitch is" can catch.
    """

    model_name = "field"

    def detect(self, frame: np.ndarray) -> FieldRegion | None:
        result = self._predict(frame)
        if result is None or result.masks is None or len(result.masks) == 0:
            return None
        boxes = result.boxes
        confs = boxes.conf.cpu().numpy() if boxes is not None and len(boxes) else np.array([1.0])
        best = int(np.argmax(confs))

        try:
            xy = result.masks.xy[best]
        except (IndexError, AttributeError):
            return None
        if xy is None or len(xy) < 3:
            # Fewer than 3 vertices is not a polygon; a "region" that
            # cannot contain a point is worse than no region at all,
            # because FieldRegion.contains() would reject every keypoint
            # and mark good calibrations invalid.
            return None

        h, w = frame.shape[:2]
        polygon = [(float(px), float(py)) for px, py in xy]
        area = _polygon_area(polygon)
        return FieldRegion(
            polygon=polygon,
            confidence=float(confs[best]),
            area_fraction=area / float(w * h) if w and h else None,
        )


class GoalpostDetector(_BaseDetector):
    """Goal-mouth geometry. Returns [] when no goal is in shot, which is
    the normal case for most of a match -- an empty list is a measurement,
    not a failure."""

    model_name = "goalpost"

    def detect(self, frame: np.ndarray) -> list[GoalpostDetection]:
        result = self._predict(frame)
        if result is None or result.boxes is None or len(result.boxes) == 0:
            return []
        out: list[GoalpostDetection] = []
        boxes = result.boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            out.append(GoalpostDetection(
                x=float(x1), y=float(y1),
                width=float(x2 - x1), height=float(y2 - y1),
                confidence=float(boxes.conf[i]),
            ))
        return out


def _polygon_area(points: list[tuple[float, float]]) -> float:
    """Shoelace area in pixels^2."""
    n = len(points)
    total = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


class DetectorBundle:
    """
    The four per-frame models the pipeline needs, constructed once.

    Each detector is INDEPENDENTLY optional. `strict=False` (the default
    used by run_pipeline) means a detector whose checkpoint is missing is
    logged and set to None, and the pipeline runs without that signal --
    losing goalpost geometry should not stop player tracking from
    producing metrics. Player detection is the exception the caller
    enforces: without it there is nothing to track at all.

    Set `strict=True` in tooling that genuinely requires all five, so the
    missing checkpoint surfaces as an error rather than as quietly reduced
    output.
    """

    def __init__(self, strict: bool = False, enable: set[str] | None = None,
                 device: str | None = None) -> None:
        self.errors: dict[str, str] = {}
        wanted = enable if enable is not None else {"ball", "field", "goalpost"}
        # One resolved device for all three detectors, so they cannot each
        # silently pick a different one. None = ultralytics auto-selects
        # (unchanged historical behaviour); run_pipeline() passes a probed
        # device -- see backend/pipeline/device.py.
        self.device = device

        self.ball = self._build(BallDetector, "ball", strict, wanted)
        self.field = self._build(FieldDetector, "field", strict, wanted)
        self.goalpost = self._build(GoalpostDetector, "goalpost", strict, wanted)

    def _build(self, cls, name: str, strict: bool, wanted: set[str]):
        if name not in wanted:
            return None
        try:
            return cls(**({"device": self.device} if self.device is not None else {}))
        except Exception as exc:  # noqa: BLE001
            if strict:
                raise
            self.errors[name] = str(exc)
            logger.warning(
                "%s detector unavailable -- continuing without it. %s", name, exc
            )
            return None

    def detect_all(self, frame: np.ndarray) -> dict:
        """Runs every available detector on one frame.

        Returns a dict with keys ball/field/goalposts whose values are the
        honest "absent" sentinel (None / None / []) for any detector that
        is unavailable or found nothing. The caller cannot tell the two
        apart from this dict alone, by design -- both mean "no measurement
        this frame", and `self.errors` records the structural reason once
        rather than per frame.
        """
        return {
            "ball": self.ball.detect(frame) if self.ball else None,
            "field": self.field.detect(frame) if self.field else None,
            "goalposts": self.goalpost.detect(frame) if self.goalpost else [],
        }

    def stats(self) -> dict:
        return {
            name: det.stats()
            for name, det in (("ball", self.ball), ("field", self.field),
                              ("goalpost", self.goalpost))
            if det is not None
        }

    def available(self) -> list[str]:
        return [n for n, d in (("ball", self.ball), ("field", self.field),
                               ("goalpost", self.goalpost)) if d is not None]
