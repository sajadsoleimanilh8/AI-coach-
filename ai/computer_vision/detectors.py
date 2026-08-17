"""
Persistent, registry-resolved wrappers around the four per-frame detection
models (ball, field, goalpost, and the shared player checkpoint).
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
    """Small-object ball detector."""

    model_name = "ball"

    def detect(self, frame: np.ndarray) -> BallObservation | None:
        result = self._predict(frame)
        if result is None or result.boxes is None or len(result.boxes) == 0:
            return None
        boxes = result.boxes
        confs = boxes.conf.cpu().numpy()
        best = int(np.argmax(confs))
        x1, y1, x2, y2 = boxes.xyxy[best].tolist()
        return BallObservation(
            pixel_x=(x1 + x2) / 2.0,
            pixel_y=(y1 + y2) / 2.0,
            confidence=float(confs[best]),
            source=BallSource.detected,
        )


class FieldDetector(_BaseDetector):
    """Pitch-region segmentation: *where is the pitch in this image*."""

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
    """

    def __init__(self, strict: bool = False, enable: set[str] | None = None,
                 device: str | None = None) -> None:
        self.errors: dict[str, str] = {}
        wanted = enable if enable is not None else {"ball", "field", "goalpost"}
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
        """Runs every available detector on one frame."""
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
