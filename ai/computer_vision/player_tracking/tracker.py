"""
Player & ball tracking layer: assigns stable identities to per-frame YOLO
detections using ByteTrack.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterator


TRACKABLE_CLASS_NAMES = {"player", "goalkeeper"}
BALL_CLASS_NAME = "ball"


@dataclass
class TrackedDetection:
    """One tracked box in one frame. Field names match PlayerDetection
    (docs/database_schema.md) so this can be inserted with no remapping."""

    frame_number: int
    player_id: int
    class_name: str
    team_id: str | None
    team_assignment_confidence: float | None
    x: float
    y: float
    width: float
    height: float
    confidence: float

    def foot_point(self) -> tuple[float, float]:
        """
        Pixel position to feed into homography.pixel_to_pitch().
        """
        return (self.x + self.width / 2.0, self.y + self.height)

    def to_dict(self) -> dict:
        return asdict(self)


def _xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)


def track_video(
    model_path: str,
    video_path: str,
    tracker_config: str = "bytetrack.yaml",
    conf: float = 0.25,
    classes: list[str] | None = None,
    device: str | None = None,
) -> Iterator[list[TrackedDetection]]:
    """
    Run YOLO detection + ByteTrack tracking together, frame by frame.
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    class_name_by_id = model.names

    class_filter_ids = None
    if classes is not None:
        wanted = set(classes)
        class_filter_ids = [i for i, name in class_name_by_id.items() if name in wanted]

    track_kwargs = dict(
        source=video_path,
        tracker=tracker_config,
        conf=conf,
        classes=class_filter_ids,
        persist=True,
        stream=True,
        verbose=False,
    )
    if device is not None:
        track_kwargs["device"] = device

    results = model.track(**track_kwargs)

    for frame_number, result in enumerate(results):
        frame_detections: list[TrackedDetection] = []

        if result.boxes is None or result.boxes.id is None:
            yield frame_detections
            continue

        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            class_name = class_name_by_id[cls_id]
            track_id = int(boxes.id[i])
            conf_score = float(boxes.conf[i])
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            x, y, w, h = _xyxy_to_xywh(x1, y1, x2, y2)

            frame_detections.append(
                TrackedDetection(
                    frame_number=frame_number,
                    player_id=track_id,
                    class_name=class_name,
                    team_id=None,
                    team_assignment_confidence=None,
                    x=x, y=y, width=w, height=h,
                    confidence=conf_score,
                )
            )

        yield frame_detections


def track_detections(
    frames: list[dict],
    classes: list[str] | None = None,
) -> list[list[TrackedDetection]]:
    """
    Re-track an already-computed per-frame detection list using our own
    ByteTrack implementation (bytetrack.py -- adapted from the uploaded
    reference implementation, with two bugfixes: a missing frame_id
    assignment in STrack.activate() that crashed on any track seen for
    """
    import os
    import sys

    import numpy as np

    _PLAYER_TRACKING_DIR = os.path.dirname(os.path.abspath(__file__))
    if _PLAYER_TRACKING_DIR not in sys.path:
        sys.path.insert(0, _PLAYER_TRACKING_DIR)

    from bytetrack import BYTETracker

    class_name_order = TRACKABLE_CLASS_NAMES | {BALL_CLASS_NAME, "referee"}
    class_name_order = sorted(class_name_order)
    name_to_id = {name: i for i, name in enumerate(class_name_order)}
    id_to_name = {i: name for name, i in name_to_id.items()}

    byte_tracker = BYTETracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
    byte_tracker.reset()

    output: list[list[TrackedDetection]] = []

    for frame_entry in frames:
        frame_number = frame_entry["frame"]
        raw_detections = frame_entry["detections"]

        if classes is not None:
            raw_detections = [d for d in raw_detections if d["class"] in classes]

        if raw_detections:
            output_results = np.array(
                [
                    [*d["bbox"], d["confidence"], name_to_id.get(d["class"], -1)]
                    for d in raw_detections
                    if d["class"] in name_to_id
                ],
                dtype=np.float64,
            )
        else:
            output_results = np.empty((0, 6), dtype=np.float64)

        tracked_stracks = byte_tracker.update(output_results)

        frame_detections: list[TrackedDetection] = []
        for strack in tracked_stracks:
            x1, y1, x2, y2 = strack.tlbr
            x, y, w, h = _xyxy_to_xywh(x1, y1, x2, y2)
            frame_detections.append(
                TrackedDetection(
                    frame_number=frame_number,
                    player_id=strack.track_id,
                    class_name=id_to_name.get(strack.class_id, "unknown"),
                    team_id=None,
                    team_assignment_confidence=None,
                    x=float(x), y=float(y), width=float(w), height=float(h),
                    confidence=float(strack.score),
                )
            )
        output.append(frame_detections)

    return output
