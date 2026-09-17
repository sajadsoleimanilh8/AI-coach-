"""
Player & ball tracking layer: assigns stable identities to per-frame YOLO
detections using ByteTrack.

Bridges `ai/computer_vision/player detection` (per-frame boxes, no
identity -- see that folder's test.py / detections.json) to:
  - docs/database_schema.md's PlayerDetection / PlayerTracking tables
    (continuous trajectories need a stable player_id per frame)
  - ai/computer_vision/tactical_analysis/homography.py (pixel -> pitch,
    needs to know *which* pixel belongs to *which* player across frames
    to compute speed/distance)

IMPORTANT (matches docs/database_schema.md's own note): the `player_id`
produced here is a ByteTrack tracking ID, valid only within one continuous
video/match. It is NOT a persistent player identity across matches -- that
requires jersey-number OCR or Re-ID, which is out of scope here (flagged as
a Phase 3+ item in tracking_system_design.md §13).

TWO TRACKERS, ONE PRODUCTION PATH -- the explicit decision
-----------------------------------------------------------------------
This package contains two ByteTrack implementations. They are NOT
alternatives to each other and must not be treated as interchangeable.
The split was reviewed in Phase 2 (2026-08-13) and deliberately kept:

  track_video()      -> ultralytics' built-in ByteTrack. THE PRODUCTION
                        PATH. backend/pipeline/runner.py calls only this.
                        One pass over the video, detection and tracking
                        fused, no intermediate JSON.

  track_detections() -> the hand-rolled bytetrack.py in this package
                        (KalmanFilter/STrack/BYTETracker). OFFLINE
                        re-tracking of detections that already exist.

WHY bytetrack.py IS KEPT RATHER THAN DELETED AS DEAD CODE
    It does something ultralytics' tracker structurally cannot:
    ultralytics' `.track()` only tracks the output of the one model it is
    running, live, over a video stream. It has no entry point that accepts
    a list of boxes from somewhere else. So any workflow that tracks
    detections NOT produced by a single live YOLO pass needs its own
    tracker. With five specialised models now in the registry (see
    configs/models.yaml), re-tracking merged or ensembled detections is a
    real workflow, not a hypothetical one -- and re-running detection just
    to re-track is wasteful when the boxes are already on disk.

    It is also not untested code. Two suites exercise it directly:
      - tests/test_bytetrack_fixes.py -- regression tests for the two
        real bugs fixed in it (the missing frame_id assignment in
        STrack.activate() that crashed on single-frame tracks, and the
        always-True is_activated flag that skipped the one-frame
        confirmation delay)
      - tests/test_tracking_pipeline.py -- end-to-end over
        track_detections()

    The rule this decision creates, which future changes must respect:
    run_pipeline() uses track_video() and nothing else. Introducing a
    second tracker into the production path is what was rejected here --
    not the existence of an offline re-tracking tool with its own tests.

Both return the same canonical per-frame shape (see `TrackedDetection`),
matching PlayerDetection field names 1:1 so the output can be written
straight to that table with no renaming/reshaping step.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass

# The CURRENT player model is single-class: {0: player} (configs/models.yaml).
# The names below are kept deliberately wider than that:
#   - "goalkeeper" so downstream heuristic labelling (see
#     docs/pipeline_architecture.md's goalkeeper/referee decision) can mark
#     a track without this set having to change;
#   - "ball"/"referee" because track_detections() re-tracks detections from
#     ARBITRARY sources, including the legacy 4-class checkpoint
#     (0: ball, 1: goalkeeper, 2: player, 3: referee) whose JSON output is
#     still on disk.
# The ball no longer arrives through this path in production -- it comes
# from the dedicated small-object ball model via
# ai/computer_vision/detectors.py::BallDetector.
TRACKABLE_CLASS_NAMES = {"player", "goalkeeper"}  # referees excluded from player tracking by default
BALL_CLASS_NAME = "ball"

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

#: Project-local ByteTrack settings. Stock "bytetrack.yaml" is tuned for
#: generic short clips; configs/trackers/ssc_bytetrack.yaml documents each
#: divergence and why broadcast football needs it. Callers can still pass
#: any ultralytics tracker name or path via track_video(tracker_config=...).
DEFAULT_TRACKER_CONFIG = os.path.join(_REPO_ROOT, "configs", "trackers", "ssc_bytetrack.yaml")


def resolve_tracker_config(tracker_config: str) -> str:
    """
    Accepts a repo-relative name ("ssc_botsort.yaml"), an absolute path, or a
    stock ultralytics name ("bytetrack.yaml") and returns what
    model.track(tracker=...) should be given.

    A stock name is passed through untouched -- ultralytics resolves those
    against its own cfg/trackers directory.
    """
    if os.path.isabs(tracker_config) and os.path.exists(tracker_config):
        return tracker_config
    local = os.path.join(_REPO_ROOT, "configs", "trackers", os.path.basename(tracker_config))
    if os.path.exists(local):
        return local
    return tracker_config


@dataclass
class TrackedDetection:
    """One tracked box in one frame. Field names match PlayerDetection
    (docs/database_schema.md) so this can be inserted with no remapping."""

    frame_number: int
    player_id: int          # ByteTrack ID -- stable within this video only
    class_name: str         # "player" | "goalkeeper" | "ball" | "referee"
    team_id: str | None     # always None here -- team assignment (jersey
                             # clustering) is a separate, not-yet-implemented
                             # module (see recent_updates: hardcoded "neutral")
    team_assignment_confidence: float | None
    x: float                # pixel-space, top-left corner
    y: float
    width: float
    height: float
    confidence: float       # YOLO detection confidence, 0-1

    def foot_point(self) -> tuple[float, float]:
        """
        Pixel position to feed into homography.pixel_to_pitch().

        Use bottom-center of the box, not the box center: the homography
        maps points that lie ON the pitch plane (z=0), and a standing
        player's *feet* are on that plane -- the bbox center is roughly at
        chest height and will project to the wrong pitch location,
        especially for players near the edges of the frame where the
        camera angle is shallow. This is a common, easy-to-miss source of
        several-meter position error that reprojection-error checks alone
        won't catch (see tactical_analysis/README.md's calibration notes).
        """
        return (self.x + self.width / 2.0, self.y + self.height)

    def to_dict(self) -> dict:
        return asdict(self)


def _xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)


def track_video(
    model_path: str,
    video_path: str,
    tracker_config: str = DEFAULT_TRACKER_CONFIG,
    conf: float = 0.25,
    classes: list[str] | None = None,
    device: str | None = None,
) -> Iterator[list[TrackedDetection]]:
    """
    Run YOLO detection + ByteTrack tracking together, frame by frame.

    Args:
        model_path: path to a trained .pt checkpoint (e.g. from
            phase0_1_pipeline.py's `../runs/<name>/weights/best.pt`).
        video_path: path to the match video.
        tracker_config: ultralytics tracker config name. "bytetrack.yaml"
            (default) needs no Re-ID model and is fast; "botsort.yaml" adds
            appearance-based Re-ID and handles longer occlusions better at
            higher compute cost -- worth trying in Phase 2 if ID switches
            during player pile-ups (corners, goal celebrations) turn out
            to be a problem with ByteTrack alone.
        conf: YOLO confidence threshold. Kept at 0.25 (matches
            phase0_1_pipeline.py's suggested inference conf) rather than
            YOLO's 0.5 default, since the ball class is small and easy to
            under-detect -- see phase0_1_pipeline.py's imgsz note.
        classes: optional whitelist of class names to keep (e.g.
            ["player", "ball"] to drop referees). None = keep everything
            the model detects.
        device: ultralytics device string ("cuda:0" / "cpu"). None keeps
            ultralytics' own auto-selection, which is the historical
            behaviour and what the tests rely on. run_pipeline() passes a
            resolved, PROBED device (see backend/pipeline/device.py) so a
            GPU that reports itself available but cannot actually run a
            kernel does not take the whole job down here at frame 1.

    Yields:
        One list[TrackedDetection] per video frame, in frame order.
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    class_name_by_id = model.names  # {0: "ball", 1: "goalkeeper", ...}

    class_filter_ids = None
    if classes is not None:
        wanted = set(classes)
        class_filter_ids = [i for i, name in class_name_by_id.items() if name in wanted]

    track_kwargs = dict(
        source=video_path,
        tracker=resolve_tracker_config(tracker_config),
        conf=conf,
        classes=class_filter_ids,
        persist=True,   # keep ByteTrack state across the stream call
        stream=True,    # generator, don't load the whole video into memory
        verbose=False,
    )
    # Only forward `device` when the caller resolved one. Omitting the key
    # entirely (rather than passing device=None) keeps ultralytics on its
    # own auto-selection path, byte-for-byte the previous behaviour.
    if device is not None:
        track_kwargs["device"] = device

    results = model.track(**track_kwargs)

    for frame_number, result in enumerate(results):
        frame_detections: list[TrackedDetection] = []

        if result.boxes is None or result.boxes.id is None:
            # No detections, or ByteTrack hasn't assigned IDs yet this frame
            # (can happen on the very first frames) -- yield an empty frame
            # rather than skipping it, so frame_number stays contiguous.
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
    ByteTrack implementation (bytetrack.py, adapted from the reference
    implementation). It sets frame_id when a track is activated and applies
    ByteTrack's one-frame confirmation delay for brand-new tracks; see the
    comments in bytetrack.py.

    Args:
        frames: same shape as detections.json from
            `ai/computer_vision/player detection/test.py`:
            [{"frame": int, "detections": [{"class": str, "confidence": float,
              "bbox": [x1, y1, x2, y2]}, ...]}, ...]
        classes: optional whitelist of class names to keep.

    Returns:
        list of list[TrackedDetection], same length and frame order as
        `frames`.
    """
    import numpy as np

    from ai.computer_vision.player_tracking.bytetrack import BYTETracker

    # Fixed class-name <-> id mapping for the WHOLE call, not recomputed per
    # frame. The previous supervision-based version did
    # `sorted(set(class_names))` inside the per-frame loop, which is only
    # self-consistent because class_id round-trips through the same dict
    # within that one frame -- but if frame 3 has only {"player","ball"}
    # while frame 4 has {"player","referee"}, "player" would silently get
    # a different numeric id in each frame. Doesn't break tracking itself
    # (IoU distance doesn't use class_id), but is a correctness footgun for
    # anything downstream that reads class_id directly instead of the
    # class_name we already reattach below.
    class_name_order = TRACKABLE_CLASS_NAMES | {BALL_CLASS_NAME, "referee"}
    class_name_order = sorted(class_name_order)
    name_to_id = {name: i for i, name in enumerate(class_name_order)}
    id_to_name = {i: name for name, i in name_to_id.items()}

    byte_tracker = BYTETracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
    byte_tracker.reset()  # fresh player_id counter for this call, see reset_id_counter()

    output: list[list[TrackedDetection]] = []

    for frame_entry in frames:
        frame_number = frame_entry["frame"]
        raw_detections = frame_entry["detections"]

        if classes is not None:
            raw_detections = [d for d in raw_detections if d["class"] in classes]

        # bytetrack.py's BYTETracker.update() expects an (N, 6) array of
        # [x1, y1, x2, y2, score, class_id] -- see BYTETracker.update()'s
        # docstring-equivalent unpacking (`d[0]..d[5]`) in bytetrack.py.
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
