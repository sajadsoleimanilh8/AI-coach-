"""
Builds per-player trajectories from tracked detections and computes
speed / distance / acceleration in pitch meters.

Implements the formulas from docs/tracking_system_design.md §7 exactly:
    distance     = sqrt((x2-x1)^2 + (y2-y1)^2)
    speed        = distance / time
    acceleration = (speed2 - speed1) / time

...but computed in PITCH meters (via tactical_analysis/homography.py), not
pixels -- pixel-space "speed" is meaningless (a player far from camera
moves fewer pixels/frame than the same real speed up close). This is the
concrete link between player_tracking (this module) and tactical_analysis
(homography) that docs/database_schema.md's PlayerTracking table assumes.

Per Analysis Logic Design v3 §0.2/§0.6: any frame where homography_confidence
is below HOMOGRAPHY_CONFIDENCE_MIN is dropped from the trajectory entirely
(pitch coordinates unusable), not imputed -- a speed computed across a
dropped frame would silently use a bad position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai.computer_vision.frame_data import CalibrationState as CalibrationLike
    from ai.computer_vision.pose_estimation.pose import OrientationResult

from ai.computer_vision.player_tracking.tracker import TrackedDetection
from ai.computer_vision.tactical_analysis.constants import HOMOGRAPHY_CONFIDENCE_MIN
from ai.computer_vision.tactical_analysis.homography import pixel_to_pitch

#: Largest calibration/detection gap, in frames, that speed and distance are
#: still integrated across. Per-frame calibration means a player's pitch
#: position is unavailable on the frames where calibration failed, so
#: consecutive samples for one player are often several frames apart. Up to
#: this gap the straight line between two samples is a fair stand-in for the
#: path run; past it the displacement understates the path badly enough that
#: emitting a speed would be a guess, so None is emitted instead.
MAX_SPEED_GAP_FRAMES = 5


@dataclass
class TrackingPoint:
    """One row of docs/database_schema.md's PlayerTracking table."""

    match_id: str
    player_id: int
    frame_id: int
    team_id: str | None
    pixel_x: float
    pixel_y: float
    pitch_x_m: float | None
    pitch_y_m: float | None
    homography_confidence: float | None
    speed: float | None          # m/s
    distance: float | None       # meters, delta since previous point
    acceleration: float | None   # m/s^2
    body_orientation_deg: float | None = None          # shoulder-line angle, 0-360; None if not sampled/not visible
    body_orientation_confidence: float | None = None   # min shoulder-landmark visibility, 0-1


def enrich_with_pitch_coordinates(
    frames: list[list[TrackedDetection]],
    match_id: str,
    H,
    homography_confidence: float,
    fps: float,
    calibration_by_frame: dict[int, CalibrationLike] | None = None,
) -> dict[int, list[TrackingPoint]]:
    """
    Convert tracked pixel detections into pitch-meter TrackingPoints, then
    compute speed/distance/acceleration per player along the way.

    TWO PROJECTION MODES
    Pass `calibration_by_frame` -- {frame_number: CalibrationState} -- and
    each detection is projected through THAT frame's own homography, gated
    on that frame's `.valid`. This is the mode the video pipeline uses. It
    matters because the broadcast camera pans: one match-wide matrix maps a
    stationary player to a pitch position that slides as the camera moves,
    so every time-resolved metric would read camera motion as player motion.

    Omit it and the legacy single-matrix behaviour applies: `H` is used for
    every frame, gated once on `homography_confidence`. That path is still
    correct for a MANUAL calibration, which is by construction one fixed
    matrix for the whole clip, and it is what the synthetic tests use.

    A frame whose calibration is not valid yields pitch_x_m/pitch_y_m =
    None -- never a carried-forward or match-representative matrix, which
    would be a plausible-looking wrong position rather than an honest gap.
    Downstream already treats None as "unavailable".
    """
    trajectories: dict[int, list[TrackingPoint]] = {}
    last_point_by_player: dict[int, TrackingPoint] = {}
    last_speed_by_player: dict[int, float] = {}

    per_frame = calibration_by_frame is not None
    legacy_usable = H is not None and homography_confidence >= HOMOGRAPHY_CONFIDENCE_MIN

    for frame_number, detections in enumerate(frames):
        if per_frame:
            cal = calibration_by_frame.get(frame_number)
            frame_H = cal.H if (cal is not None and cal.valid) else None
            frame_confidence = cal.confidence if cal is not None else 0.0
        else:
            frame_H = H if legacy_usable else None
            frame_confidence = homography_confidence

        for det in detections:
            if det.class_name not in ("player", "goalkeeper"):
                continue  # ball/referee trajectories, if needed, go through the same
                          # function separately -- kept out here so player speed/
                          # distance aggregates (ACWR, sprint counts) aren't polluted

            px, py = det.foot_point()

            if frame_H is None:
                point = TrackingPoint(
                    match_id=match_id, player_id=det.player_id, frame_id=frame_number,
                    team_id=det.team_id, pixel_x=px, pixel_y=py,
                    pitch_x_m=None, pitch_y_m=None,
                    homography_confidence=frame_confidence,
                    speed=None, distance=None, acceleration=None,
                )
                trajectories.setdefault(det.player_id, []).append(point)
                continue

            pitch_x, pitch_y = pixel_to_pitch(px, py, frame_H)

            # Frames between `prev` and now may have had no valid calibration,
            # so the elapsed time is the FRAME GAP, not one frame interval --
            # dividing a multi-frame displacement by 1/fps would report a
            # sprint every time calibration blinked. Beyond MAX_SPEED_GAP_
            # FRAMES the straight line between two samples stops resembling
            # the path actually run, so no kinematics are emitted at all.
            prev = last_point_by_player.get(det.player_id)
            distance = speed = acceleration = None
            if prev is not None and prev.pitch_x_m is not None:
                gap = frame_number - prev.frame_id
                if 0 < gap <= MAX_SPEED_GAP_FRAMES:
                    dt = gap / fps
                    distance = math.hypot(pitch_x - prev.pitch_x_m, pitch_y - prev.pitch_y_m)
                    speed = distance / dt
                    prev_speed = last_speed_by_player.get(det.player_id)
                    if prev_speed is not None:
                        acceleration = (speed - prev_speed) / dt
                else:
                    last_speed_by_player.pop(det.player_id, None)

            point = TrackingPoint(
                match_id=match_id, player_id=det.player_id, frame_id=frame_number,
                team_id=det.team_id, pixel_x=px, pixel_y=py,
                pitch_x_m=pitch_x, pitch_y_m=pitch_y,
                homography_confidence=frame_confidence,
                speed=speed, distance=distance, acceleration=acceleration,
            )
            trajectories.setdefault(det.player_id, []).append(point)
            last_point_by_player[det.player_id] = point
            if speed is not None:
                last_speed_by_player[det.player_id] = speed

    return trajectories


def attach_body_orientation(
    trajectories: dict[int, list[TrackingPoint]],
    orientation_by_player_frame: dict[tuple[int, int], OrientationResult],
) -> None:
    """
    Mutates `trajectories` in place, filling in body_orientation_deg/
    body_orientation_confidence on the TrackingPoints that have a pose
    reading.

    Kept as a separate pass rather than computed inline inside
    enrich_with_pitch_coordinates() on purpose: pose estimation
    (ai/computer_vision/pose_estimation/pose.py) is expensive enough that
    it's only run on a stride-sampled subset of frames (see
    POSE_SAMPLE_STRIDE in backend/pipeline/runner.py), not every frame
    like the pixel->pitch homography above -- mixing a sparse data source
    into the main per-frame loop would make that loop's "is this frame
    usable" logic harder to reason about for no benefit, since orientation
    readings are sparse by design, not by failure.

    Args:
        trajectories: output of enrich_with_pitch_coordinates(), mutated
            in place.
        orientation_by_player_frame: {(player_id, frame_number): OrientationResult}
            for whichever (player, frame) pairs were actually sampled --
            most (player_id, frame_number) combinations in a trajectory
            will simply have no entry here, and those TrackingPoints keep
            their default None/None (dataclass defaults above), which is
            the correct "not measured" state, not an error.
    """
    for player_id, points in trajectories.items():
        for point in points:
            key = (player_id, point.frame_id)
            reading = orientation_by_player_frame.get(key)
            if reading is None:
                continue
            point.body_orientation_deg = reading.orientation_deg
            point.body_orientation_confidence = reading.confidence


def total_distance_covered(trajectory: list[TrackingPoint]) -> float:
    """Sum of per-frame distance deltas, ignoring frames with no distance
    (first frame of the trajectory, or frames dropped for low homography
    confidence)."""
    return sum(p.distance for p in trajectory if p.distance is not None)


def sprint_count(trajectory: list[TrackingPoint], sprint_threshold_ms: float = 7.0) -> int:
    """
    Number of frames where instantaneous speed exceeds sprint_threshold_ms
    (7.0 m/s ~= 25.2 km/h is a commonly used sprint-speed threshold in
    sports-science literature). Used as `sprint_load` input for the MVP
    Injury Risk fallback in docs/data analysis.md §4.
    """
    return sum(1 for p in trajectory if p.speed is not None and p.speed > sprint_threshold_ms)
