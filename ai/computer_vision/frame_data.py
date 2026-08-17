"""
FrameData -- the single synchronised per-frame representation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from ai.computer_vision.player_tracking.tracker import TrackedDetection
from ai.computer_vision.tactical_analysis.constants import (
    HOMOGRAPHY_CONFIDENCE_MIN,
    HOMOGRAPHY_MIN_POINT_SPREAD,
)


class BallSource(str, Enum):
    """Provenance of a ball position."""

    detected = "detected"
    interpolated = "interpolated"
    missing = "missing"


class CalibrationSource(str, Enum):
    manual = "manual"
    model = "model"
    carried = "carried"
    none = "none"


class CameraMotion(str, Enum):
    static = "static"
    panning = "panning"
    cut = "cut"
    unknown = "unknown"


@dataclass
class BallObservation:
    """The ball in one frame, in pixels and (when calibrated) pitch metres."""

    pixel_x: float | None
    pixel_y: float | None
    confidence: float | None
    source: BallSource = BallSource.missing
    pitch_x_m: float | None = None
    pitch_y_m: float | None = None
    missing_frames: int = 0
    interpolated_between: tuple[int, int] | None = None
    velocity_ms: float | None = None

    @property
    def visible(self) -> bool:
        return self.source is BallSource.detected

    @property
    def usable(self) -> bool:
        """Has a position at all (detected or inferred)."""
        return self.pixel_x is not None


@dataclass
class FieldRegion:
    """Output of the field-segmentation model: where the pitch is."""

    polygon: list[tuple[float, float]]
    confidence: float
    area_fraction: float | None = None

    def contains(self, x: float, y: float) -> bool:
        """Ray-casting point-in-polygon. Used to reject calibration
        keypoints that fall outside the detected pitch, which is a
        geometric inconsistency the reprojection error alone will not
        catch."""
        pts = self.polygon
        if len(pts) < 3:
            return False
        inside = False
        j = len(pts) - 1
        for i in range(len(pts)):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if (yi > y) != (yj > y):
                denom = (yj - yi) or 1e-12
                if x < (xj - xi) * (y - yi) / denom + xi:
                    inside = not inside
            j = i
        return inside


@dataclass
class GoalpostDetection:
    """One goal mouth in pixel space, from the goalpost model."""

    x: float
    y: float
    width: float
    height: float
    confidence: float
    pitch_x_m: float | None = None
    pitch_y_m: float | None = None
    side: str | None = None

    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def goal_line_y(self) -> float:
        """Bottom edge -- where the posts meet the pitch plane. This is the
        point that lies on z=0 and can therefore be projected through the
        homography, for the same reason player anchoring uses foot point
        rather than bbox centre (see TrackedDetection.foot_point())."""
        return self.y + self.height


@dataclass
class CalibrationState:
    """Per-frame calibration status."""

    H: Any = None
    confidence: float = 0.0
    reprojection_error_m: float | None = None
    n_points: int = 0
    source: CalibrationSource = CalibrationSource.none
    valid: bool = False
    invalid_reason: str | None = None
    solved_on_frame: int | None = None
    n_inliers: int | None = None
    inlier_ratio: float | None = None
    carried_frames: int = 0
    solved_confidence: float | None = None

    @classmethod
    def unavailable(cls, reason: str = "no calibration available") -> "CalibrationState":
        return cls(H=None, confidence=0.0, valid=False, invalid_reason=reason)

    def evaluate(self, field_region: FieldRegion | None = None,
                 keypoints_px: Iterable[tuple[float, float]] | None = None,
                 frame_size: tuple[int, int] | None = None) -> "CalibrationState":
        """
        Sets `valid` from the confidence gate plus geometric consistency.
        """
        from ai.computer_vision.tactical_analysis.constants import (
            HOMOGRAPHY_MIN_INLIER_RATIO,
            HOMOGRAPHY_MIN_INLIERS,
        )

        if self.H is None:
            self.valid = False
            self.invalid_reason = self.invalid_reason or "no homography"
            return self
        if self.confidence < HOMOGRAPHY_CONFIDENCE_MIN:
            self.valid = False
            self.invalid_reason = (
                f"confidence {self.confidence:.3f} < "
                f"HOMOGRAPHY_CONFIDENCE_MIN {HOMOGRAPHY_CONFIDENCE_MIN}")
            return self
        if self.n_inliers is not None:
            if self.n_inliers < HOMOGRAPHY_MIN_INLIERS:
                self.valid = False
                self.invalid_reason = (
                    f"only {self.n_inliers} RANSAC inliers of {self.n_points} "
                    f"correspondences < HOMOGRAPHY_MIN_INLIERS "
                    f"{HOMOGRAPHY_MIN_INLIERS} (a consensus this small fits "
                    "any four points exactly and is not evidence)")
                return self
            if (self.inlier_ratio is not None
                    and self.inlier_ratio < HOMOGRAPHY_MIN_INLIER_RATIO):
                self.valid = False
                self.invalid_reason = (
                    f"RANSAC inlier ratio {self.inlier_ratio:.2f} < "
                    f"HOMOGRAPHY_MIN_INLIER_RATIO {HOMOGRAPHY_MIN_INLIER_RATIO} "
                    f"({self.n_inliers}/{self.n_points} correspondences agreed; "
                    "most detected landmarks contradict the accepted fit)")
                return self
        if field_region is not None and keypoints_px:
            pts = list(keypoints_px)
            outside = sum(1 for (x, y) in pts if not field_region.contains(x, y))
            if pts and outside / len(pts) > 0.5:
                self.valid = False
                self.invalid_reason = (
                    f"{outside}/{len(pts)} calibration keypoints fall outside "
                    f"the detected pitch region")
                return self
        if frame_size is not None and keypoints_px:
            from ai.computer_vision.tactical_analysis.homography import point_spread

            pts = list(keypoints_px)
            spread = point_spread(pts, frame_size)
            if spread < HOMOGRAPHY_MIN_POINT_SPREAD:
                self.valid = False
                self.invalid_reason = (
                    f"calibration points span {spread:.4f} of the frame area "
                    f"< HOMOGRAPHY_MIN_POINT_SPREAD {HOMOGRAPHY_MIN_POINT_SPREAD} "
                    f"({len(pts)} points, too clustered to constrain the "
                    "homography away from where they sit)")
                return self
        if frame_size is not None:
            from ai.computer_vision.tactical_analysis.homography import (
                homography_geometry_problems,
            )

            problems = homography_geometry_problems(
                self.H, frame_size,
                keypoints_px=list(keypoints_px) if keypoints_px else None)
            if problems:
                self.valid = False
                self.invalid_reason = "; ".join(problems)
                return self
        self.valid = True
        self.invalid_reason = None
        return self


@dataclass
class CameraState:
    """Frame-to-frame camera motion, used to decide whether an existing
    calibration can be carried forward."""

    motion: CameraMotion = CameraMotion.unknown
    shift_px: float | None = None
    recalibration_advised: bool = False


@dataclass
class FrameData:
    """Everything known about one video frame, from every model."""

    frame_id: int
    timestamp: float
    players: list[TrackedDetection] = field(default_factory=list)
    ball: BallObservation | None = None
    field_region: FieldRegion | None = None
    goalposts: list[GoalpostDetection] = field(default_factory=list)
    calibration: CalibrationState = field(default_factory=CalibrationState.unavailable)
    camera: CameraState = field(default_factory=CameraState)

    @property
    def pitch_coordinates_usable(self) -> bool:
        """The single question most downstream consumers actually ask."""
        return self.calibration.valid

    def players_by_team(self) -> dict[str | None, list[TrackedDetection]]:
        """Groups this frame's players by team_id."""
        out: dict[str | None, list[TrackedDetection]] = {}
        for det in self.players:
            out.setdefault(det.team_id, []).append(det)
        return out

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "n_players": len(self.players),
            "ball": None if self.ball is None else {
                "source": self.ball.source.value,
                "confidence": self.ball.confidence,
                "pitch_x_m": self.ball.pitch_x_m,
                "pitch_y_m": self.ball.pitch_y_m,
                "missing_frames": self.ball.missing_frames,
            },
            "field_detected": self.field_region is not None,
            "n_goalposts": len(self.goalposts),
            "calibration": {
                "valid": self.calibration.valid,
                "confidence": self.calibration.confidence,
                "source": self.calibration.source.value,
                "invalid_reason": self.calibration.invalid_reason,
                "reprojection_error_m": self.calibration.reprojection_error_m,
                "n_points": self.calibration.n_points,
                "n_inliers": self.calibration.n_inliers,
                "inlier_ratio": self.calibration.inlier_ratio,
                "carried_frames": self.calibration.carried_frames,
                "solved_on_frame": self.calibration.solved_on_frame,
            },
            "camera": {
                "motion": self.camera.motion.value,
                "shift_px": self.camera.shift_px,
            },
        }



def interpolate_ball_gaps(frames: list[FrameData], *, max_gap: int = 5,
                          max_speed_px_per_frame: float = 120.0) -> int:
    """
    Fills SHORT ball-detection gaps by linear interpolation between two
    real detections, and marks every filled point `source=interpolated`.
    """
    detected_idx = [i for i, f in enumerate(frames)
                    if f.ball is not None and f.ball.source is BallSource.detected]
    filled = 0

    for a, b in zip(detected_idx, detected_idx[1:]):
        gap = b - a - 1
        if gap <= 0 or gap > max_gap:
            continue
        ball_a, ball_b = frames[a].ball, frames[b].ball
        if ball_a is None or ball_b is None:
            continue
        dx = ball_b.pixel_x - ball_a.pixel_x
        dy = ball_b.pixel_y - ball_a.pixel_y
        dist = math.hypot(dx, dy)
        if dist / (b - a) > max_speed_px_per_frame:
            continue

        for step in range(1, gap + 1):
            t = step / (b - a)
            frames[a + step].ball = BallObservation(
                pixel_x=ball_a.pixel_x + dx * t,
                pixel_y=ball_a.pixel_y + dy * t,
                confidence=None,
                source=BallSource.interpolated,
                missing_frames=step,
                interpolated_between=(frames[a].frame_id, frames[b].frame_id),
            )
            filled += 1

    since = 0
    for f in frames:
        if f.ball is not None and f.ball.usable:
            since = 0 if f.ball.source is BallSource.detected else since + 1
        else:
            since += 1
            f.ball = BallObservation(pixel_x=None, pixel_y=None, confidence=None,
                                     source=BallSource.missing, missing_frames=since)
    return filled
