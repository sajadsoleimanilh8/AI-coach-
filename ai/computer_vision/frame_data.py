"""
FrameData -- the single synchronised per-frame representation.

WHY THIS EXISTS
    Before this module the pipeline passed detector output around as loose
    lists and dicts: `frames` was a list[list[TrackedDetection]], the ball's
    pitch track was a separate list[dict] built inline in
    runner.py::_build_trajectories(), calibration was a bare
    `(H, confidence)` tuple, and there was no representation at all for
    field or goalpost detections (those models did not exist).

    With five models now feeding the pipeline, "which frame does this
    belong to" has to be answered in one place. FrameData is that place.

WHAT IT IS NOT
    Not a replacement for TrackedDetection or TrackingPoint -- it composes
    them. TrackedDetection stays the per-box detector output whose field
    names match the PlayerDetection table; TrackingPoint stays the enriched
    pitch-space trajectory sample. FrameData is the per-frame envelope that
    holds them together with the other four models' output.

HONESTY RULES ENCODED HERE (do not weaken)
    1. Every inferred value is distinguishable from a measured one.
       BallObservation.source is `detected` | `interpolated` | `missing`,
       and interpolated points carry the frame span they were inferred
       across. Downstream consumers can therefore refuse to count an
       interpolated ball as evidence of a pass or shot.
    2. Absence is representable. `ball=None`, `field=None`,
       `goalposts=[]` and `calibration.valid=False` are all legal states
       that mean "not measured this frame" -- never silently substituted
       with a plausible-looking default.
    3. Calibration validity is explicit, not re-derived by each consumer.
       CalibrationState.valid is computed once, from the confidence gate
       plus geometric consistency, and read everywhere.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ai.computer_vision.player_tracking.tracker import TrackedDetection
from ai.computer_vision.tactical_analysis.constants import (
    HOMOGRAPHY_CONFIDENCE_MIN,
    HOMOGRAPHY_MIN_POINT_SPREAD,
)


class BallSource(str, Enum):
    """Provenance of a ball position.

    `interpolated` exists so that possession/pass/shot heuristics can weigh
    (or reject) inferred positions. A ball position invented with no nearby
    detection is never emitted at all -- see interpolate_ball_gaps()."""

    detected = "detected"
    interpolated = "interpolated"
    missing = "missing"


class CalibrationSource(str, Enum):
    manual = "manual"          # operator clicks -- manual_calibration.py
    model = "model"            # automatic keypoint detection
    carried = "carried"        # reused from an earlier frame (static camera)
    none = "none"


class CameraMotion(str, Enum):
    static = "static"          # below the motion threshold -- reuse calibration
    panning = "panning"        # meaningful shift -- recalculation warranted
    cut = "cut"                # scene change -- previous calibration invalid
    unknown = "unknown"        # not measured (e.g. first frame)


@dataclass
class BallObservation:
    """The ball in one frame, in pixels and (when calibrated) pitch metres."""

    pixel_x: float | None
    pixel_y: float | None
    confidence: float | None
    source: BallSource = BallSource.missing
    pitch_x_m: float | None = None
    pitch_y_m: float | None = None
    # Frames since the last real detection. 0 on a detected frame. Lets a
    # consumer apply its own staleness policy instead of guessing.
    missing_frames: int = 0
    # Set only on interpolated points: the detected frames either side.
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
    """Output of the field-segmentation model: where the pitch is.

    Deliberately separate from CalibrationState. Field detection answers
    "where is the pitch in this image"; calibration answers "how do pixels
    map to pitch coordinates". Merging them was explicitly rejected -- see
    docs/pipeline_architecture.md."""

    polygon: list[tuple[float, float]]     # pixel-space vertices
    confidence: float
    area_fraction: float | None = None     # fraction of the frame covered

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
    # "left" | "right" once a homography exists to place it on the pitch.
    # None until then -- never guessed from pixel position alone, because
    # which side of the frame a goal appears on depends on camera angle.
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
    """Per-frame calibration status.

    `valid` is computed ONCE here rather than each consumer re-deriving
    `confidence >= HOMOGRAPHY_CONFIDENCE_MIN`. Several call sites in the
    old code applied that gate and several did not; centralising it is the
    point."""

    H: Any = None                                  # 3x3 ndarray or None
    confidence: float = 0.0
    reprojection_error_m: float | None = None
    n_points: int = 0
    source: CalibrationSource = CalibrationSource.none
    valid: bool = False
    # Populated when a geometric check fails, so a low-confidence frame can
    # be explained rather than merely flagged.
    invalid_reason: str | None = None
    # Frame this calibration was actually solved on (may be earlier than
    # the frame carrying it, when the camera is static and it is reused).
    solved_on_frame: int | None = None
    # RANSAC consensus behind this fit. None means "not measured" (an exact
    # 4-point fit, or a manual calibration), NOT "every point agreed" -- the
    # consensus gates below are skipped rather than assumed satisfied.
    n_inliers: int | None = None
    inlier_ratio: float | None = None
    # How many consecutive frames this calibration has been carried forward
    # without being re-solved, and the confidence it had when first solved.
    # 0 on the frame it was solved on. Lets a consumer apply its own
    # staleness policy, the same way BallObservation.missing_frames does.
    carried_frames: int = 0
    solved_confidence: float | None = None

    @classmethod
    def unavailable(cls, reason: str = "no calibration available") -> CalibrationState:
        return cls(H=None, confidence=0.0, valid=False, invalid_reason=reason)

    def evaluate(self, field_region: FieldRegion | None = None,
                 keypoints_px: Iterable[tuple[float, float]] | None = None,
                 frame_size: tuple[int, int] | None = None) -> CalibrationState:
        """
        Sets `valid` from the confidence gate plus geometric consistency.

        Five independent reasons to reject, each recorded in
        `invalid_reason` rather than collapsed into a bare False:
          - confidence below HOMOGRAPHY_CONFIDENCE_MIN (0.6), the existing
            project-wide gate;
          - too small a RANSAC consensus, by absolute count
            (HOMOGRAPHY_MIN_INLIERS) or as a fraction of the offered
            correspondences (HOMOGRAPHY_MIN_INLIER_RATIO). This gate is what
            makes it safe for confidence to be scored on the consensus set:
            four correspondences fit a homography with exactly zero residual
            whether or not they are correct, so "low error" is only evidence
            when it was earned across enough points;
          - keypoints that fall outside the detected pitch region, which
            indicates the landmarks were matched to the wrong part of the
            image even though they may reproject consistently among
            themselves;
          - keypoints covering less than HOMOGRAPHY_MIN_POINT_SPREAD of the
            frame area. Confidence CANNOT catch this: reprojection error is
            lowest precisely when the points are clustered, so a handful of
            landmarks bunched in one corner scores ~1.0 confidence while the
            matrix extrapolates badly across the rest of the frame. The
            spread gate is additional to the confidence gate, not a
            replacement for it;
          - and LAST, geometrically impossible fits -- degenerate/singular
            matrices, mirrored pitch orientation, collapsed or
            horizon-crossing projections (see
            homography_geometry_problems). Reprojection error is
            structurally blind to these, because a consistently mislabelled
            set of landmarks has small residuals by construction. This gate
            runs last because every gate above names a more specific
            property of the EVIDENCE, and those are the more actionable
            reasons to report when they apply.

        A rejection here degrades to the existing `valid=False` path -- the
        same honest-null behaviour used everywhere else. No default or
        guessed homography is ever substituted.

        `frame_size` is (width, height). When it is None the spread and
        geometry checks are SKIPPED, because hull area is meaningless without
        knowing what it is a fraction of. Likewise the consensus gate is
        skipped when `n_inliers` is None, which means "RANSAC did not run"
        (an exact 4-point fit, or a manual calibration) and not "every point
        agreed". Callers that have the frame should pass it; the remaining
        unthreaded caller is noted in auto_calibration.py.
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
            # Imported here rather than at module scope: homography.py pulls
            # in cv2, and frame_data.py is imported by lightweight consumers
            # (schema/serialisation paths) that should not pay for OpenCV.
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
            # LAST, deliberately. Every gate above names a specific, more
            # actionable property of the EVIDENCE -- too few points agreed,
            # they sat outside the pitch, they were bunched together. This
            # one inspects the resulting MATRIX and catches what survives
            # all of that: a fit that is structurally impossible rather than
            # merely poorly supported. Running it earlier would relabel a
            # clustered point set as "collapsed projection", which is the
            # same fact reported one step further from its cause.
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
    # Median feature displacement in pixels since the previous frame.
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

    # ---- convenience accessors ------------------------------------
    @property
    def pitch_coordinates_usable(self) -> bool:
        """The single question most downstream consumers actually ask."""
        return self.calibration.valid

    def players_by_team(self) -> dict[str | None, list[TrackedDetection]]:
        """Groups this frame's players by team_id.

        Exists because pooling every player regardless of team_id was a
        real bug in _score_team_intelligence() -- having the split
        available on the frame object makes the correct thing the easy
        thing."""
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


# ----------------------------------------------------------------------
# Ball gap interpolation
# ----------------------------------------------------------------------

def interpolate_ball_gaps(frames: list[FrameData], *, max_gap: int = 5,
                          max_speed_px_per_frame: float = 120.0) -> int:
    """
    Fills SHORT ball-detection gaps by linear interpolation between two
    real detections, and marks every filled point `source=interpolated`.

    Constraints that keep this honest -- a fabricated ball position is
    worse than an absent one, because possession and pass detection treat
    a ball position as evidence:

      - Only gaps of at most `max_gap` frames are filled. Longer gaps stay
        `missing`; the ball genuinely left frame or was occluded too long
        to infer.
      - Both endpoints must be real detections. A gap at the start or end
        of the clip is never extrapolated, only interpolated.
      - The implied speed between the two endpoints must be physically
        plausible (<= `max_speed_px_per_frame`). Two detections far apart
        in space and time are more likely to be a missed detection plus a
        false positive than one continuous ball path, and interpolating
        between them would draw a straight line through positions the ball
        never occupied.

    Returns the number of frames filled.
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
            continue                     # implausible -- leave the gap open

        for step in range(1, gap + 1):
            t = step / (b - a)
            frames[a + step].ball = BallObservation(
                pixel_x=ball_a.pixel_x + dx * t,
                pixel_y=ball_a.pixel_y + dy * t,
                confidence=None,          # not a detection -- has no detector confidence
                source=BallSource.interpolated,
                missing_frames=step,
                interpolated_between=(frames[a].frame_id, frames[b].frame_id),
            )
            filled += 1

    # Anything still without a ball is explicitly `missing`, with a running
    # count of how long it has been gone.
    since = 0
    for f in frames:
        if f.ball is not None and f.ball.usable:
            since = 0 if f.ball.source is BallSource.detected else since + 1
        else:
            since += 1
            f.ball = BallObservation(pixel_x=None, pixel_y=None, confidence=None,
                                     source=BallSource.missing, missing_frames=since)
    return filled
