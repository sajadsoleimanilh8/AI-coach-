"""
Shared constants for tactical analysis and scoring functions.
Analysis Logic Design v3 (docs/data analysis.md §0.4).
"""

PRESSURE_RADIUS_M = 5.0  # opponent within this = "applying pressure"
PRESSURE_THRESHOLD = 0.5  # pressure_level above this = "under pressure"
RETENTION_WINDOW_S = 3.0  # min time to count possession as "retained"
TOUCH_EVAL_WINDOW_S = 2.0  # window after first touch to evaluate outcome
MAX_TOUCH_DISTANCE_M = 5.0  # ball travel beyond this after touch = poor control
DECISION_TIME_MIN_S = 0.3  # fastest realistic decision
DECISION_TIME_MAX_S = 2.0  # slowest before treated as "too slow"
MIN_SAMPLE_EVENTS = 5  # below this, flag score as "low_sample"
HOMOGRAPHY_CONFIDENCE_MIN = 0.6  # below this, pitch coordinates are unusable
TEAM_ASSIGNMENT_CONFIDENCE_MIN = 0.5  # below this, treat team_id as unassigned
SCHEMA_VERSION = "v3"

BALL_CONTROL_RADIUS_M = 1.5  # player within 1.5m of ball = possession/touch
PASS_MIN_DISTANCE_M = 3.0  # min ball distance to qualify as a pass

# ---------------------------------------------------------------------------
# IMAGE-SPACE FALLBACK SCALE
#
# Used ONLY when calibration did not validate and possession is therefore
# resolved in pixels rather than pitch metres (see possession.py's
# get_ball_possessor_image_space). A tracked player's bounding-box HEIGHT is
# the one length in the image whose real-world size is approximately known,
# so it is the local pixels-per-metre ruler at that player's depth in frame.
#
# 1.80 m is the mean height of a professional outfield footballer, and the
# box is the whole standing player. What this ruler is and is not:
#
#   - It is a real measurement of the detector's own box, converted by one
#     stated constant. It is not a fabricated coordinate, and it never
#     produces one: pitch_x_m/pitch_y_m stay None on every event derived
#     this way.
#   - It is per-player, so it tracks perspective correctly -- a player at
#     the far touchline has a shorter box and therefore a finer ruler,
#     which is exactly the depth correction a single global scale lacks.
#   - It is wrong for a player who is not standing upright (sliding,
#     jumping, bent over a tackle) and for a box clipped by the frame edge.
#     Both shrink the box and inflate the estimated distance, which biases
#     this fallback toward MISSING possession rather than inventing it.
#
# Individual height variation is roughly +/-8%, so a distance estimated
# this way carries at least that much error before pose and clipping are
# considered. Treat anything derived from it as an estimate with a stated
# instrument, never as a metric measurement.
PLAYER_HEIGHT_M = 1.80

# A box shorter than this is too small for its height to be a usable ruler:
# rounding alone is several percent, and at this size the detector is
# frequently boxing a partially-occluded player. Possession is declined
# rather than estimated from it.
MIN_SCALE_BOX_HEIGHT_PX = 24.0
SHOT_VELOCITY_MIN_MS = 12.0  # min ball velocity to qualify as a shot
MAX_USEFUL_SEPARATION_GAIN_M = 5.0  # Off-ball movement space creation max
MAX_ACCEPTABLE_LINE_DEVIATION_M = 8.0  # Defensive line max deviation
IDEAL_DEFENSIVE_SPACING_M = 10.0  # Ideal distance between adjacent defenders
SPACING_TOLERANCE_M = 3.0  # Spacing Gaussian tolerance
REACTION_SPEED_REFERENCE_MS = 3.0  # Closing speed reference for threat reaction

# Pose / body orientation.
# NOTE on calibration status: these two are uncalibrated starting points,
# same caveat as Formation Detection's NORMALIZATION_CONSTANT (see
# docs/data analysis.md §3's calibration procedure) -- not yet tuned
# against real match clips with known-correct scan counts. Do not present
# scanning_behavior_score's absolute value to judges as validated; the
# relative ranking (more scans = higher score) is sound, the specific
# scale is not yet.
SCAN_ANGLE_THRESHOLD_DEG = 30.0  # min shoulder-angle change between consecutive
                                  # pose readings to count as one "scan" (head/shoulder check)
SCAN_MAX_GAP_S = 2.0  # ignore orientation deltas spanning more than this many
                       # seconds between readings -- likely a play stoppage or a
                       # tracking gap, not one continuous scanning motion
SCANS_PER_MINUTE_TARGET = 8.0  # scans/min mapped to a 100 score; placeholder,
                                # needs the same real-clip calibration procedure
                                # as NORMALIZATION_CONSTANT before judges see it
                                # presented as anything but a relative ranking

# Team assignment (jersey-colour clustering).
# UNCALIBRATED STARTING POINT -- same caveat as SCANS_PER_MINUTE_TARGET/
# IDEAL_SQUARENESS_DEG above: these are defensible starting values, not
# numbers tuned against real broadcast footage yet. See
# ai/computer_vision/tactical_analysis/team_assignment.py's module
# docstring for the calibration procedure these should eventually go
# through.
TEAM_ASSIGNMENT_SAMPLE_STRIDE = 5  # every Nth frame is cropped/clustered; matches FRAME_PERSIST_STRIDE's default density
TEAM_ASSIGNMENT_CROP_TOP_FRACTION = 0.5  # only the top half of a player bbox is sampled -- biases toward torso/jersey over shorts/socks/grass at the feet
TEAM_ASSIGNMENT_MIN_USABLE_PIXELS = 40  # crops with fewer non-grass/non-skin pixels than this are discarded as unusable
TEAM_ASSIGNMENT_KMEANS_MAX_ITERS = 50  # Lloyd's-algorithm iteration cap for the k=2 fit; bounds worst-case runtime, real fits converge well before this

# OpenCV hue is 0-179 (not 0-359). Grass range: yellow-green to blue-green.
GRASS_HUE_RANGE = (35, 85)
GRASS_MIN_SATURATION = 40  # below this, treat as too desaturated to call "grass" -- avoids masking dark/shadowed jersey pixels that happen to fall in the green hue band
GRASS_MIN_VALUE = 40

# Skin tone wraps across the 0/179 hue boundary (red-adjacent), hence two ranges.
SKIN_HUE_RANGES = ((0, 20), (170, 179))
SKIN_SATURATION_RANGE = (20, 150)
SKIN_MIN_VALUE = 60

# Passing vision / finishing efficiency.
PASS_DIRECTION_SECTORS = 8  # compass buckets for pass-angle-diversity entropy; not yet calibrated against real distributions of pass angles
SHOT_DISTANCE_TOLERANCE_M = 18.0  # ~edge-of-box; uncalibrated Gaussian tolerance for finishing_efficiency_score's distance sub-score, needs real-shot-data tuning
MAX_REALISTIC_SHOOTING_ANGLE_DEG = 90.0  # angle at/beyond which sight-of-goal is treated as saturated to full score; uncalibrated placeholder

# Body Orientation Score: "squareness" (0deg = shoulders parallel to the
# pitch's long axis, i.e. facing straight down the pitch; 90deg = fully
# side-on) mapped to a Gaussian centered on IDEAL_SQUARENESS_DEG. This
# encodes the common coaching cue that a "half-turned"/open stance
# (angled diagonally, not square-on or fully side-on) maximizes a
# player's visual field of both the ball and forward options. Same
# calibration caveat as SCANS_PER_MINUTE_TARGET above -- this is a
# defensible starting point grounded in a real coaching heuristic, not a
# number fit to labeled data, and should be described to judges that way.
IDEAL_SQUARENESS_DEG = 45.0
SQUARENESS_TOLERANCE_DEG = 25.0

# Pitch Dimensions (Standard FIFA Pitch in meters)
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
GOAL_WIDTH_M = 7.32
SIX_YARD_DEPTH_M = 5.5
SIX_YARD_WIDTH_M = 18.32
PENALTY_AREA_DEPTH_M = 16.5
PENALTY_AREA_WIDTH_M = 40.32
PENALTY_SPOT_DISTANCE_M = 11.0
CENTER_CIRCLE_RADIUS_M = 9.15

_L = PITCH_LENGTH_M
_W = PITCH_WIDTH_M
_CY = _W / 2.0

# tests/test_homography.py asserts against the six-yard-box and
# center-circle-top/bottom entries below. Keep them when merging constants into
# this file.
REFERENCE_POINTS = {
    "corner_bottom_left": (0.0, 0.0),
    "corner_top_left": (0.0, _W),
    "corner_bottom_right": (_L, 0.0),
    "corner_top_right": (_L, _W),
    "halfway_bottom": (_L / 2.0, 0.0),
    "halfway_top": (_L / 2.0, _W),
    "center_spot": (_L / 2.0, _CY),
    "center_circle_top": (_L / 2.0, _CY + CENTER_CIRCLE_RADIUS_M),
    "center_circle_bottom": (_L / 2.0, _CY - CENTER_CIRCLE_RADIUS_M),
    "left_penalty_area_bottom_near": (0.0, _CY - PENALTY_AREA_WIDTH_M / 2.0),
    "left_penalty_area_top_near": (0.0, _CY + PENALTY_AREA_WIDTH_M / 2.0),
    "left_penalty_area_bottom_far": (PENALTY_AREA_DEPTH_M, _CY - PENALTY_AREA_WIDTH_M / 2.0),
    "left_penalty_area_top_far": (PENALTY_AREA_DEPTH_M, _CY + PENALTY_AREA_WIDTH_M / 2.0),
    "left_six_yard_bottom_near": (0.0, _CY - SIX_YARD_WIDTH_M / 2.0),
    "left_six_yard_top_near": (0.0, _CY + SIX_YARD_WIDTH_M / 2.0),
    "left_six_yard_bottom_far": (SIX_YARD_DEPTH_M, _CY - SIX_YARD_WIDTH_M / 2.0),
    "left_six_yard_top_far": (SIX_YARD_DEPTH_M, _CY + SIX_YARD_WIDTH_M / 2.0),
    "left_penalty_spot": (PENALTY_SPOT_DISTANCE_M, _CY),
    "right_penalty_area_bottom_near": (_L, _CY - PENALTY_AREA_WIDTH_M / 2.0),
    "right_penalty_area_top_near": (_L, _CY + PENALTY_AREA_WIDTH_M / 2.0),
    "right_penalty_area_bottom_far": (_L - PENALTY_AREA_DEPTH_M, _CY - PENALTY_AREA_WIDTH_M / 2.0),
    "right_penalty_area_top_far": (_L - PENALTY_AREA_DEPTH_M, _CY + PENALTY_AREA_WIDTH_M / 2.0),
    "right_six_yard_bottom_near": (_L, _CY - SIX_YARD_WIDTH_M / 2.0),
    "right_six_yard_top_near": (_L, _CY + SIX_YARD_WIDTH_M / 2.0),
    "right_six_yard_bottom_far": (_L - SIX_YARD_DEPTH_M, _CY - SIX_YARD_WIDTH_M / 2.0),
    "right_six_yard_top_far": (_L - SIX_YARD_DEPTH_M, _CY + SIX_YARD_WIDTH_M / 2.0),
    "right_penalty_spot": (_L - PENALTY_SPOT_DISTANCE_M, _CY),
}

# ---------------------------------------------------------------------------
# Automatic calibration + temporal stability (auto_calibration.py).
#
# UNCALIBRATED STARTING POINTS -- same caveat as SCANS_PER_MINUTE_TARGET and
# the team-assignment constants above. These are defensible values derived
# from the geometry, not numbers fitted to labelled footage.
# ---------------------------------------------------------------------------

# Median background-feature displacement between consecutive frames, in
# pixels, below which the camera counts as static and its existing
# calibration is reused untouched. 2 px at 1280x720 is roughly the noise
# floor of sparse optical flow on real footage -- set it much lower and
# tracking jitter alone forces a recalculation every frame, which defeats
# the purpose of the reuse path.
CAMERA_STATIC_SHIFT_PX = 2.0

# Displacement above which the shift is treated as a scene cut rather than
# a pan. A broadcast pan moves single-digit pixels per frame at 25-30 fps;
# 40 px between consecutive frames is not a camera movement, it is a
# different shot, and the previous homography cannot survive it.
CAMERA_CUT_SHIFT_PX = 40.0

# How far, in pitch metres, two consecutive accepted homographies may
# disagree about where the same image points land before the newer one is
# rejected as implausible. A real camera pan moves the projected view by a
# few metres between recalculations; a 15 m jump means one of the two fits
# is wrong, and the prior stable one is the safer bet (see
# auto_calibration.homography_jump_m, which measures this in metres rather
# than differencing matrix entries -- a homography is only defined up to
# scale, so entry-wise differences have no meaningful unit).
CALIBRATION_MAX_JUMP_M = 15.0

# Minimum convex-hull area of the calibration points, as a FRACTION of the
# frame area, for a homography to be accepted.
#
# WHY THIS IS SEPARATE FROM HOMOGRAPHY_CONFIDENCE_MIN (and does not replace
# it): confidence is derived from reprojection error, and reprojection error
# is smallest exactly when the points are clustered. Four landmarks within
# one corner of the frame fit a homography almost perfectly among
# themselves -- near-zero error, confidence ~1.0 -- while saying nothing
# about the rest of the image, where every player actually is. The
# extrapolation error off in the far half can be tens of metres. The
# confidence gate cannot see this; it is a measure of self-consistency, not
# of coverage. Both gates are applied, and a fit must clear both.
#
# UNCALIBRATED STARTING POINT -- same caveat as SCANS_PER_MINUTE_TARGET and
# the auto-calibration constants above. 0.05 (5% of frame area) is a
# geometric argument, not a fitted number: it is roughly the hull of four
# points spanning ~22% of the frame in each axis, below which the fit is
# extrapolating over more than 4x the region it was constrained on. It has
# NOT been tuned against real broadcast footage yet. Tune it by sweeping
# this value over real calibrations with known-good ground truth and
# picking the knee -- do not present the current value as validated.
HOMOGRAPHY_MIN_POINT_SPREAD = 0.05

# ---------------------------------------------------------------------------
# Robust homography fitting: outlier rejection and consensus gates.
#
# WHY THESE EXIST (measured, 2026-08-16 -- do not delete this note)
#   On broadcast footage the calibration model returns a mix of well-placed
#   landmarks and landmarks placed on the WRONG pitch feature (the mirrored /
#   confused-identity failure already documented in
#   scripts/validate_auto_calibration.py). Fitting all of them together gives
#   a mean reprojection error of 7-25 m, so the confidence gate rejected
#   every frame -- correctly, because that fit IS wrong.
#
#   RANSAC's whole purpose is to separate those two populations. The fix is
#   to let it do so and then judge the CONSENSUS SET, while gating hard on
#   how big that consensus actually is. Without the size gates below this
#   would be a way to manufacture confidence: RANSAC will always find four
#   points that agree perfectly, and four points agreeing perfectly means
#   nothing at all.
# ---------------------------------------------------------------------------

# Max reprojection error, IN PITCH METRES, for a correspondence to count as a
# RANSAC inlier. Note the unit: cv2.findHomography measures this in the
# DESTINATION space, which here is metres, not pixels. The previous value of
# 3.0 m was loose enough that badly mislocalised landmarks were admitted to
# the consensus set and dragged the fit. 2.0 m is roughly the width of a
# penalty spot marking plus the model's own localisation spread.
HOMOGRAPHY_RANSAC_THRESHOLD_M = 2.0

# A homography has 8 degrees of freedom, so 4 correspondences determine it
# EXACTLY -- with zero residual, always, regardless of whether those four
# points are right. A consensus set of 4 is therefore not evidence at all,
# and 6 leaves only 4 residual degrees of freedom to disagree in. 8 doubles
# the constraints against the parameters.
#
# MEASURED, AND WHAT IT DID NOT FIX (2026-08-16). Raising this from 6 to 8
# and the ratio below from 0.5 to 0.8 reduces the number of frames accepted
# on broadcast footage. It does NOT make the accepted ones correct: fits
# with 8/8 inliers and 0.5 m reprojection error were rendered against the
# painted pitch lines and are still wrong by metres (see the WHAT REMAINS
# BROKEN note in auto_calibration.py). These values are set where they are
# because a smaller consensus is demonstrably not evidence, not because
# they were tuned until the output looked good.
HOMOGRAPHY_MIN_INLIERS = 8

# Fraction of the offered correspondences that must survive RANSAC. A fit
# where a large minority of detected landmarks disagree with the accepted
# model means the model's keypoint IDENTITIES are unreliable on this frame,
# even if the surviving majority is self-consistent -- and identity error is
# invisible to every residual-based measure, so this ratio is the only place
# it shows up numerically at all.
HOMOGRAPHY_MIN_INLIER_RATIO = 0.8

# ---------------------------------------------------------------------------
# Geometric plausibility of a fitted homography.
#
# These catch the failure mode reprojection error structurally CANNOT: a fit
# that is perfectly self-consistent but describes the wrong pitch -- mirrored,
# collapsed, or projecting the visible frame to an area no camera could see.
# ---------------------------------------------------------------------------

# |det(H)| below this is singular to numerical precision: the matrix has
# collapsed the image plane onto a line and is not a projective map at all.
HOMOGRAPHY_MIN_DETERMINANT = 1e-12

# A pixel->pitch homography must preserve orientation. If the image corners
# come back wound the opposite way, the fit has MIRRORED the pitch -- players
# on the left touchline are reported on the right. This is the single most
# damaging silent failure available to this system, and it is invisible to
# reprojection error because a mirrored assignment of landmark identities is
# internally consistent.
HOMOGRAPHY_REJECT_MIRRORED = True

# How much pitch the visible frame may map to, in metres, before the fit is
# implausible. A broadcast camera sees somewhere between a penalty area and
# rather more than the whole pitch; it never sees a 6 m strip (collapsed fit)
# and never sees 900 m (a near-degenerate fit projecting toward the horizon).
# Generous on purpose -- this is a sanity bound, not a tuning knob.
HOMOGRAPHY_MIN_VIEW_SPAN_M = 15.0
HOMOGRAPHY_MAX_VIEW_SPAN_M = 400.0

# How far outside the pitch rectangle the projected frame corners may fall.
# Some overshoot is completely normal (stands, technical area and warm-up
# strips are all visible above the touchline), so this is deliberately wide;
# it exists to catch fits that place the visible image hundreds of metres off
# the pitch, not to police framing.
HOMOGRAPHY_MAX_OUT_OF_BOUNDS_M = 150.0

# ---------------------------------------------------------------------------
# Temporal fallback: reusing a recent good calibration on frames that fail.
#
# WHY THIS IS BOUNDED (the honesty constraint)
#   Carrying the last good homography across a few bad frames is correct --
#   the pitch has not moved and one dropped detection is not new information.
#   Carrying it indefinitely is NOT: after a long gap there is no evidence the
#   camera still sees what it saw, and a stale matrix reported as `valid`
#   would put players at pitch coordinates nothing measured. The fallback
#   therefore expires, and its confidence decays while it is in use so a
#   consumer can see the evidence ageing rather than being told a stale fit is
#   as good as a fresh one.
# ---------------------------------------------------------------------------

# Longest run of consecutive frames a carried calibration may be served for
# before it expires and the frame degrades to valid=False. At 25-30 fps this
# is ~0.7-0.8 s of footage.
CALIBRATION_MAX_FALLBACK_FRAMES = 20

# Multiplicative confidence decay applied per carried frame. Over the full
# CALIBRATION_MAX_FALLBACK_FRAMES span this takes a fit that solved at 0.80
# down to ~0.59, i.e. a calibration that was only marginally above
# HOMOGRAPHY_CONFIDENCE_MIN drops below it and expires EARLY, on its own,
# without waiting for the frame budget. That interaction is intended: weaker
# evidence should survive fewer frames than stronger evidence.
CALIBRATION_FALLBACK_DECAY = 0.985

# Blend weight applied to a newly accepted homography against the one it
# replaces, 0.0 = ignore the new fit entirely, 1.0 = no smoothing. Smoothing
# is applied in PITCH SPACE via projected probe points and refitted (see
# auto_calibration.blend_homographies) -- averaging the matrices entry-wise
# is not a valid operation on projective transforms.
CALIBRATION_SMOOTHING = 0.6

CALIBRATION_POINT_ORDER = [
    "corner_bottom_left", "corner_top_left", "corner_bottom_right", "corner_top_right",
    "halfway_bottom", "halfway_top", "center_spot",
    "left_penalty_area_bottom_near", "left_penalty_area_top_near",
    "left_penalty_area_bottom_far", "left_penalty_area_top_far", "left_penalty_spot",
    "right_penalty_area_bottom_near", "right_penalty_area_top_near",
    "right_penalty_area_bottom_far", "right_penalty_area_top_far", "right_penalty_spot",
]
MIN_CALIBRATION_POINTS = 4

def pressure_level(distance_to_nearest_opponent_m: float | None) -> float | None:
    """Computes pressure level (0.0 to 1.0) based on opponent proximity."""
    if distance_to_nearest_opponent_m is None:
        return None
    return max(0.0, min(1.0, 1.0 - (distance_to_nearest_opponent_m / PRESSURE_RADIUS_M)))
