"""
Shared constants for tactical analysis and scoring functions.
Analysis Logic Design v3 (docs/data_analysis.md §0.4).
"""

PRESSURE_RADIUS_M = 5.0
PRESSURE_THRESHOLD = 0.5
RETENTION_WINDOW_S = 3.0
TOUCH_EVAL_WINDOW_S = 2.0
MAX_TOUCH_DISTANCE_M = 5.0
DECISION_TIME_MIN_S = 0.3
DECISION_TIME_MAX_S = 2.0
MIN_SAMPLE_EVENTS = 5
HOMOGRAPHY_CONFIDENCE_MIN = 0.6
TEAM_ASSIGNMENT_CONFIDENCE_MIN = 0.5
SCHEMA_VERSION = "v3"

BALL_CONTROL_RADIUS_M = 1.5
PASS_MIN_DISTANCE_M = 3.0

PLAYER_HEIGHT_M = 1.80

MIN_SCALE_BOX_HEIGHT_PX = 24.0
SHOT_VELOCITY_MIN_MS = 12.0
MAX_USEFUL_SEPARATION_GAIN_M = 5.0
MAX_ACCEPTABLE_LINE_DEVIATION_M = 8.0
IDEAL_DEFENSIVE_SPACING_M = 10.0
SPACING_TOLERANCE_M = 3.0
REACTION_SPEED_REFERENCE_MS = 3.0

SCAN_ANGLE_THRESHOLD_DEG = 30.0
SCAN_MAX_GAP_S = 2.0
SCANS_PER_MINUTE_TARGET = 8.0

TEAM_ASSIGNMENT_SAMPLE_STRIDE = 5
TEAM_ASSIGNMENT_CROP_TOP_FRACTION = 0.5
TEAM_ASSIGNMENT_MIN_USABLE_PIXELS = 40
TEAM_ASSIGNMENT_KMEANS_MAX_ITERS = 50

GRASS_HUE_RANGE = (35, 85)
GRASS_MIN_SATURATION = 40
GRASS_MIN_VALUE = 40

SKIN_HUE_RANGES = ((0, 20), (170, 179))
SKIN_SATURATION_RANGE = (20, 150)
SKIN_MIN_VALUE = 60

PASS_DIRECTION_SECTORS = 8
SHOT_DISTANCE_TOLERANCE_M = 18.0
MAX_REALISTIC_SHOOTING_ANGLE_DEG = 90.0

IDEAL_SQUARENESS_DEG = 45.0
SQUARENESS_TOLERANCE_DEG = 25.0

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


CAMERA_STATIC_SHIFT_PX = 2.0

CAMERA_CUT_SHIFT_PX = 40.0

CALIBRATION_MAX_JUMP_M = 15.0

HOMOGRAPHY_MIN_POINT_SPREAD = 0.05


HOMOGRAPHY_RANSAC_THRESHOLD_M = 2.0

HOMOGRAPHY_MIN_INLIERS = 8

HOMOGRAPHY_MIN_INLIER_RATIO = 0.8


HOMOGRAPHY_MIN_DETERMINANT = 1e-12

HOMOGRAPHY_REJECT_MIRRORED = True

HOMOGRAPHY_MIN_VIEW_SPAN_M = 15.0
HOMOGRAPHY_MAX_VIEW_SPAN_M = 400.0

HOMOGRAPHY_MAX_OUT_OF_BOUNDS_M = 150.0


CALIBRATION_MAX_FALLBACK_FRAMES = 20

CALIBRATION_FALLBACK_DECAY = 0.985

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
