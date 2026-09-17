"""
*** UNFILLED TODO ARTIFACT -- THIS MODULE RAISES NotImplementedError ***

    Class-id -> pitch-metre table for the 28-class field-landmark DETECTION
    dataset (datasets/field_landmarks/field_yolo_v2). Every entry in
    PITCH_LANDMARK_COORDS is currently None. Nothing in the pipeline may
    consume this module until a human fills it in; the only sanctioned
    reader, landmarks_to_correspondences(), raises NotImplementedError
    while any required coordinate is still None.

WHY IT IS EMPTY RATHER THAN GUESSED
    Guessing this mapping would be silently catastrophic. A wrong class ->
    landmark assignment still produces a perfectly well-conditioned
    homography -- it just maps players to the wrong pitch locations.
    Reprojection error is computed against the assumed pitch coordinates,
    so a consistently-wrong table reports a LOW error and HIGH confidence.
    Every downstream gate in this codebase (HOMOGRAPHY_CONFIDENCE_MIN, the
    field-region check, the spread check) would pass. There is no automatic
    check that catches it. That is why this file refuses to run instead of
    shipping plausible defaults.

    The same reasoning is written up at greater length in pitch_keypoints.py
    ("HOW THIS MAPPING WAS ESTABLISHED (not guessed)"), which is the worked
    example of doing this correctly.

READ THIS BEFORE FILLING IT IN -- YOU MAY NOT NEED THIS FILE AT ALL
    The repo ALREADY has a verified landmark -> metre table:
    pitch_keypoints.PITCH_KEYPOINTS_32, covering 32 indexed pitch keypoints,
    empirically derived and verified to a median 0.247 m reprojection error
    across 255 training images (`python -m scripts.validate_pitch_keypoints`).
    It is consumed by auto_calibration.py and is the live calibration path.

    That table serves a POSE dataset (configs/datasets.yaml `calibration`,
    on disk at datasets/processed/field_datasets/2 -- see the NAMING WARNING
    in configs/datasets.yaml). This file would serve a different, 28-class
    DETECTION dataset with a different and unknown class ordering. The two
    are NOT interchangeable and 28 != 32.

    So the first decision is not "what are these 28 coordinates" but "is a
    second, weaker landmark scheme worth maintaining alongside a verified
    one". If the answer is no, delete this file and the field_yolo_v2
    dataset rather than half-filling it.

HOW TO FILL IT IN (if the answer is yes)
    Do NOT read class ids off a pitch diagram and assign by eye. Follow the
    procedure that produced pitch_keypoints.py:

    1. Render class ids onto several labelled frames from
       datasets/field_landmarks/field_yolo_v2 (use the label box centres).
    2. Identify 4+ UNAMBIGUOUS anchors visible in one frame -- corner flags
       and the two ends of the halfway line are the usual choices, because
       they cannot be confused with any other landmark.
    3. Fit a homography from those anchors alone, project every other
       detected landmark into pitch metres, and take the MEDIAN across many
       frames. Real landmarks cluster tightly on canonical pitch features;
       a mis-assignment scatters.
    4. Fill the table from those clusters, then VERIFY: refit with RANSAC
       using the full table across the whole train split and check the
       median reprojection error is sub-metre. Cross-clip agreement at that
       precision cannot happen by accident.
    5. Record the measured error in this docstring, the way
       pitch_keypoints.py records its 0.247 m.

    A partially-filled table is legitimate: landmarks_to_correspondences()
    drops unmapped ids rather than defaulting them, exactly as
    keypoints_to_correspondences() does. See `strict` below for the one
    case where partial filling should still be a hard error.

COORDINATE FRAME (must match the rest of the codebase)
    FIFA standard pitch, metres. x in [0, 105] runs goal line to goal line;
    y in [0, 68] runs touchline to touchline. ORIGIN (0, 0) is the corner
    where the LEFT goal line meets the y = 0 touchline -- the touchline
    FURTHEST from a standard main camera.

    This is the identical frame used by constants.REFERENCE_POINTS and
    pitch_keypoints.PITCH_KEYPOINTS_32 (constants.PITCH_LENGTH_M = 105.0,
    constants.PITCH_WIDTH_M = 68.0). Do not introduce a second convention;
    a table in a different frame is a wrong table.
"""

from __future__ import annotations

from ai.computer_vision.tactical_analysis.constants import (
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
)

L = PITCH_LENGTH_M  # 105.0
W = PITCH_WIDTH_M   # 68.0

#: Number of classes in datasets/field_landmarks/field_yolo_v2/data.yaml.
#: Kept here so a mismatch between the two is detectable rather than
#: discovered as an IndexError during training.
N_LANDMARK_CLASSES = 28

#: False until every entry in PITCH_LANDMARK_COORDS is a real coordinate.
#: Flip it to True ONLY after step 4 of the procedure above (the sub-metre
#: reprojection verification) has actually been run and its result recorded
#: in the module docstring. It is not a checkbox for "I typed in numbers".
MAPPING_VERIFIED = False

#: class id -> (x_m, y_m) in the coordinate frame documented above, or None
#: for "not yet determined". None is a first-class value here: it means
#: this landmark is unusable for calibration, NOT that it sits at (0, 0).
#: Never substitute a placeholder coordinate for a None.
PITCH_LANDMARK_COORDS: dict[int, tuple[float, float] | None] = dict.fromkeys(range(N_LANDMARK_CLASSES))


def unmapped_ids() -> list[int]:
    """Class ids still awaiting a real coordinate. Empty == table complete."""
    return [idx for idx in sorted(PITCH_LANDMARK_COORDS)
            if PITCH_LANDMARK_COORDS[idx] is None]


def is_ready() -> bool:
    """
    True only when the table is complete AND its verification has been
    recorded. Both halves matter: a complete-but-unverified table is
    precisely the silent-failure mode described at the top of this file.
    """
    return MAPPING_VERIFIED and not unmapped_ids()


def _refuse(detail: str) -> NotImplementedError:
    return NotImplementedError(
        "pitch_landmarks.PITCH_LANDMARK_COORDS is not filled in -- "
        f"{detail}\n"
        "This module is a deliberate TODO artifact. Using it with None "
        "coordinates would produce a confident, low-error homography that "
        "maps players to the wrong pitch positions, and no downstream gate "
        "would catch it.\n"
        "Either fill in the table (procedure in this module's docstring) "
        "or use the already-verified 32-keypoint path in "
        "pitch_keypoints.PITCH_KEYPOINTS_32 instead."
    )


def landmarks_to_correspondences(
    detections: dict[int, tuple[float, float]],
    min_confidence: float = 0.0,
    confidences: dict[int, float] | None = None,
    strict: bool = True,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[int]]:
    """
    The ONLY sanctioned reader of PITCH_LANDMARK_COORDS.

    Mirrors pitch_keypoints.keypoints_to_correspondences() so that, once the
    table is filled, this drops into homography.compute_homography() the
    same way the 32-keypoint path already does.

    Args:
        detections: {class_id: (pixel_x, pixel_y)} from the landmark
            detector -- use each box's centre.
        min_confidence: detections below this are dropped, not defaulted.
        confidences: {class_id: detector confidence}, optional.
        strict: when True (the default) an unfilled table is a hard error.
            Pass False ONLY from a tool that is itself deriving the table
            (step 3 of the procedure above), where working with a partial
            mapping is the entire point.

    Returns:
        (pixel_pts, pitch_pts, used_class_ids), ready for
        compute_homography().

    Raises:
        NotImplementedError: whenever `strict` and the table is not both
            complete and verified.
    """
    if strict and not is_ready():
        missing = unmapped_ids()
        if missing:
            raise _refuse(f"{len(missing)} of {N_LANDMARK_CLASSES} class ids "
                          f"have no coordinate: {missing}")
        raise _refuse("the table is complete but MAPPING_VERIFIED is still "
                      "False, so its correctness has never been measured")

    pixel_pts: list[tuple[float, float]] = []
    pitch_pts: list[tuple[float, float]] = []
    used: list[int] = []
    for class_id, (px, py) in sorted(detections.items()):
        coord = PITCH_LANDMARK_COORDS.get(class_id)
        # An unmapped landmark is dropped rather than defaulted: an
        # unreliable correspondence biases the fit while looking like extra
        # evidence (same rule as keypoints_to_correspondences()).
        if coord is None:
            continue
        if confidences is not None and confidences.get(class_id, 0.0) < min_confidence:
            continue
        pixel_pts.append((float(px), float(py)))
        pitch_pts.append(coord)
        used.append(class_id)
    return pixel_pts, pitch_pts, used


def verify_table() -> list[str]:
    """
    Self-check, mirroring pitch_keypoints.verify_table(). Returns a list of
    problems (empty when the table is consistent and complete).

    Deliberately reports the unfilled state as a problem -- this module is
    not "fine" until someone does the work.
    """
    problems: list[str] = []

    missing_ids = set(range(N_LANDMARK_CLASSES)) - set(PITCH_LANDMARK_COORDS)
    if missing_ids:
        problems.append(f"class ids absent from the table: {sorted(missing_ids)}")

    extra_ids = set(PITCH_LANDMARK_COORDS) - set(range(N_LANDMARK_CLASSES))
    if extra_ids:
        problems.append(
            f"class ids beyond nc={N_LANDMARK_CLASSES}: {sorted(extra_ids)}")

    unmapped = unmapped_ids()
    if unmapped:
        problems.append(
            f"{len(unmapped)} class ids still have coordinate None: {unmapped}")

    for idx, coord in sorted(PITCH_LANDMARK_COORDS.items()):
        if coord is None:
            continue
        x, y = coord
        if not (0.0 <= x <= L and 0.0 <= y <= W):
            problems.append(
                f"class {idx} coordinate ({x:.2f}, {y:.2f}) is outside the "
                f"{L}x{W} m pitch -- wrong coordinate frame?")

    if not unmapped and not MAPPING_VERIFIED:
        problems.append(
            "table is complete but MAPPING_VERIFIED is False -- run the "
            "reprojection verification (step 4) and record the result")

    return problems


if __name__ == "__main__":
    for problem in verify_table():
        print(f"  - {problem}")
    print(f"\nis_ready() = {is_ready()}")
