"""
*** UNFILLED TODO ARTIFACT -- THIS MODULE RAISES NotImplementedError ***
"""

from __future__ import annotations

from ai.computer_vision.tactical_analysis.constants import (
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
)

L = PITCH_LENGTH_M
W = PITCH_WIDTH_M

N_LANDMARK_CLASSES = 28

MAPPING_VERIFIED = False

PITCH_LANDMARK_COORDS: dict[int, tuple[float, float] | None] = {
    idx: None for idx in range(N_LANDMARK_CLASSES)
}


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


def _refuse(detail: str) -> "NotImplementedError":
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
