"""The tracking overlay payload must state whether its pitch coordinates
are trustworthy.

Before Phase 3 this endpoint returned `pitch_x_m`/`pitch_y_m` per player and
nothing about calibration, so a consumer could not distinguish "projected
under a homography that passed the gate" from "None because calibration
failed". On current broadcast footage every one of them is None, and the
frontend rendered that as an ordinary empty overlay.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import models as M
from backend.database.session import Base


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path/'t.db'}",
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    M.Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


def _match_with_tracking(db, *, valid: bool, positioned: bool):
    match = M.Match(home_team="H", away_team="A", video_path="x.mp4", duration=1.0)
    db.add(match)
    db.flush()
    for f in range(5):
        db.add(M.PlayerTracking(
            match_id=match.match_id, player_id=1, frame_id=f, team_id="team-home",
            pixel_x=10.0 + f, pixel_y=20.0,
            pitch_x_m=(30.0 if positioned else None),
            pitch_y_m=(40.0 if positioned else None),
            homography_confidence=(0.8 if positioned else None),
            speed=None, distance=None, acceleration=None,
        ))
    db.add(M.CalibrationStatus(
        match_id=match.match_id, frame_start=0, frame_end=4, valid=valid,
        invalid_reason=None if valid else "no pitch keypoints above the visibility floor",
        confidence=0.8 if valid else 0.0, reprojection_error_m=0.4 if valid else None,
        n_points=8 if valid else 0, source=M.CalibrationSourceKind.model,
        solved_on_frame=0, camera_motion="static", camera_shift_px=0.0,
        homography_matrix=None, method=M.MetricMethod.ml_trained,
    ))
    db.commit()
    return match.match_id


def _get(db, match_id):
    from backend.api.tracking import get_tracking_window

    return get_tracking_window(match_id, start_frame=0, end_frame=10, db=db)


def test_invalid_calibration_is_reported_not_silently_empty(db):
    match_id = _match_with_tracking(db, valid=False, positioned=False)
    payload = _get(db, match_id)

    cal = payload["calibration"]
    assert cal["has_status_rows"] is True
    assert cal["valid_in_window"] is False
    assert cal["valid_frame_ranges"] == []
    assert cal["tracking_rows_in_window"] == 5
    # The point of the field: 5 rows exist, 0 of them are positioned.
    assert cal["rows_with_pitch_coordinates"] == 0
    assert "never imputed" in cal["note"]


def test_valid_calibration_reports_its_frame_ranges(db):
    match_id = _match_with_tracking(db, valid=True, positioned=True)
    cal = _get(db, match_id)["calibration"]

    assert cal["valid_in_window"] is True
    assert cal["valid_frame_ranges"] == [(0, 4)] or cal["valid_frame_ranges"] == [[0, 4]]
    assert cal["rows_with_pitch_coordinates"] == 5


def test_pixel_positions_survive_an_invalid_calibration(db):
    """Pixel tracking does not depend on the homography, so an invalid
    calibration must not blank the overlay itself."""
    match_id = _match_with_tracking(db, valid=False, positioned=False)
    payload = _get(db, match_id)

    assert payload["frames"], "pixel-space frames should still be returned"
    first = payload["frames"][0]["players"][0]
    assert first["pixel_x"] is not None and first["pixel_y"] is not None
    assert first["pitch_x_m"] is None
