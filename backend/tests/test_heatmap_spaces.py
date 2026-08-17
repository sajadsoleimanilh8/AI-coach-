"""The heatmap's two coordinate spaces must stay distinguishable."""
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


def _match_with_pixels_only(db, n: int = 40):
    """The real-world case: tracked boxes, but calibration never solved."""
    match = M.Match(home_team="H", away_team="A", video_path="x.mp4", duration=1.0)
    db.add(match)
    db.flush()
    for f in range(n):
        db.add(M.PlayerTracking(
            match_id=match.match_id, player_id=7, frame_id=f, team_id="team-home",
            pixel_x=100.0 + f, pixel_y=200.0,
            pitch_x_m=None, pitch_y_m=None, homography_confidence=None,
            speed=None, distance=None, acceleration=None,
        ))
    db.add(M.CalibrationStatus(
        match_id=match.match_id, frame_start=0, frame_end=n, valid=False,
        invalid_reason="no pitch keypoints above the visibility floor",
        confidence=0.0, reprojection_error_m=None, n_points=0,
        source=M.CalibrationSourceKind.model, solved_on_frame=None,
        camera_motion="static", camera_shift_px=0.0,
        homography_matrix=None, method=M.MetricMethod.ml_trained,
    ))
    db.commit()
    return match.match_id


def _heatmap(db, match_id, player_id, space, monkeypatch=None):
    from backend.api import tracking

    return tracking.get_player_heatmap(match_id, player_id, space=space, db=db)


def test_pitch_space_is_empty_when_calibration_never_validated(db):
    match_id = _match_with_pixels_only(db)

    result = _heatmap(db, match_id, 7, "pitch")

    assert result["space"] == "pitch"
    assert result["cells"] == []
    assert result["sample_size"] == 40
    assert result["usable_sample_size"] == 0
    assert result["frame_width_px"] is None


def test_image_space_bins_the_pixels_the_same_rows_carry(db, monkeypatch):
    match_id = _match_with_pixels_only(db)
    monkeypatch.setattr("backend.api.tracking._frame_dimensions", lambda *_: (1280, 720))

    result = _heatmap(db, match_id, 7, "image")

    assert result["space"] == "image"
    assert result["cells"], "pixel positions exist, so image space must produce cells"
    assert result["usable_sample_size"] == 40
    assert (result["frame_width_px"], result["frame_height_px"]) == (1280, 720)
    assert result["confidence"] != "low_upstream_confidence"


def test_image_space_is_never_the_default(db):
    match_id = _match_with_pixels_only(db)

    assert _heatmap(db, match_id, 7, "pitch")["space"] == "pitch"


def test_image_space_without_frame_dimensions_returns_empty_not_guessed(db, monkeypatch):
    match_id = _match_with_pixels_only(db)
    monkeypatch.setattr("backend.api.tracking._frame_dimensions", lambda *_: None)

    result = _heatmap(db, match_id, 7, "image")

    assert result["cells"] == []
    assert result["usable_sample_size"] == 0
    assert result["frame_width_px"] is None
