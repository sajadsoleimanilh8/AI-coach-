"""GET /api/matches -- the endpoint that makes a match id discoverable.

Before this existed the dashboard's four match-scoped tabs (Player, Team,
Calibration, Simulation) could only be used by someone who had uploaded a clip
in the current browser session or who already knew a UUID. These tests pin the
two properties that make the listing safe to choose from:

  - a match with no tracking rows is LISTED, and reports zero, rather than
    being hidden -- someone whose job failed still has to be able to see that
    their upload exists;
  - `video_file_exists` reflects the filesystem, not the row, so the picker
    never advertises a clip that has been deleted from disk.
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


def _list(db, limit: int = 50):
    """Called directly, so `limit` must be a real int.

    FastAPI resolves Query(default=...) only when it dispatches a request; a
    direct call would otherwise hand SQLAlchemy the Query object itself.
    """
    from backend.api.tracking import list_matches

    return list_matches(limit=limit, db=db)


def _add_match(db, *, tracking_rows: int = 0, storage_path: str | None = None,
               job_status: M.ProcessingStatus | None = None):
    match = M.Match(home_team="H", away_team="A", video_path="x.mp4", duration=1.0)
    db.add(match)
    db.flush()

    for f in range(tracking_rows):
        db.add(M.PlayerTracking(
            match_id=match.match_id, player_id=1, frame_id=f, team_id="team-home",
            pixel_x=1.0, pixel_y=2.0, pitch_x_m=None, pitch_y_m=None,
            homography_confidence=None, speed=None, distance=None, acceleration=None,
        ))

    if storage_path is not None:
        video = M.Video(match_id=match.match_id, original_filename="clip.mp4",
                        # videos.stored_filename is UNIQUE -- key it to this
                        # match so a test can create more than one video.
                        stored_filename=f"{match.match_id}.mp4", file_size=1,
                        storage_path=storage_path)
        db.add(video)
        db.flush()
        if job_status is not None:
            db.add(M.ProcessingJob(video_id=video.id, status=job_status))

    db.commit()
    return match.match_id


def test_lists_matches_with_their_real_tracking_counts(db):
    empty = _add_match(db, tracking_rows=0)
    full = _add_match(db, tracking_rows=7)

    rows = {r.match_id: r for r in _list(db)}

    assert set(rows) == {empty, full}
    assert rows[full].tracking_rows == 7
    # Listed, not hidden -- and honest about having nothing.
    assert rows[empty].tracking_rows == 0


def test_video_file_existence_is_read_from_disk(db, tmp_path):
    present = tmp_path / "present.mp4"
    present.write_bytes(b"\x00")

    on_disk = _add_match(db, storage_path=str(present),
                         job_status=M.ProcessingStatus.completed)
    deleted = _add_match(db, storage_path=str(tmp_path / "gone.mp4"),
                         job_status=M.ProcessingStatus.completed)

    rows = {r.match_id: r for r in _list(db)}

    assert rows[on_disk].video_file_exists is True
    # The Video row exists; the file does not. Reporting True here is how the
    # picker would hand the UI a video URL that 404s.
    assert rows[deleted].video_file_exists is False


def test_job_status_is_reported_and_absent_job_is_none(db, tmp_path):
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"\x00")

    with_job = _add_match(db, storage_path=str(clip),
                          job_status=M.ProcessingStatus.failed)
    no_video = _add_match(db)

    rows = {r.match_id: r for r in _list(db)}

    assert rows[with_job].job_status == "failed"
    assert rows[with_job].job_id is not None
    # Never processed is None, not a fabricated "pending".
    assert rows[no_video].job_status is None
    assert rows[no_video].job_id is None


def test_limit_is_honoured(db):
    for _ in range(4):
        _add_match(db)

    assert len(_list(db, limit=2)) == 2


def test_empty_database_returns_empty_list(db):
    assert _list(db) == []
