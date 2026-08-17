"""The annotated render, and the colour rule it shares with the frontend."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.pipeline.overlay_video import (
    UNASSIGNED_BGR,
    TEAM_COLOURS_BGR,
    find_overlay_output,
    media_type_for,
    overlay_output_path,
    render_overlay_video,
    team_colour_bgr,
)


class _Det:
    def __init__(self, player_id, x, y, w, h, team_id):
        self.player_id = player_id
        self.x, self.y, self.width, self.height = x, y, w, h
        self.team_id = team_id


class _Ball:
    """Shaped like frame_data.BallObservation, which is what the pipeline
    actually hands the renderer -- pixel_x/pixel_y, NOT x/y. The renderer
    read `.x`/`.y` through getattr with a None default, so the ball marker
    silently never drew. A stand-in with `.x` would have hidden that."""

    def __init__(self, pixel_x, pixel_y):
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y


def test_the_two_teams_get_different_colours():
    home = team_colour_bgr("team-home")
    away = team_colour_bgr("team-away")

    assert home != away
    assert home == TEAM_COLOURS_BGR["team-home"]
    assert away == TEAM_COLOURS_BGR["team-away"]


def test_unknown_and_missing_team_ids_read_as_unassigned():
    assert team_colour_bgr(None) == UNASSIGNED_BGR
    assert team_colour_bgr("referee") == UNASSIGNED_BGR
    assert team_colour_bgr("") == UNASSIGNED_BGR


def test_team_ids_are_matched_case_and_whitespace_insensitively():
    assert team_colour_bgr("  TEAM-HOME ") == TEAM_COLOURS_BGR["team-home"]


def test_output_path_is_beside_the_upload_and_keyed_by_video_id():
    path = overlay_output_path("/data/uploads/clip.mp4", "vid-123")

    assert path.stem == "vid-123_tracked"
    assert path.parent.name == "processed"


def test_find_overlay_output_still_finds_a_legacy_webm_render(tmp_path):
    """A clip processed before the H.264 switch has a .webm on disk. It must
    keep playing rather than being reported as "not rendered" and silently
    re-rendered."""
    uploads = tmp_path / "uploads"
    (uploads / "processed").mkdir(parents=True)
    legacy = uploads / "processed" / "vid-9_tracked.webm"
    legacy.write_bytes(b"not really a video, but non-empty")

    found = find_overlay_output(str(uploads / "clip.mp4"), "vid-9")

    assert found == legacy
    assert media_type_for(found) == "video/webm"


def test_find_overlay_output_ignores_a_zero_byte_stub(tmp_path):
    """An encoder that fails to open leaves a 0-byte file. Reporting that as
    a processed video hands the browser a URL that yields nothing."""
    uploads = tmp_path / "uploads"
    (uploads / "processed").mkdir(parents=True)
    (uploads / "processed" / "vid-9_tracked.mp4").write_bytes(b"")

    assert find_overlay_output(str(uploads / "clip.mp4"), "vid-9") is None


def test_missing_source_video_is_reported_not_raised(tmp_path):
    result = render_overlay_video(
        str(tmp_path / "nope.mp4"), [[]], tmp_path / "out.webm", fps=25.0,
    )

    assert result.path is None
    assert "missing" in (result.skipped_reason or "")


@pytest.mark.parametrize("max_width,expected_width", [(0, 320), (160, 160)])
def test_render_writes_a_playable_file_at_the_configured_width(
    tmp_path, monkeypatch, max_width, expected_width,
):
    cv2 = pytest.importorskip("cv2")

    source = tmp_path / "src.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (320, 240))
    if not writer.isOpened():
        pytest.skip("no encoder available to build the fixture video")
    for _ in range(6):
        writer.write(np.full((240, 320, 3), 60, np.uint8))
    writer.release()

    monkeypatch.setattr("backend.pipeline.overlay_video.OVERLAY_MAX_WIDTH", max_width)

    frames = [[_Det(1, 10, 10, 40, 80, "team-home")] for _ in range(6)]
    out = tmp_path / "out.mp4"
    result = render_overlay_video(str(source), frames, out, fps=25.0)

    assert result.skipped_reason is None, result.skipped_reason
    assert result.frames_written > 0

    written = Path(result.path)
    assert written.exists() and written.stat().st_size > 0
    assert result.codec

    assert result.verified_frames and result.verified_frames > 0
    assert result.width == expected_width

    check = cv2.VideoCapture(str(written))
    try:
        assert check.isOpened()
        assert int(check.get(cv2.CAP_PROP_FRAME_WIDTH)) == expected_width
    finally:
        check.release()


def test_odd_source_dimensions_are_evened_before_encoding(tmp_path, monkeypatch):
    """Both codecs in the chain subsample chroma in 2x2 blocks and reject an
    odd dimension. Evening was previously applied only to the height of a
    DOWNSCALED frame, so an odd native size fell through it entirely when
    OVERLAY_MAX_WIDTH=0 and the writer failed to open."""
    cv2 = pytest.importorskip("cv2")

    source = tmp_path / "src.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (321, 241))
    if not writer.isOpened():
        pytest.skip("no encoder available to build the fixture video")
    for _ in range(4):
        writer.write(np.full((241, 321, 3), 60, np.uint8))
    writer.release()

    monkeypatch.setattr("backend.pipeline.overlay_video.OVERLAY_MAX_WIDTH", 0)
    result = render_overlay_video(str(source), [[] for _ in range(4)],
                                  tmp_path / "out.mp4", fps=25.0)

    assert result.skipped_reason is None, result.skipped_reason
    assert result.width % 2 == 0 and result.height % 2 == 0


def test_the_ball_marker_is_actually_drawn(tmp_path):
    """Regression: the renderer read `ball.x`/`ball.y`, but the objects the
    pipeline passes are BallObservation (pixel_x/pixel_y). getattr's default
    turned that into None on every frame, so no annotated video this project
    ever produced contained a ball marker -- and nothing failed."""
    cv2 = pytest.importorskip("cv2")

    source = tmp_path / "src.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (320, 240))
    if not writer.isOpened():
        pytest.skip("no encoder available to build the fixture video")
    for _ in range(4):
        writer.write(np.zeros((240, 320, 3), np.uint8))
    writer.release()

    import backend.pipeline.overlay_video as overlay_module

    drawn: list[tuple] = []
    original_circle = cv2.circle

    def spy_circle(image, centre, radius, colour, thickness, *args, **kwargs):
        drawn.append((centre, colour))
        return original_circle(image, centre, radius, colour, thickness, *args, **kwargs)

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(cv2, "circle", spy_circle)
        overlay_module.render_overlay_video(
            str(source), [[] for _ in range(4)], tmp_path / "out.mp4", fps=25.0,
            ball_by_frame={0: _Ball(100.0, 50.0), 2: _Ball(110.0, 60.0)},
        )
    finally:
        monkey.undo()

    ball_colours = {colour for _centre, colour in drawn}
    assert ball_colours, "no ball marker was drawn at all"
    assert overlay_module.BALL_BGR in ball_colours
