"""The annotated render, and the colour rule it shares with the frontend.

team_colour_bgr() replaced two independent substring matches -- the minimap's
`includes('a')` and the video overlay's `includes('b')`. Every team id in this
system is "team-home" or "team-away": both contain an "a" (in "team") and
neither contains a "b", so one helper painted both teams the same colour and
so did the other. These tests pin the exact-match rule so that cannot return.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.pipeline.overlay_video import (
    FIELD_BGR,
    GOALPOST_BGR,
    PITCH_LINE_BGR,
    TEAM_COLOURS_BGR,
    UNASSIGNED_BGR,
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


class _Field:
    """Shaped like frame_data.FieldRegion."""

    def __init__(self, polygon, confidence=0.86):
        self.polygon = polygon
        self.confidence = confidence


class _Goalpost:
    """Shaped like frame_data.GoalpostDetection."""

    def __init__(self, x, y, w, h, confidence=0.9):
        self.x, self.y, self.width, self.height = x, y, w, h
        self.confidence = confidence


class _Calib:
    """Shaped like frame_data.CalibrationState, as far as the HUD looks."""

    def __init__(self, valid, confidence=0.71):
        self.valid = valid
        self.confidence = confidence


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

    # The whole point. Substring matching made these equal.
    assert home != away
    assert home == TEAM_COLOURS_BGR["team-home"]
    assert away == TEAM_COLOURS_BGR["team-away"]


def test_unknown_and_missing_team_ids_read_as_unassigned():
    # Not bucketed into a team by how the id happens to be spelled.
    assert team_colour_bgr(None) == UNASSIGNED_BGR
    assert team_colour_bgr("referee") == UNASSIGNED_BGR
    assert team_colour_bgr("") == UNASSIGNED_BGR


def test_team_ids_are_matched_case_and_whitespace_insensitively():
    assert team_colour_bgr("  TEAM-HOME ") == TEAM_COLOURS_BGR["team-home"]


def test_output_path_is_beside_the_upload_and_keyed_by_video_id():
    path = overlay_output_path("/data/uploads/clip.mp4", "vid-123")

    # Suffix follows the codec chain's first rung (H.264/MP4 since the VP8
    # replacement -- see the CODEC note in overlay_video.py). The location
    # and the id-keyed name are the parts that must not drift.
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

    # A render failure must never fail a run whose metrics are already
    # computed and committed -- see render_overlay_video's docstring.
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

    # The written path is whichever codec actually opened, so it is read off
    # the result rather than assumed from what was requested.
    written = Path(result.path)
    assert written.exists() and written.stat().st_size > 0
    assert result.codec

    # The artifact check that the render itself now performs: the file must
    # decode back, not merely exist. A writer that accepted every frame and
    # produced an undecodable container used to pass as a successful render.
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

    # Patch on the cv2 module the renderer imports locally.
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


def _fixture_video(cv2, path, n_frames=4, size=(320, 240)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, size)
    if not writer.isOpened():
        pytest.skip("no encoder available to build the fixture video")
    for _ in range(n_frames):
        writer.write(np.zeros((size[1], size[0], 3), np.uint8))
    writer.release()


def _colours_drawn(cv2, monkey, function_name, image_arg=0):
    """Record the colour argument of every cv2 draw call of one kind."""
    seen: list = []
    original = getattr(cv2, function_name)

    def spy(*args, **kwargs):
        seen.append(args[3] if len(args) > 3 else kwargs.get("color"))
        return original(*args, **kwargs)

    monkey.setattr(cv2, function_name, spy)
    return seen


def test_field_polygon_and_goalposts_are_drawn(tmp_path):
    """The field and goalpost models fired on every frame of the broadcast clip
    and NONE of it reached the render, which is what "only player detection
    works" actually meant."""
    cv2 = pytest.importorskip("cv2")
    source = tmp_path / "src.mp4"
    _fixture_video(cv2, source)

    monkey = pytest.MonkeyPatch()
    try:
        polylines = _colours_drawn(cv2, monkey, "polylines")
        rectangles = _colours_drawn(cv2, monkey, "rectangle")
        result = render_overlay_video(
            str(source), [[] for _ in range(4)], tmp_path / "out.mp4", fps=25.0,
            field_by_frame={1: _Field([(10, 10), (300, 20), (290, 220), (20, 210)])},
            goalposts_by_frame={1: [_Goalpost(40, 40, 60, 50)]},
        )
    finally:
        monkey.undo()

    assert result.skipped_reason is None, result.skipped_reason
    assert FIELD_BGR in polylines
    assert GOALPOST_BGR in rectangles


def test_a_valid_homography_reprojects_the_pitch_outline(tmp_path):
    """The visual proof that calibration produced a USABLE H, as opposed to a
    `calib: valid` label that says nothing about whether the matrix is any good."""
    cv2 = pytest.importorskip("cv2")
    source = tmp_path / "src.mp4"
    _fixture_video(cv2, source)

    pixel = np.float32([[20, 220], [300, 220], [260, 40], [60, 40]])
    pitch = np.float32([[0, 68], [105, 68], [105, 0], [0, 0]])
    H, _ = cv2.findHomography(pixel, pitch, method=0)

    monkey = pytest.MonkeyPatch()
    try:
        polylines = _colours_drawn(cv2, monkey, "polylines")
        result = render_overlay_video(
            str(source), [[] for _ in range(4)], tmp_path / "out.mp4", fps=25.0,
            homography_by_frame={1: H},
        )
    finally:
        monkey.undo()

    assert result.skipped_reason is None, result.skipped_reason
    assert PITCH_LINE_BGR in polylines


@pytest.mark.parametrize("bad_homography", [
    np.zeros((3, 3)),                      # singular
    np.full((3, 3), np.nan),               # not finite
    np.eye(2),                             # wrong shape
    "not a matrix",
])
def test_a_degenerate_homography_does_not_abort_the_render(tmp_path, bad_homography):
    """A single unusable frame must cost that frame's outline, not the whole
    video -- this module reports failures, it does not raise them."""
    cv2 = pytest.importorskip("cv2")
    source = tmp_path / "src.mp4"
    _fixture_video(cv2, source)

    result = render_overlay_video(
        str(source), [[] for _ in range(4)], tmp_path / "out.mp4", fps=25.0,
        homography_by_frame={1: bad_homography},
    )

    assert result.skipped_reason is None, result.skipped_reason
    assert result.frames_written == 4


def test_a_degenerate_field_polygon_does_not_abort_the_render(tmp_path):
    cv2 = pytest.importorskip("cv2")
    source = tmp_path / "src.mp4"
    _fixture_video(cv2, source)

    result = render_overlay_video(
        str(source), [[] for _ in range(4)], tmp_path / "out.mp4", fps=25.0,
        field_by_frame={0: _Field([]), 1: _Field([(1, 2)]), 2: _Field(None)},
        goalposts_by_frame={1: [_Goalpost(10, 10, 0, 0)]},
    )

    assert result.skipped_reason is None, result.skipped_reason
    assert result.frames_written == 4


def test_the_hud_names_every_model_and_marks_the_silent_ones():
    cv2 = pytest.importorskip("cv2")
    import backend.pipeline.overlay_video as overlay_module

    texts: list[str] = []
    monkey = pytest.MonkeyPatch()
    original = cv2.putText

    def spy(image, text, *args, **kwargs):
        texts.append(text)
        return original(image, text, *args, **kwargs)

    image = np.zeros((240, 320, 3), np.uint8)
    try:
        monkey.setattr(cv2, "putText", spy)
        overlay_module._draw_hud(
            cv2, image, 12, 0.48, [None] * 14, _Calib(True, 0.71),
            ball=_Ball(10.0, 20.0), field_region=_Field([(0, 0)], 0.86),
            goalposts=[_Goalpost(1, 1, 2, 2)] * 2,
        )
        overlay_module._draw_hud(cv2, image, 13, 0.52, [], False)
    finally:
        monkey.undo()

    assert "players:14" in texts[0]
    assert "field:Y(0.86)" in texts[0]
    assert "gp:2" in texts[0]
    assert "calib:valid(0.71)" in texts[0]

    assert "ball:N" in texts[1] and "field:N" in texts[1]
    assert "gp:-" in texts[1] and "calib:none" in texts[1]


def test_the_hud_still_accepts_the_legacy_boolean_calibration_map():
    """Callers that pass `{frame: bool}` predate the CalibrationState upgrade
    and must keep rendering the same valid/none label."""
    pytest.importorskip("cv2")
    import backend.pipeline.overlay_video as overlay_module

    assert overlay_module._calibration_flag(True) == "valid"
    assert overlay_module._calibration_flag(False) == "none"
    assert overlay_module._calibration_flag(None) == "n/a"
