"""
Burns the pipeline's own tracking output into a playable video.

WHY THIS EXISTS
The dashboard's player served the RAW uploaded clip and drew tracking boxes
over it as an SVG layer, rebuilt per 150-frame window. That works, but it is
not the processed video -- it is the original video with a separate overlay
that only exists inside this one React component. There was no artifact
anywhere on disk that showed what the pipeline actually saw, so "watch the
processed clip" was not a thing anyone could do, and nothing could be shared,
downloaded, or checked outside the browser.

This renders that artifact: one frame per source frame, each carrying the
boxes, track ids and team colours that ByteTrack and jersey clustering
actually produced for it, plus a HUD strip naming the frame, the timestamp
and the calibration state that frame was analysed under.

WHAT IT DRAWS, AND WHAT IT REFUSES TO DRAW
Only detections that exist. A frame with no detections is written through
unmodified rather than carrying forward the previous frame's boxes -- a
"sticky" box would show a player the detector did not find, which is exactly
the kind of confident-looking fabrication the rest of this pipeline is built
to avoid. Unassigned players are drawn in neutral grey and labelled without a
team, never defaulted into one. The HUD reports calibration as the pipeline
measured it for that frame, including "no calibration", rather than omitting
the row when the answer is unflattering.

CODEC -- MEASURED 2026-08-16, AND THIS REPLACED VP8
    This stage used to hardcode VP8/WebM. On the last full production run it
    cost 555.93s of a 1179.63s job: 47% of the entire pipeline, more than
    player detection, ball detection, calibration and pose estimation put
    together, to produce a 568 MB file for a 9.7-minute clip.

    Re-measured on this machine (375 frames of samples/sample_15s.mp4 at
    960x540, encode only, same frames, same writer API):

        avc1 (H.264)    0.38s     988 enc-fps    24.35 MB   PSNR 39-43 dB
        VP80 (VP8)     15.95s      23 enc-fps    13.07 MB   PSNR 37-40 dB
        VP90 (VP9)     79.25s       5 enc-fps    16.05 MB
        mp4v           0.56s      670 enc-fps     5.03 MB   (see below)

    H.264 is ~40x faster than the VP8 it replaces and decodes at HIGHER
    fidelity. The output was verified to be genuine H.264, not a silent
    fallback: the written file carries an `avcC` box (SPS/PPS) and re-decodes
    to the full frame count.

    The previous docstring claimed H.264 "opens but contains no decodable
    video" here because libopenh264 is absent. libopenh264 IS absent -- the
    loader logs a failure for it -- but this OpenCV's FFmpeg then falls back
    to a working H.264 encoder and produces a valid file. That claim was
    tested and is false on this build, which is why the codec changed.

    mp4v is NOT in the fallback chain despite being fast and small: it is
    MPEG-4 Part 2, which browsers reject with MEDIA_ERR_SRC_NOT_SUPPORTED.
    A file that encodes but will not play is worse than no file, because it
    fails in the browser rather than here where it can be reported.

    VP8 is kept as the fallback rung so a build without a usable H.264
    encoder still produces something a browser can play. Set OVERLAY_CODEC
    to force one (e.g. OVERLAY_CODEC=VP80).

CONTAINER SEEKING
    OpenCV writes the moov atom at the END of the mp4 (there is no
    faststart option through this API). Browsers handle that by issuing a
    range request for the tail before playing, which works because the
    endpoint serving this file supports HTTP range requests -- see
    backend/api/main.py::get_processed_video_file. If that endpoint ever
    stops honouring Range, seeking in the annotated video breaks here.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai.computer_vision.tactical_analysis.constants import (
    PENALTY_AREA_DEPTH_M,
    PENALTY_AREA_WIDTH_M,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
)

logger = logging.getLogger(__name__)

# BGR, because that is the order cv2 draws in. These are the same three
# colours the dashboard's SVG overlay uses, so the burned-in video and the
# live overlay cannot disagree about which team a player is on:
#   home       #00d4ff  neon blue
#   away       #f472b6  neon pink
#   unassigned #64748b  slate
TEAM_COLOURS_BGR = {
    "team-home": (255, 212, 0),
    "team-away": (182, 114, 244),
}
UNASSIGNED_BGR = (139, 116, 100)
BALL_BGR = (0, 255, 255)
FIELD_BGR = (0, 200, 0)
GOALPOST_BGR = (0, 0, 255)
PITCH_LINE_BGR = (0, 215, 255)
HUD_TEXT_BGR = (235, 235, 240)
HUD_PANEL_BGR = (24, 18, 16)

FIELD_FILL_ALPHA = 0.18

_PITCH_CY = PITCH_WIDTH_M / 2.0
_PENALTY_HALF = PENALTY_AREA_WIDTH_M / 2.0

#: Pitch markings in METRES, as polylines. Reprojected through inv(H) they are
#: the visual proof that the calibration model produced a usable homography:
#: if these land on the real touchlines, H is right.
PITCH_OUTLINE_M: tuple[tuple[tuple[float, float], ...], ...] = (
    # touchlines + goal lines
    ((0.0, 0.0), (PITCH_LENGTH_M, 0.0), (PITCH_LENGTH_M, PITCH_WIDTH_M),
     (0.0, PITCH_WIDTH_M), (0.0, 0.0)),
    # halfway line
    ((PITCH_LENGTH_M / 2.0, 0.0), (PITCH_LENGTH_M / 2.0, PITCH_WIDTH_M)),
    # left penalty area
    ((0.0, _PITCH_CY - _PENALTY_HALF), (PENALTY_AREA_DEPTH_M, _PITCH_CY - _PENALTY_HALF),
     (PENALTY_AREA_DEPTH_M, _PITCH_CY + _PENALTY_HALF), (0.0, _PITCH_CY + _PENALTY_HALF)),
    # right penalty area
    ((PITCH_LENGTH_M, _PITCH_CY - _PENALTY_HALF),
     (PITCH_LENGTH_M - PENALTY_AREA_DEPTH_M, _PITCH_CY - _PENALTY_HALF),
     (PITCH_LENGTH_M - PENALTY_AREA_DEPTH_M, _PITCH_CY + _PENALTY_HALF),
     (PITCH_LENGTH_M, _PITCH_CY + _PENALTY_HALF)),
)

#: A reprojected point further than this many pixels outside the frame is a
#: sign of a near-degenerate homography; drawing it would smear the whole image.
_MAX_REPROJECTED_PX = 100_000.0

# Rendering is a presentation stage, not a measurement one, so it is allowed
# to be skipped without affecting a single stored number.
RENDER_OVERLAY_VIDEO = os.getenv("RENDER_OVERLAY_VIDEO", "1") not in ("0", "false", "False")

# Output is capped at this width. Encoding cost and file size both scale with
# pixel count, and 960 keeps a player box and its track-id plate legible.
# This is a review artifact, not a master -- the source clip is still on disk
# and still served by /file, so nothing is lost by not re-encoding it at full
# resolution. Set OVERLAY_MAX_WIDTH=0 to render at native size.
OVERLAY_MAX_WIDTH = int(os.getenv("OVERLAY_MAX_WIDTH", "960"))

# (fourcc, container suffix, MIME type), tried in order. See the CODEC note
# in this module's docstring for why this order and why mp4v is not in it.
OVERLAY_CODEC_CHAIN: tuple[tuple[str, str, str], ...] = (
    ("avc1", ".mp4", "video/mp4"),
    ("VP80", ".webm", "video/webm"),
)

# Every suffix this module has ever written, newest first. Used to FIND an
# existing render rather than to choose one to write: a video processed
# before the codec change has a .webm on disk and must keep playing.
OVERLAY_SUFFIXES: tuple[str, ...] = (".mp4", ".webm")

MIME_BY_SUFFIX = {".mp4": "video/mp4", ".webm": "video/webm"}

_FORCED_CODEC = os.getenv("OVERLAY_CODEC", "").strip()


def _codec_chain() -> tuple[tuple[str, str, str], ...]:
    """The codecs to try, honouring OVERLAY_CODEC when it names a known one."""
    if not _FORCED_CODEC:
        return OVERLAY_CODEC_CHAIN
    forced = [c for c in OVERLAY_CODEC_CHAIN if c[0].lower() == _FORCED_CODEC.lower()]
    if forced:
        return tuple(forced)
    logger.warning(
        "OVERLAY_CODEC=%s is not one of %s -- ignoring it and using the default chain",
        _FORCED_CODEC, [c[0] for c in OVERLAY_CODEC_CHAIN],
    )
    return OVERLAY_CODEC_CHAIN


@dataclass
class OverlayRenderResult:
    path: str | None
    frames_written: int
    skipped_reason: str | None = None
    codec: str | None = None
    # Filled from a re-open of the written file, not from what was intended:
    # "I wrote 14502 frames" and "this file contains 14502 decodable frames"
    # are different claims, and only the second one is what a browser will
    # experience. See _verify_output().
    verified_frames: int | None = None
    width: int | None = None
    height: int | None = None
    size_bytes: int | None = None
    fps: float | None = None

    @property
    def ok(self) -> bool:
        return self.path is not None and self.skipped_reason is None

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "codec": self.codec,
            "frames_written": self.frames_written,
            "verified_frames": self.verified_frames,
            "width": self.width,
            "height": self.height,
            "size_bytes": self.size_bytes,
            "fps": self.fps,
            "skipped_reason": self.skipped_reason,
        }


def overlay_output_path(storage_path: str, video_id: str,
                        suffix: str | None = None) -> Path:
    """Where the annotated render for this video is WRITTEN.

    Derived by convention next to the upload rather than stored in a new
    column: this is a regenerable artifact, and adding a nullable path column
    would mean a migration plus a second source of truth about a file whose
    real state is "is it on disk".

    `suffix` defaults to the first codec in the chain. Callers that want to
    know whether a render exists must use find_overlay_output() instead --
    the suffix depends on which codec actually opened at render time, and on
    whether the run predates the H.264 switch.
    """
    resolved = suffix or _codec_chain()[0][1]
    return Path(storage_path).parent / "processed" / f"{video_id}_tracked{resolved}"


def find_overlay_output(storage_path: str, video_id: str) -> Path | None:
    """The annotated render for this video that is actually ON DISK, or None.

    Checks every suffix this module has ever written, newest format first, so
    a clip rendered as VP8/WebM before the codec change keeps playing without
    being re-rendered. Existence on disk is the single source of truth about
    whether a processed video exists -- there is no DB column claiming it.
    """
    for candidate_suffix in OVERLAY_SUFFIXES:
        candidate = overlay_output_path(storage_path, video_id, candidate_suffix)
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def media_type_for(path: Path | str) -> str:
    """MIME type for a rendered overlay, from its suffix."""
    return MIME_BY_SUFFIX.get(Path(path).suffix.lower(), "application/octet-stream")


def team_colour_bgr(team_id: str | None) -> tuple[int, int, int]:
    """Colour for a team id, neutral for anything unrecognised.

    Matched exactly, never by substring. The frontend's equivalent helper
    tested `String(team_id).includes('a')` -- and BOTH "team-home" and
    "team-away" contain an "a", so every player on both teams was drawn in
    the same colour and the overlay's team distinction was decorative. An
    unknown id here reads as unassigned, which is true, instead of being
    silently bucketed into whichever branch its spelling happened to hit.
    """
    if team_id is None:
        return UNASSIGNED_BGR
    return TEAM_COLOURS_BGR.get(str(team_id).strip().lower(), UNASSIGNED_BGR)


def _even(value: int) -> int:
    """Nearest even size >= 2.

    Both encoders in the chain subsample chroma in 2x2 blocks and reject (or
    silently pad) an odd dimension, so callers apply this to BOTH dimensions,
    for native-size frames (OVERLAY_MAX_WIDTH=0) as well as downscaled ones.
    """
    return max(2, int(value) // 2 * 2)


def _open_writer(cv2, out_path: Path, fps: float, size: tuple[int, int]):
    """First codec in the chain that actually opens. Returns (writer, path, codec).

    Tries each rung and, critically, checks isOpened() rather than trusting
    the constructor: cv2.VideoWriter never raises, it returns a closed writer
    whose write() calls are silent no-ops. That is the exact failure shape
    that produces a 0-byte file and a job that still says "completed".
    """
    attempts: list[str] = []
    for fourcc, suffix, _mime in _codec_chain():
        candidate = out_path.with_suffix(suffix)
        writer = cv2.VideoWriter(
            str(candidate), cv2.VideoWriter_fourcc(*fourcc), fps, size,
        )
        if writer.isOpened():
            if attempts:
                logger.warning("overlay codec %s unavailable (%s); using %s",
                               attempts[0], ", ".join(attempts), fourcc)
            return writer, candidate, fourcc
        writer.release()
        # A rung that failed leaves a 0-byte stub behind. Remove it, or
        # find_overlay_output() would later report a "render" that is an
        # empty file.
        try:
            if candidate.exists() and candidate.stat().st_size == 0:
                candidate.unlink()
        except OSError:
            pass
        attempts.append(fourcc)
    return None, None, None


def _verify_output(cv2, path: Path, expected_frames: int) -> tuple[int, int, int, float]:
    """Re-open the written file and report what is REALLY in it.

    An encoder that opened, accepted every write and produced a file is still
    not proof of a playable video -- this reads the container back and counts
    decodable frames. Returns (frames, width, height, fps).
    """
    check = cv2.VideoCapture(str(path))
    try:
        if not check.isOpened():
            return 0, 0, 0, 0.0
        width = int(check.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(check.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(check.get(cv2.CAP_PROP_FPS) or 0.0)
        # CAP_PROP_FRAME_COUNT is a container metadata read, not a decode --
        # trusted only when it agrees with what was written. When it does
        # not, decode far enough to tell whether the file is genuinely short
        # or the header is merely unreliable (common for streamed mp4).
        reported = int(check.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if reported >= expected_frames > 0:
            return reported, width, height, fps
        decoded = 0
        while True:
            ok, _ = check.read()
            if not ok:
                break
            decoded += 1
        return decoded, width, height, fps
    finally:
        check.release()


def render_overlay_video(
    video_path: str,
    frames,
    out_path: Path,
    *,
    fps: float,
    ball_by_frame: dict | None = None,
    calibration_by_frame: dict | None = None,
    field_by_frame: dict | None = None,
    goalposts_by_frame: dict | None = None,
    homography_by_frame: dict | None = None,
    progress=None,
) -> OverlayRenderResult:
    """
    Draw `frames` (list[list[TrackedDetection]], indexed by frame number)
    onto the source video and write an annotated clip beside it.

    Args:
        ball_by_frame: {frame_number: BallObservation}. Only DETECTED
            observations should be passed -- an interpolated position drawn
            in the same marker as a measured one would be indistinguishable
            from evidence the ball was seen there.
        calibration_by_frame: {frame_number: bool} -- whether that frame's
            calibration was valid. Drives the HUD's calibration chip. Absent
            means the HUD reports "n/a" rather than implying either answer.

    Never raises into the pipeline. A failed render costs a video file, not a
    run: every metric this job produced is already computed and persisted by
    the time this is called, so an encoder problem must not turn a successful
    analysis into a failed job. It is REPORTED rather than swallowed -- the
    returned result carries the reason, run_pipeline() persists it, and the
    API serves it, so "no processed video" is never a silent outcome.
    """
    if not RENDER_OVERLAY_VIDEO:
        return OverlayRenderResult(None, 0, "RENDER_OVERLAY_VIDEO=0")
    if not os.path.exists(video_path):
        return OverlayRenderResult(None, 0, f"source video missing: {video_path}")

    try:
        import cv2
    except ImportError:
        return OverlayRenderResult(None, 0, "cv2 not available")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return OverlayRenderResult(None, 0, "could not open source video")

    writer = None
    written = 0
    written_path: Path | None = None
    codec: str | None = None
    ball_by_frame = ball_by_frame or {}
    calibration_by_frame = calibration_by_frame or {}
    field_by_frame = field_by_frame or {}
    goalposts_by_frame = goalposts_by_frame or {}
    homography_by_frame = homography_by_frame or {}
    scale = 1.0
    out_size = None
    out_fps = float(fps) if fps and fps > 0 else 25.0

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        total = len(frames)
        report_every = max(1, total // 20) if total else 1

        for frame_number in range(total):
            ok, image = cap.read()
            if not ok or image is None:
                break

            if writer is None:
                height, width = image.shape[:2]
                if 0 < OVERLAY_MAX_WIDTH < width:
                    scale = OVERLAY_MAX_WIDTH / width
                    out_size = (_even(OVERLAY_MAX_WIDTH), _even(round(height * scale)))
                else:
                    out_size = (_even(width), _even(height))

                writer, written_path, codec = _open_writer(cv2, out_path, out_fps, out_size)
                if writer is None:
                    return OverlayRenderResult(
                        None, 0,
                        "no usable video encoder: none of "
                        f"{[c[0] for c in _codec_chain()]} could open a writer at "
                        f"{out_size[0]}x{out_size[1]} @ {out_fps:.2f}fps",
                    )

            # Resize FIRST, then draw at the output scale. Drawing at source
            # size and shrinking afterwards would thin every box line and
            # every glyph by the same factor, which is what makes a
            # downscaled annotated video unreadable.
            if out_size != (image.shape[1], image.shape[0]):
                image = cv2.resize(image, out_size, interpolation=cv2.INTER_AREA)

            field_region = field_by_frame.get(frame_number)
            goalposts = goalposts_by_frame.get(frame_number)
            ball = ball_by_frame.get(frame_number)
            calibration = calibration_by_frame.get(frame_number)

            _draw_frame(cv2, image, frames[frame_number], ball, scale,
                        field_region=field_region, goalposts=goalposts,
                        homography=homography_by_frame.get(frame_number))
            _draw_hud(cv2, image, frame_number, frame_number / out_fps,
                      frames[frame_number], calibration,
                      ball=ball, field_region=field_region, goalposts=goalposts)
            writer.write(image)
            written += 1

            if progress and (written % report_every == 0 or written == total):
                progress(written, total)

        # See this function's docstring: a render failure is not a run failure.
    except Exception as error:  # noqa: BLE001 - overlay is optional output; report failure, keep the job
        logger.warning("overlay render failed after %d frames: %s", written, error)
        return OverlayRenderResult(None, written, str(error), codec=codec)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if written == 0 or written_path is None:
        return OverlayRenderResult(None, 0, "no frames decoded", codec=codec)

    # The artifact check. Everything above proves the encoder accepted the
    # frames; only this proves a decoder can get them back out.
    size_bytes = written_path.stat().st_size if written_path.exists() else 0
    if size_bytes == 0:
        return OverlayRenderResult(
            None, written,
            f"encoder {codec} produced a 0-byte file at {written_path}",
            codec=codec, size_bytes=0,
        )

    verified, vw, vh, vfps = _verify_output(cv2, written_path, written)
    if verified == 0:
        return OverlayRenderResult(
            None, written,
            f"wrote {written} frames with {codec} but the resulting file contains "
            f"no decodable video ({size_bytes} bytes at {written_path})",
            codec=codec, size_bytes=size_bytes,
        )

    if verified < written * 0.9:
        # Short by more than a rounding error. Reported, not hidden: a clip
        # that stops a third of the way through still plays, so nothing here
        # would otherwise notice.
        logger.warning(
            "annotated render is short: wrote %d frames, file decodes %d",
            written, verified,
        )

    logger.info(
        "annotated render: %s codec=%s %dx%d %d frames %.1f MB",
        written_path, codec, vw, vh, verified, size_bytes / 1e6,
    )
    return OverlayRenderResult(
        str(written_path), written, None,
        codec=codec, verified_frames=verified,
        width=vw, height=vh, size_bytes=size_bytes,
        fps=vfps or out_fps,
    )


def _scaled_points(points, scale: float):
    """`points` in ORIGINAL pixel space -> an int32 Nx1x2 array in render space,
    or None when the input is unusable. Every overlay goes through this, so
    nothing can forget the OVERLAY_MAX_WIDTH downscale the player boxes apply."""
    try:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    except (ValueError, TypeError):
        return None
    if pts.shape[0] < 2 or not np.isfinite(pts).all():
        return None
    if np.abs(pts).max() > _MAX_REPROJECTED_PX:
        return None
    return np.rint(pts * float(scale)).astype(np.int32).reshape(-1, 1, 2)


def _draw_field(cv2, image, field_region, scale: float) -> None:
    """The segmentation model's pitch mask: translucent fill plus a hard outline."""
    if field_region is None:
        return
    polygon = _scaled_points(getattr(field_region, "polygon", None) or [], scale)
    if polygon is None or len(polygon) < 3:
        return

    fill = image.copy()
    cv2.fillPoly(fill, [polygon], FIELD_BGR)
    cv2.addWeighted(fill, FIELD_FILL_ALPHA, image, 1.0 - FIELD_FILL_ALPHA, 0.0, image)
    cv2.polylines(image, [polygon], True, FIELD_BGR, 2, cv2.LINE_AA)

    confidence = getattr(field_region, "confidence", None)
    if confidence is None:
        return
    anchor = polygon.reshape(-1, 2)
    top = anchor[np.argmin(anchor[:, 1])]
    cv2.putText(image, f"FIELD {float(confidence):.2f}",
                (int(top[0]) + 4, max(12, int(top[1]) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, FIELD_BGR, 1, cv2.LINE_AA)


def _draw_goalposts(cv2, image, goalposts, scale: float) -> None:
    """The goalpost detector's boxes."""
    height, width = image.shape[:2]
    for post in goalposts or []:
        try:
            x1 = int(post.x * scale)
            y1 = int(post.y * scale)
            x2 = int((post.x + post.width) * scale)
            y2 = int((post.y + post.height) * scale)
        except (AttributeError, TypeError, ValueError):
            continue
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(image, (x1, y1), (x2, y2), GOALPOST_BGR, 2)
        confidence = getattr(post, "confidence", None)
        label = "GP" if confidence is None else f"GP {float(confidence):.2f}"
        cv2.putText(image, label, (x1 + 3, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, GOALPOST_BGR, 1, cv2.LINE_AA)


def _draw_pitch_outline(cv2, image, homography, scale: float) -> None:
    """Reproject the pitch markings through inv(H). If they land on the real
    touchlines, the calibration model's homography is usable -- which is the
    one thing a `calib: valid` text label cannot show."""
    if homography is None:
        return
    try:
        inverse = np.linalg.inv(np.asarray(homography, dtype=np.float64).reshape(3, 3))
    except (np.linalg.LinAlgError, ValueError, TypeError):
        return
    if not np.isfinite(inverse).all():
        return

    for line in PITCH_OUTLINE_M:
        source = np.asarray(line, dtype=np.float32).reshape(-1, 1, 2)
        try:
            projected = cv2.perspectiveTransform(source, inverse).reshape(-1, 2)
        except cv2.error:
            continue
        points = _scaled_points(projected, scale)
        if points is None:
            continue
        cv2.polylines(image, [points], False, PITCH_LINE_BGR, 2, cv2.LINE_AA)


def _draw_frame(cv2, image, detections, ball, scale: float = 1.0, *,
                field_region=None, goalposts=None, homography=None) -> None:
    height, width = image.shape[:2]

    # Field first: it is a filled region, and must sit UNDER everything else.
    _draw_field(cv2, image, field_region, scale)
    _draw_pitch_outline(cv2, image, homography, scale)
    _draw_goalposts(cv2, image, goalposts, scale)

    for det in detections or []:
        colour = team_colour_bgr(getattr(det, "team_id", None))
        # Detections are in SOURCE pixel space; the canvas may be smaller.
        x1 = max(0, int(det.x * scale))
        y1 = max(0, int(det.y * scale))
        x2 = min(width, int((det.x + det.width) * scale))
        y2 = min(height, int((det.y + det.height) * scale))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)

        # The track id is the only thing a viewer can use to follow one
        # player across frames, so it is drawn on a filled plate rather than
        # bare text -- unreadable white-on-white over a bright pitch is the
        # same as not labelling at all.
        label = f"#{det.player_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(th + 4, y1)
        cv2.rectangle(image, (x1, ty - th - 4), (x1 + tw + 6, ty), colour, -1)
        cv2.putText(image, label, (x1 + 3, ty - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (16, 16, 24), 1, cv2.LINE_AA)

    if ball is not None:
        # frame_data.BallObservation's fields are `pixel_x` / `pixel_y`; there
        # is no `.x`. A getattr default on the wrong name fails silently and
        # the ball marker is never drawn.
        bx = getattr(ball, "pixel_x", None)
        by = getattr(ball, "pixel_y", None)
        if bx is not None and by is not None:
            cx, cy = int(bx * scale), int(by * scale)
            cv2.circle(image, (cx, cy), 7, BALL_BGR, 2)
            cv2.circle(image, (cx, cy), 2, BALL_BGR, -1)


def _confidence_flag(value, attribute: str = "confidence") -> str:
    """`Y(0.86)` when a model produced something for this frame, `N` when it
    did not -- the HUD's answer to "did this model fire?"."""
    if value is None:
        return "N"
    confidence = getattr(value, attribute, None)
    if confidence is None:
        return "Y"
    try:
        return f"Y({float(confidence):.2f})"
    except (TypeError, ValueError):
        return "Y"


def _calibration_flag(calibration) -> str:
    """Accepts the legacy bool as well as a CalibrationState-shaped object."""
    if calibration is None:
        return "n/a"
    if isinstance(calibration, bool):
        return "valid" if calibration else "none"
    if not getattr(calibration, "valid", False):
        return "none"
    confidence = getattr(calibration, "confidence", None)
    try:
        return f"valid({float(confidence):.2f})"
    except (TypeError, ValueError):
        return "valid"


def _draw_hud(cv2, image, frame_number: int, timestamp: float,
              detections, calibration, *,
              ball=None, field_region=None, goalposts=None) -> None:
    """One line per frame naming what EVERY model produced for it, so a model
    that silently stopped firing shows up as an `N` rather than as an overlay
    that simply looks a bit emptier."""
    height, width = image.shape[:2]
    n_players = len(detections or [])
    n_goalposts = len(goalposts or [])

    ball_visible = ball if getattr(ball, "pixel_x", None) is not None else None

    text = (f"f{frame_number}  {timestamp:6.2f}s  "
            f"players:{n_players:2d}  "
            f"ball:{_confidence_flag(ball_visible)}  "
            f"field:{_confidence_flag(field_region)}  "
            f"gp:{n_goalposts if n_goalposts else '-'}  "
            f"calib:{_calibration_flag(calibration)}")
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    pad = 6
    box_h = th + pad * 2
    # Bottom-left: the top of a broadcast frame carries the scoreboard and
    # the middle carries the play.
    y0 = max(0, height - box_h)
    cv2.rectangle(image, (0, y0), (min(width, tw + pad * 2), height),
                  HUD_PANEL_BGR, -1)
    cv2.putText(image, text, (pad, height - pad - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_TEXT_BGR, 1, cv2.LINE_AA)
