"""
Burns the pipeline's own tracking output into a playable video.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

TEAM_COLOURS_BGR = {
    "team-home": (255, 212, 0),
    "team-away": (182, 114, 244),
}
UNASSIGNED_BGR = (139, 116, 100)
BALL_BGR = (0, 255, 255)
HUD_TEXT_BGR = (235, 235, 240)
HUD_PANEL_BGR = (24, 18, 16)

RENDER_OVERLAY_VIDEO = os.getenv("RENDER_OVERLAY_VIDEO", "1") not in ("0", "false", "False")

OVERLAY_MAX_WIDTH = int(os.getenv("OVERLAY_MAX_WIDTH", "960"))

OVERLAY_CODEC_CHAIN: tuple[tuple[str, str, str], ...] = (
    ("avc1", ".mp4", "video/mp4"),
    ("VP80", ".webm", "video/webm"),
)

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
    """Where the annotated render for this video is WRITTEN."""
    resolved = suffix or _codec_chain()[0][1]
    return Path(storage_path).parent / "processed" / f"{video_id}_tracked{resolved}"


def find_overlay_output(storage_path: str, video_id: str) -> Path | None:
    """The annotated render for this video that is actually ON DISK, or None."""
    for candidate_suffix in OVERLAY_SUFFIXES:
        candidate = overlay_output_path(storage_path, video_id, candidate_suffix)
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def media_type_for(path: Path | str) -> str:
    """MIME type for a rendered overlay, from its suffix."""
    return MIME_BY_SUFFIX.get(Path(path).suffix.lower(), "application/octet-stream")


def team_colour_bgr(team_id: str | None) -> tuple[int, int, int]:
    """Colour for a team id, neutral for anything unrecognised."""
    if team_id is None:
        return UNASSIGNED_BGR
    return TEAM_COLOURS_BGR.get(str(team_id).strip().lower(), UNASSIGNED_BGR)


def _even(value: int) -> int:
    """Nearest even size >= 2."""
    return max(2, int(value) // 2 * 2)


def _open_writer(cv2, out_path: Path, fps: float, size: tuple[int, int]):
    """First codec in the chain that actually opens. Returns (writer, path, codec)."""
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
        try:
            if candidate.exists() and candidate.stat().st_size == 0:
                candidate.unlink()
        except OSError:
            pass
        attempts.append(fourcc)
    return None, None, None


def _verify_output(cv2, path: Path, expected_frames: int) -> tuple[int, int, int, float]:
    """Re-open the written file and report what is REALLY in it."""
    check = cv2.VideoCapture(str(path))
    try:
        if not check.isOpened():
            return 0, 0, 0, 0.0
        width = int(check.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(check.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(check.get(cv2.CAP_PROP_FPS) or 0.0)
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
    progress=None,
) -> OverlayRenderResult:
    """
    Draw `frames` (list[list[TrackedDetection]], indexed by frame number)
    onto the source video and write an annotated clip beside it.
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

            if out_size != (image.shape[1], image.shape[0]):
                image = cv2.resize(image, out_size, interpolation=cv2.INTER_AREA)

            _draw_frame(cv2, image, frames[frame_number],
                        ball_by_frame.get(frame_number), scale)
            _draw_hud(cv2, image, frame_number, frame_number / out_fps,
                      frames[frame_number], calibration_by_frame.get(frame_number))
            writer.write(image)
            written += 1

            if progress and (written % report_every == 0 or written == total):
                progress(written, total)

    except Exception as error:
        logger.warning("overlay render failed after %d frames: %s", written, error)
        return OverlayRenderResult(None, written, str(error), codec=codec)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if written == 0 or written_path is None:
        return OverlayRenderResult(None, 0, "no frames decoded", codec=codec)

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


def _draw_frame(cv2, image, detections, ball, scale: float = 1.0) -> None:
    height, width = image.shape[:2]

    for det in detections or []:
        colour = team_colour_bgr(getattr(det, "team_id", None))
        x1 = max(0, int(det.x * scale))
        y1 = max(0, int(det.y * scale))
        x2 = min(width, int((det.x + det.width) * scale))
        y2 = min(height, int((det.y + det.height) * scale))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)

        label = f"#{det.player_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(th + 4, y1)
        cv2.rectangle(image, (x1, ty - th - 4), (x1 + tw + 6, ty), colour, -1)
        cv2.putText(image, label, (x1 + 3, ty - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (16, 16, 24), 1, cv2.LINE_AA)

    if ball is not None:
        bx = getattr(ball, "pixel_x", None)
        by = getattr(ball, "pixel_y", None)
        if bx is not None and by is not None:
            cx, cy = int(bx * scale), int(by * scale)
            cv2.circle(image, (cx, cy), 7, BALL_BGR, 2)
            cv2.circle(image, (cx, cy), 2, BALL_BGR, -1)


def _draw_hud(cv2, image, frame_number: int, timestamp: float,
              detections, calibration_valid: bool | None) -> None:
    """A one-line strip naming what this frame is and what it was analysed under."""
    height, width = image.shape[:2]
    n_players = len(detections or [])
    if calibration_valid is None:
        calib = "n/a"
    else:
        calib = "valid" if calibration_valid else "none"

    text = (f"f{frame_number}  {timestamp:6.2f}s  players:{n_players:2d}  "
            f"calib:{calib}")
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    pad = 6
    box_h = th + pad * 2
    y0 = max(0, height - box_h)
    cv2.rectangle(image, (0, y0), (min(width, tw + pad * 2), height),
                  HUD_PANEL_BGR, -1)
    cv2.putText(image, text, (pad, height - pad - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_TEXT_BGR, 1, cv2.LINE_AA)
