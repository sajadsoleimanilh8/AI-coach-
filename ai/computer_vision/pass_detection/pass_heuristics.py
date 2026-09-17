"""
Pass detection heuristics.
Implementation Spec §2.
"""

from __future__ import annotations

import math

from ai.computer_vision.tactical_analysis.constants import PASS_MIN_DISTANCE_M


def _image_space_distance_m(p1: dict, p2: dict) -> float | None:
    """Separation between two possession points, in metres, from PIXELS.

    The fallback rung, used only when neither point carries pitch metres.
    Both points must have supplied `px_per_m` -- the local pixels-per-metre
    ruler derived from that player's own bounding-box height (see
    possession.py::image_scale_px_per_m). The mean of the two rulers is used
    because a pass spans both players' depths.

    Returns None whenever anything needed is missing, so the caller declines
    to emit a pass rather than emitting one on a guessed distance.
    """
    x1, y1 = p1.get("pixel_x"), p1.get("pixel_y")
    x2, y2 = p2.get("pixel_x"), p2.get("pixel_y")
    s1, s2 = p1.get("px_per_m"), p2.get("px_per_m")
    if None in (x1, y1, x2, y2, s1, s2):
        return None
    scale = (float(s1) + float(s2)) / 2.0
    if scale <= 0:
        return None
    return math.hypot(x2 - x1, y2 - y1) / scale


def detect_passes(possession_sequence: list[dict]) -> list[dict]:
    """
    Pass detection: possession transfers from player A to player B on the same team,
    ball travels a minimum distance (>= 3m) with no opponent touch in between.

    Args:
        possession_sequence: chronological list of possession events with keys:
            player_id, team_id, pitch_x_m, pitch_y_m, timestamp, homography_confidence.

            IMAGE-SPACE FALLBACK: when pitch_x_m/pitch_y_m are None -- the
            normal case on footage this project's calibration cannot solve --
            entries may instead carry pixel_x/pixel_y plus px_per_m, and the
            minimum-distance gate is then evaluated on a distance ESTIMATED
            from those pixels (see _image_space_distance_m and the
            PLAYER_HEIGHT_M note in tactical_analysis/constants.py).

            The resulting event still reports pitch_x_m/pitch_y_m as None. The
            gate is applied to an estimate; the position is not invented.
            metadata_json carries `space` so no consumer has to infer which
            of the two paths produced a given pass.

    Returns:
        list of pass Event dicts.
    """
    events = []
    if len(possession_sequence) < 2:
        return events

    for i in range(len(possession_sequence) - 1):
        p1 = possession_sequence[i]
        p2 = possession_sequence[i + 1]

        pid1, team1 = p1.get("player_id"), p1.get("team_id")
        pid2, team2 = p2.get("player_id"), p2.get("team_id")

        if pid1 is None or pid2 is None:
            continue

        # Same team, different players
        if team1 == team2 and pid1 != pid2:
            x1, y1 = p1.get("pitch_x_m"), p1.get("pitch_y_m")
            x2, y2 = p2.get("pitch_x_m"), p2.get("pitch_y_m")

            in_pitch_space = None not in (x1, y1, x2, y2)
            if in_pitch_space:
                dist = math.hypot(x2 - x1, y2 - y1)
            else:
                dist = _image_space_distance_m(p1, p2)
                if dist is None:
                    continue

            if dist >= PASS_MIN_DISTANCE_M:
                metadata = {
                    "pass_distance_m": round(dist, 2),
                    "start_x": x1,
                    "start_y": y1,
                    "space": "pitch" if in_pitch_space else "image",
                }
                if not in_pitch_space:
                    # Named so the number is never read as a metric
                    # measurement, and the pixel endpoints are kept so the
                    # UI can place a marker on the video without any
                    # pitch-space claim being made.
                    metadata["pass_distance_m_is_estimate"] = True
                    metadata["start_pixel_x"] = p1.get("pixel_x")
                    metadata["start_pixel_y"] = p1.get("pixel_y")
                    metadata["end_pixel_x"] = p2.get("pixel_x")
                    metadata["end_pixel_y"] = p2.get("pixel_y")
                events.append({
                    "event_type": "pass",
                    "player_id": pid1,
                    "related_player_id": pid2,
                    "team_id": team1,
                    "timestamp": p2.get("timestamp", 0.0),
                    "pitch_x_m": x2,
                    "pitch_y_m": y2,
                    "homography_confidence": p2.get("homography_confidence", 1.0),
                    "metadata_json": metadata,
                })

    return events


def detect_turnovers(possession_sequence: list[dict]) -> list[dict]:
    """
    Turnover detection: possession transfers to a player on the opposing team.

    Returns list of turnover Event dicts.
    """
    events = []
    if len(possession_sequence) < 2:
        return events

    for i in range(len(possession_sequence) - 1):
        p1 = possession_sequence[i]
        p2 = possession_sequence[i + 1]

        pid1, team1 = p1.get("player_id"), p1.get("team_id")
        pid2, team2 = p2.get("player_id"), p2.get("team_id")

        if pid1 is None or pid2 is None:
            continue

        # Different teams
        if team1 != team2 and team1 is not None and team2 is not None:
            events.append({
                "event_type": "turnover",
                "player_id": pid1,  # lost by
                "related_player_id": pid2,  # won by
                "team_id": team1,
                "timestamp": p2.get("timestamp", 0.0),
                "pitch_x_m": p2.get("pitch_x_m"),
                "pitch_y_m": p2.get("pitch_y_m"),
                "homography_confidence": p2.get("homography_confidence", 1.0),
                "metadata_json": {"lost_by": pid1, "won_by": pid2},
            })

    return events
