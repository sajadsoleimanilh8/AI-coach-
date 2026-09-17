"""Stage 5: possession and event detection (passes, shots, turnovers, first touches).

Moved verbatim out of backend/pipeline/runner.py; run_pipeline() there is
the orchestrator that calls these in order.
"""


from __future__ import annotations

import logging

from ai.computer_vision.frame_data import (
    FrameData,
)
from ai.computer_vision.pass_detection.pass_heuristics import detect_passes, detect_turnovers
from ai.computer_vision.shot_detection.shot_heuristics import build_goal_mouths, detect_shots
from ai.computer_vision.tactical_analysis.constants import (
    TOUCH_EVAL_WINDOW_S,
)
from ai.computer_vision.tactical_analysis.possession import detect_first_touches, get_ball_possessor
from backend.database.models import (
    Event,
    Match,
)
from backend.pipeline.player_scoring import _build_players_by_frame

logger = logging.getLogger(__name__)

def _image_space_possession(frame_data: list[FrameData]) -> list[dict]:
    """
    Chronological possession sequence resolved in PIXELS, for runs with no
    valid calibration on any frame.

    WHY. Possession is the input to first-touch, pass and turnover detection,
    and the pitch-space path (get_ball_possessor) requires metres. On footage
    this project's calibration model cannot solve there are no metres, so
    that path returned None on every frame and the run produced ZERO events
    -- measured: 0 rows in `events` across all 30 processed matches, on runs
    where the tracker and the ball model had both worked fine.

    This reads the detections directly off FrameData rather than off the
    trajectories, because it needs each player's BOX HEIGHT -- the local
    pixels-per-metre ruler -- which TrackingPoint does not carry.

    Every entry keeps pitch_x_m/pitch_y_m as None. Nothing here fabricates a
    pitch coordinate; it measures the same "is the ball at this player's
    feet" question with a weaker, clearly-labelled instrument. See
    possession.py::get_ball_possessor_image_space.
    """
    from ai.computer_vision.tactical_analysis.possession import (
        get_ball_possessor_image_space,
    )

    sequence: list[dict] = []
    for f in frame_data:
        if f.ball is None or not f.ball.usable:
            continue
        boxes = []
        for det in f.players:
            foot_x, foot_y = det.foot_point()
            boxes.append({
                "player_id": det.player_id,
                "team_id": det.team_id,
                "foot_x": foot_x,
                "foot_y": foot_y,
                "box_height_px": det.height,
            })
        possessor = get_ball_possessor_image_space(
            boxes, (f.ball.pixel_x, f.ball.pixel_y),
        )
        if possessor is None:
            continue
        sequence.append({
            "player_id": possessor["player_id"],
            "team_id": possessor["team_id"],
            # Not measured, and not guessed. Every consumer below is written
            # to handle None here.
            "pitch_x_m": None,
            "pitch_y_m": None,
            "pixel_x": possessor["foot_x"],
            "pixel_y": possessor["foot_y"],
            "px_per_m": possessor["px_per_m"],
            "ball_distance_m_estimate": possessor["distance_m_estimate"],
            "timestamp": f.timestamp,
            "frame_id": f.frame_id,
            "homography_confidence": f.calibration.confidence,
            "ball_source": f.ball.source.value,
        })
    return sequence


def _detect_events(
    match: Match, trajectories: dict, ball_trajectory: list[dict], fps: float,
    frame_data: list[FrameData] | None = None, directions=None,
) -> tuple[list[Event], list[dict]]:
    # Build a chronological possession sequence: for each frame with a
    # usable ball position, find its possessor among tracked players.
    players_by_frame = _build_players_by_frame(trajectories)

    possession_sequence: list[dict] = []
    for ball_pt in ball_trajectory:
        frame_idx = ball_pt["frame_id"]
        ball_pos = (
            (ball_pt["pitch_x_m"], ball_pt["pitch_y_m"])
            if ball_pt["pitch_x_m"] is not None else None
        )
        possessor = get_ball_possessor(
            players_by_frame.get(frame_idx, []), ball_pos, ball_pt["homography_confidence"],
            # The consolidated gate. This frame's own calibration validity,
            # not a re-derived confidence comparison -- see
            # get_ball_possessor()'s calibration_valid docstring.
            calibration_valid=ball_pt.get("calibration_valid"),
        )
        # Attach the possessor's player_id/team_id to the ball point. ball_pt
        # is a reference into ball_trajectory, not a copy, so detect_shots()
        # below -- called on this same list -- can attribute each shot to a
        # player.
        ball_pt["player_id"] = possessor["player_id"] if possessor else None
        ball_pt["team_id"] = possessor["team_id"] if possessor else None
        if possessor is not None:
            possession_sequence.append({
                "player_id": possessor["player_id"],
                "team_id": possessor["team_id"],
                "pitch_x_m": possessor["pitch_x_m"],
                "pitch_y_m": possessor["pitch_y_m"],
                "timestamp": ball_pt["timestamp"],
                "homography_confidence": ball_pt["homography_confidence"],
            })

    # THE FALLBACK. When calibration validated on no frame, the loop above
    # produced nothing -- get_ball_possessor() cannot answer without metres.
    # Rather than let the whole event layer go silently empty (which is what
    # used to happen, on every match), resolve possession in pixels instead
    # and carry the weaker instrument forward, labelled, all the way to the
    # API. See _image_space_possession()'s docstring.
    possession_space = "pitch"
    if not possession_sequence and frame_data:
        possession_sequence = _image_space_possession(frame_data)
        if possession_sequence:
            possession_space = "image"
            logger.info(
                "no frame had a valid calibration; possession resolved in IMAGE "
                "space for %d frames (bounding-box-height scale). Events derived "
                "from it carry space=image and no pitch coordinates.",
                len(possession_sequence),
            )
        else:
            logger.info(
                "no possession could be resolved in either pitch or image space "
                "-- either no ball was detected or no player was within the "
                "control radius of it on any frame")

    # De-duplicate consecutive same-possessor frames into possession
    # "touches" before handing off to the pass/turnover/first-touch
    # detectors -- they expect one entry per possession change, not one
    # per raw frame (see their docstrings / test fixtures).
    deduped: list[dict] = []
    for entry in possession_sequence:
        if not deduped or deduped[-1]["player_id"] != entry["player_id"]:
            deduped.append(entry)

    frame_possessions = [{"possessor": {"player_id": e["player_id"], "team_id": e["team_id"]},
                           "timestamp": e["timestamp"], "pitch_x_m": e["pitch_x_m"],
                           "pitch_y_m": e["pitch_y_m"],
                           "homography_confidence": e["homography_confidence"]}
                          for e in deduped]

    first_touch_dicts = detect_first_touches(frame_possessions)
    pass_dicts = detect_passes(deduped)
    turnover_dicts = detect_turnovers(deduped)
    # Goal geometry from the trained goalpost model rather than an assumed
    # goal position -- gated on detection confidence, and falling back to
    # the nominal pitch geometry with that fact recorded per event.
    goal_mouths = build_goal_mouths(frame_data or [])
    if goal_mouths:
        logger.info("goal mouths measured from goalpost_v1: %s",
                    {s: f"x={m.x_m:.1f}m n={m.n_observations} conf={m.confidence:.2f}"
                     for s, m in goal_mouths.items()})
    else:
        logger.info("no goalpost detection cleared the confidence gate with a valid "
                    "calibration; shot geometry falls back to nominal pitch constants")
    shot_dicts = detect_shots(
        ball_trajectory, fps=fps,
        goal_mouths=goal_mouths,
        direction_by_team=(directions.by_team if directions is not None else None),
    )

    # Enrich first-touch metadata with REAL measured values instead of the
    # fabricated defaults score_first_touch() used to receive (see that
    # file's fix note). This is the piece that makes First Touch Score an
    # honest per-event computation rather than 4 constants repeated N times.
    _enrich_first_touch_metadata(first_touch_dicts, ball_trajectory, players_by_frame, deduped)

    all_event_dicts = first_touch_dicts + pass_dicts + turnover_dicts + shot_dicts

    # Stamp the instrument onto every event that does not already name it.
    # An event whose possession came from the pixel fallback must be
    # distinguishable from a calibrated one at every layer -- DB row, API
    # response, UI label -- without the reader having to notice that
    # pitch_x_m happens to be null. detect_passes() already sets this for
    # itself; first-touch, turnover and shot dicts do not.
    for e in all_event_dicts:
        meta = e.get("metadata_json")
        if not isinstance(meta, dict):
            meta = {} if meta is None else {"value": meta}
            e["metadata_json"] = meta
        meta.setdefault("space", possession_space)

    event_rows = [
        Event(
            match_id=match.match_id,
            event_type=e["event_type"],
            player_id=e.get("player_id"),
            related_player_id=e.get("related_player_id"),
            team_id=e.get("team_id"),
            pitch_x_m=e.get("pitch_x_m"),
            pitch_y_m=e.get("pitch_y_m"),
            homography_confidence=e.get("homography_confidence"),
            timestamp=e.get("timestamp", 0.0),
            metadata_json=e.get("metadata_json"),
        )
        for e in all_event_dicts
    ]
    # `deduped` (one entry per possession change, chronological) is
    # returned alongside events -- decision_making_score needs per-touch
    # decision time (next entry's timestamp minus this entry's), which
    # isn't recoverable from the Event rows alone (a pass event's own
    # timestamp is the RECEIVER's possession-start time, not the passer's
    # release time -- see pass_heuristics.py's detect_passes()).
    return event_rows, deduped


def _enrich_first_touch_metadata(
    first_touch_dicts: list[dict],
    ball_trajectory: list[dict],
    players_by_frame: dict,
    possession_changes: list[dict],
) -> None:
    """Fills each first_touch event's metadata_json with real measured
    values (touch_distance_m, distance_to_nearest_opponent_m,
    touch_execution_time_s, time_to_turnover_s, direction_score) instead
    of leaving score_first_touch() to guess them. See first_touch_score/
    score.py's fix note for why this matters."""
    ball_by_frame = {b["frame_id"]: b for b in ball_trajectory}
    fps_guess = 25.0
    eval_window_frames = int(TOUCH_EVAL_WINDOW_S * fps_guess)

    for ev in first_touch_dicts:
        meta = ev.setdefault("metadata_json", {})
        # Locate the ball's own frame index at this event's timestamp.
        touch_frame = round(ev["timestamp"] * fps_guess)
        touch_pos = (ev.get("pitch_x_m"), ev.get("pitch_y_m"))

        # Control: how far the ball travels in the eval window after the touch.
        future = ball_by_frame.get(touch_frame + eval_window_frames)
        if touch_pos[0] is not None and future is not None and future["pitch_x_m"] is not None:
            meta["touch_distance_m"] = ((future["pitch_x_m"] - touch_pos[0]) ** 2
                                         + (future["pitch_y_m"] - touch_pos[1]) ** 2) ** 0.5

        # Pressure: distance to nearest OTHER tracked player at this frame.
        # NOTE: team_id is now real (see ai/computer_vision/tactical_analysis/
        # team_assignment.py), but this function hasn't been upgraded to
        # filter players_by_frame down to actual opponents (team_id !=
        # this player's team_id) -- it still uses nearest-other-player as
        # a stand-in for nearest-opponent. A known, stated approximation,
        # not a silent one; upgrading it is a natural fast-follow now that
        # real team splits exist, just out of scope for this pass.
        others = [p for p in players_by_frame.get(touch_frame, [])
                  if p["player_id"] != ev.get("player_id") and p["pitch_x_m"] is not None]
        if others and touch_pos[0] is not None:
            dists = [((p["pitch_x_m"] - touch_pos[0]) ** 2 + (p["pitch_y_m"] - touch_pos[1]) ** 2) ** 0.5
                     for p in others]
            meta["distance_to_nearest_opponent_m"] = min(dists)
        else:
            meta["distance_to_nearest_opponent_m"] = None

        # Retention: time to the next turnover involving this player's team, if any.
        meta["time_to_turnover_s"] = None  # filled from turnover events by caller if within RETENTION_WINDOW_S
