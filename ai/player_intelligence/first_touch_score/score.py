"""
First Touch Score implementation.
Analysis Logic Design v3 (docs/data_analysis.md §1).
"""

from __future__ import annotations
from ai.computer_vision.tactical_analysis.constants import (
    DECISION_TIME_MAX_S,
    DECISION_TIME_MIN_S,
    HOMOGRAPHY_CONFIDENCE_MIN,
    MAX_TOUCH_DISTANCE_M,
    MIN_SAMPLE_EVENTS,
    RETENTION_WINDOW_S,
    SCHEMA_VERSION,
    pressure_level,
)
from ai.computer_vision.tactical_analysis.utils import clip

REQUIRED_EVENT_FIELDS = (
    "touch_distance_m",
    "time_to_turnover_s",
    "direction_score",
    "touch_execution_time_s",
    "distance_to_nearest_opponent_m",
)


def score_first_touch(
    events: list[dict],
    homography_confidence: float = 1.0,
) -> dict:
    """
    Computes First Touch Score for a player based on first_touch event list.
    """
    if homography_confidence < HOMOGRAPHY_CONFIDENCE_MIN:
        return {
            "metric_name": "first_touch_score",
            "value": None,
            "method": "heuristic_proxy",
            "confidence": "low_upstream_confidence",
            "sample_size": len(events),
            "sub_scores": {},
            "schema_version": SCHEMA_VERSION,
        }

    valid_events = [e for e in events if e.get("homography_confidence", 1.0) >= HOMOGRAPHY_CONFIDENCE_MIN]

    control_scores, retention_scores, direction_scores = [], [], []
    speed_scores, pressure_scores = [], []
    contributing_event_ids: set = set()

    for ev in valid_events:
        meta = ev.get("metadata_json") or {}
        event_key = ev.get("event_id", id(ev))
        contributed = False

        touch_dist = meta.get("touch_distance_m")
        if touch_dist is not None:
            ctrl = 100.0 * (1.0 - min(touch_dist / MAX_TOUCH_DISTANCE_M, 1.0))
            control_scores.append(ctrl)
            contributed = True
        else:
            ctrl = None

        if "time_to_turnover_s" in meta:
            turnover_s = meta.get("time_to_turnover_s")
            if turnover_s is not None and turnover_s < RETENTION_WINDOW_S:
                ret = 100.0 * (turnover_s / RETENTION_WINDOW_S)
            else:
                ret = 100.0
            retention_scores.append(ret)
            contributed = True

        dir_score = meta.get("direction_score")
        if dir_score is not None:
            direction_scores.append(dir_score)
            contributed = True

        exec_time = meta.get("touch_execution_time_s")
        if exec_time is not None:
            spd = 100.0 * clip(
                (DECISION_TIME_MAX_S - exec_time) / (DECISION_TIME_MAX_S - DECISION_TIME_MIN_S), 0.0, 1.0
            )
            speed_scores.append(spd)
            contributed = True

        if "distance_to_nearest_opponent_m" in meta:
            opp_dist = meta.get("distance_to_nearest_opponent_m")
            p = pressure_level(opp_dist)
            if p is None or p < 0.2:
                press = 70.0
            elif ctrl is not None:
                press = 100.0 * (ctrl / 100.0) * p
            else:
                press = 70.0 * p + 70.0 * (1 - p)
            pressure_scores.append(press)
            contributed = True

        if contributed:
            contributing_event_ids.add(event_key)

    n_events = len(contributing_event_ids)

    if n_events == 0:
        return {
            "metric_name": "first_touch_score",
            "value": None,
            "method": "heuristic_proxy",
            "confidence": "low_sample",
            "sample_size": 0,
            "sub_scores": {},
            "schema_version": SCHEMA_VERSION,
        }

    def _avg(values: list[float]) -> float | None:
        return float(sum(values) / len(values)) if values else None

    avg_control = _avg(control_scores)
    avg_retention = _avg(retention_scores)
    avg_direction = _avg(direction_scores)
    avg_speed = _avg(speed_scores)
    avg_pressure = _avg(pressure_scores)

    components = {
        "control": (avg_control, 0.30),
        "retention": (avg_retention, 0.25),
        "direction": (avg_direction, 0.20),
        "speed": (avg_speed, 0.15),
        "pressure": (avg_pressure, 0.10),
    }
    valid = {k: (v, w) for k, (v, w) in components.items() if v is not None}

    confidence_enum = "low_sample" if n_events < MIN_SAMPLE_EVENTS else "normal"

    if not valid:
        final_val = None
        confidence_enum = "low_sample"
    else:
        weight_sum = sum(w for _, w in valid.values())
        final_val = sum(v * w for v, w in valid.values()) / weight_sum

    return {
        "metric_name": "first_touch_score",
        "value": round(final_val, 1) if final_val is not None else None,
        "method": "heuristic_proxy",
        "confidence": confidence_enum,
        "sample_size": n_events,
        "sub_scores": {
            "control": round(avg_control, 1) if avg_control is not None else None,
            "retention": round(avg_retention, 1) if avg_retention is not None else None,
            "direction": round(avg_direction, 1) if avg_direction is not None else None,
            "speed": round(avg_speed, 1) if avg_speed is not None else None,
            "pressure": round(avg_pressure, 1) if avg_pressure is not None else None,
        },
        "schema_version": SCHEMA_VERSION,
    }
