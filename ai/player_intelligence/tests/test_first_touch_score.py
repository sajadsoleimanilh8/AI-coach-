"""first_touch_score: the formula, and above all its honesty rules.

The module's contract (see its comment block): never substitute a plausible
constant for a missing field; drop a missing field from THAT sub-score only;
count an event toward sample_size only if it contributed to something.
"""

from __future__ import annotations

import pytest

from ai.computer_vision.tactical_analysis.constants import (
    DECISION_TIME_MAX_S,
    DECISION_TIME_MIN_S,
    MAX_TOUCH_DISTANCE_M,
    MIN_SAMPLE_EVENTS,
    RETENTION_WINDOW_S,
)
from ai.player_intelligence.first_touch_score.score import score_first_touch

FULL = {
    "touch_distance_m": 1.0,
    "time_to_turnover_s": None,           # measured: no turnover in the window
    "direction_score": 80.0,
    "touch_execution_time_s": 0.5,
    "distance_to_nearest_opponent_m": None,  # measured: no opponent nearby
}


def event(i: int, **meta) -> dict:
    return {"event_id": f"e{i}", "metadata_json": meta}


def full_events(n: int, **overrides) -> list[dict]:
    return [event(i, **{**FULL, **overrides}) for i in range(n)]


# --- upstream gates ------------------------------------------------------------

def test_unusable_calibration_reports_upstream_confidence_not_a_score():
    result = score_first_touch(full_events(10), homography_confidence=0.1)
    assert result["value"] is None
    assert result["confidence"] == "low_upstream_confidence"
    assert result["sub_scores"] == {}


def test_events_with_unusable_per_event_calibration_are_dropped():
    good = full_events(MIN_SAMPLE_EVENTS)
    bad = [{**e, "event_id": f"bad{i}", "homography_confidence": 0.1} for i, e in enumerate(full_events(3))]
    result = score_first_touch(good + bad)
    assert result["sample_size"] == MIN_SAMPLE_EVENTS


# --- no fabrication ------------------------------------------------------------

def test_events_with_no_measured_fields_produce_no_score():
    result = score_first_touch([event(i) for i in range(10)])
    assert result["value"] is None
    assert result["confidence"] == "low_sample"
    assert result["sample_size"] == 0


def test_a_missing_field_is_dropped_from_its_own_sub_score_only():
    meta = {k: v for k, v in FULL.items() if k != "direction_score"}
    result = score_first_touch([event(i, **meta) for i in range(MIN_SAMPLE_EVENTS)])
    assert result["sub_scores"]["direction"] is None
    assert result["sub_scores"]["control"] is not None
    assert result["value"] is not None


def test_retention_key_absent_is_not_the_same_as_measured_none():
    """None means "measured, no turnover" (= fully retained); an absent key
    means "never measured" and must not be scored as 100."""
    measured = score_first_touch(full_events(MIN_SAMPLE_EVENTS))
    assert measured["sub_scores"]["retention"] == 100.0

    unmeasured_meta = {k: v for k, v in FULL.items() if k != "time_to_turnover_s"}
    unmeasured = score_first_touch([event(i, **unmeasured_meta) for i in range(MIN_SAMPLE_EVENTS)])
    assert unmeasured["sub_scores"]["retention"] is None


def test_missing_sub_scores_redistribute_weight_instead_of_counting_as_zero():
    """With only control measured, the headline equals the control score,
    not 30% of it."""
    only_control = [event(i, touch_distance_m=0.0) for i in range(MIN_SAMPLE_EVENTS)]
    result = score_first_touch(only_control)
    assert result["sub_scores"]["control"] == 100.0
    assert result["value"] == 100.0


# --- sample size and confidence -------------------------------------------------

def test_below_min_sample_is_flagged_low_sample_but_still_scored():
    result = score_first_touch(full_events(MIN_SAMPLE_EVENTS - 1))
    assert result["confidence"] == "low_sample"
    assert result["value"] is not None


def test_at_min_sample_is_normal():
    assert score_first_touch(full_events(MIN_SAMPLE_EVENTS))["confidence"] == "normal"


def test_sample_size_counts_contributing_events_only():
    contributing = full_events(4)
    empty = [event(100 + i) for i in range(6)]
    assert score_first_touch(contributing + empty)["sample_size"] == 4


# --- formula anchors ------------------------------------------------------------

@pytest.mark.parametrize("distance, expected", [
    (0.0, 100.0),
    (MAX_TOUCH_DISTANCE_M / 2, 50.0),
    (MAX_TOUCH_DISTANCE_M, 0.0),
    (MAX_TOUCH_DISTANCE_M * 3, 0.0),   # clamped, never negative
])
def test_control_scales_with_touch_distance(distance, expected):
    result = score_first_touch(full_events(MIN_SAMPLE_EVENTS, touch_distance_m=distance))
    assert result["sub_scores"]["control"] == expected


@pytest.mark.parametrize("seconds, expected", [
    (DECISION_TIME_MIN_S, 100.0),
    (DECISION_TIME_MAX_S, 0.0),
    (DECISION_TIME_MIN_S / 2, 100.0),   # faster than the realistic minimum clamps at 100
    (DECISION_TIME_MAX_S * 2, 0.0),     # slower than the maximum clamps at 0
])
def test_speed_scales_with_execution_time(seconds, expected):
    result = score_first_touch(full_events(MIN_SAMPLE_EVENTS, touch_execution_time_s=seconds))
    assert result["sub_scores"]["speed"] == expected


def test_quick_turnover_lowers_retention_in_proportion():
    result = score_first_touch(full_events(MIN_SAMPLE_EVENTS, time_to_turnover_s=RETENTION_WINDOW_S / 3))
    assert result["sub_scores"]["retention"] == pytest.approx(100.0 / 3, abs=0.1)


def test_no_opponent_nearby_gives_the_neutral_pressure_score():
    result = score_first_touch(full_events(MIN_SAMPLE_EVENTS, distance_to_nearest_opponent_m=None))
    assert result["sub_scores"]["pressure"] == 70.0


def test_close_pressure_rewards_good_control():
    tight = score_first_touch(full_events(MIN_SAMPLE_EVENTS, distance_to_nearest_opponent_m=0.5,
                                          touch_distance_m=0.0))
    loose = score_first_touch(full_events(MIN_SAMPLE_EVENTS, distance_to_nearest_opponent_m=0.5,
                                          touch_distance_m=MAX_TOUCH_DISTANCE_M * 0.8))
    assert tight["sub_scores"]["pressure"] > loose["sub_scores"]["pressure"]


def test_result_is_a_valid_metric_envelope():
    result = score_first_touch(full_events(MIN_SAMPLE_EVENTS))
    assert result["metric_name"] == "first_touch_score"
    assert result["method"] == "heuristic_proxy"
    assert 0.0 <= result["value"] <= 100.0
