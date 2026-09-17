"""Contract tests for ai/common: the metric envelope and the score helpers."""

from __future__ import annotations

import pytest

from ai.common import scoring
from ai.common.metrics import CONTRACT_KEYS, metric_result
from ai.performance_ai.match_readiness_predictor import constants as health_constants
from ai.psychology_ai import constants as psych_constants


def _build(**overrides):
    kwargs = dict(method="heuristic_proxy", confidence="normal", sample_size=3,
                  sub_scores={"a": 1.0}, schema_version="v3")
    kwargs.update(overrides)
    return metric_result("m", 42.0, **kwargs)


def test_envelope_has_exactly_the_contract_keys():
    """CONTRACT_KEYS is the written definition of the envelope; the builder is
    the only thing that produces one, so the two must agree exactly."""
    assert set(_build()) == set(CONTRACT_KEYS)


def test_contract_keys_are_in_the_documented_order():
    assert list(_build()) == list(CONTRACT_KEYS)


def test_none_value_is_preserved_not_defaulted():
    """None means "could not be measured"; it must never become 0."""
    assert metric_result("m", None, method="heuristic_proxy", confidence="low_sample",
                         sample_size=0, sub_scores={}, schema_version="v3")["value"] is None


def test_extra_keys_pass_through():
    assert _build(team_id="team-home")["team_id"] == "team-home"


def test_schema_version_is_required():
    with pytest.raises(TypeError):
        metric_result("m", 1.0, method="deterministic", confidence="normal", sample_size=1, sub_scores={})


@pytest.mark.parametrize("value, expected", [(-5, 0.0), (50, 50.0), (150, 100.0)])
def test_clamp_bounds(value, expected):
    assert scoring.clamp(value) == expected


def test_scale_1_to_10_floor_is_ten_not_zero():
    assert scoring.scale_1_to_10(1) == 10.0
    assert scoring.scale_1_to_10(10) == 100.0


def test_invert():
    assert scoring.invert(30) == 70.0


def test_both_engines_reexport_the_single_definition():
    """The copies had already drifted once; both names must now be the same object."""
    for name in ("clamp", "scale_1_to_10", "invert"):
        assert getattr(health_constants, name) is getattr(scoring, name)
        assert getattr(psych_constants, name) is getattr(scoring, name)
