"""The Standard Output Contract every scorer returns.

One definition of the metric envelope that was previously hand-built as a
dict literal in ~35 places. It stays a plain dict at runtime -- the database
mappers in backend/pipeline/runner.py, the API responses and the NEXUS
adapters all consume it as a dict -- but `MetricResult` gives it a checkable
shape, and `metric_result()` makes it impossible to forget a key.

The allowed `method` / `confidence` values mirror MetricMethod and
MetricConfidence in backend/database/models.py. They are repeated as Literal
types here rather than imported, because ai/ must not depend on backend/.
"""

from __future__ import annotations

from typing import Any, Literal

Method = Literal["ml_trained", "deterministic", "heuristic_proxy"]
Confidence = Literal["normal", "low_sample", "low_upstream_confidence"]


# A metric envelope is a plain dict: it is stored as JSON, returned from the
# API, and passed around by the NEXUS adapters, so every consumer treats it as
# one. This alias names the shape without making it a TypedDict -- a TypedDict
# is not assignable to dict[str, Any], which forced annotation churn through
# every list and caller that carries these envelopes for no checking benefit.
# The guarantee that matters (no missing or misspelled key) comes from building
# them only through metric_result() below; CONTRACT_KEYS is the definition, and
# ai/common/tests/test_common.py asserts the builder matches it.
MetricResult = dict[str, Any]

CONTRACT_KEYS = (
    "metric_name",
    "value",       # number for most metrics; a label such as "4-3-3" for formation
    "method",
    "confidence",
    "sample_size",
    "sub_scores",
    "schema_version",
)


def metric_result(
    metric_name: str,
    value: Any,
    *,
    method: Method,
    confidence: Confidence,
    sample_size: int,
    sub_scores: dict[str, Any],
    schema_version: str,
    **extra: Any,
) -> MetricResult:
    """Build one metric envelope.

    `value=None` is a legitimate result ("could not be measured"), never a
    placeholder for zero. `schema_version` is required rather than defaulted
    because each engine versions its own contract (tactical scorers are "v3",
    the health and psychology engines "v1").
    """
    result: dict[str, Any] = {
        "metric_name": metric_name,
        "value": value,
        "method": method,
        "confidence": confidence,
        "sample_size": sample_size,
        "sub_scores": sub_scores,
        "schema_version": schema_version,
    }
    result.update(extra)
    return result
