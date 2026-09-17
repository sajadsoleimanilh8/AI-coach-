from __future__ import annotations

from nexus.personal.state import PersonalStateEngine
from nexus.sports.adapter import MatchAnalysis

# CV-derived proxies, not self-reported facts — this is the same
# reasoning SOURCE_CONFIDENCE documents for "inferred" (0.55): a tracked
# player's press_resistance_score is a heuristic estimate from tracking
# data, several inference steps removed from the player stating anything
# about themselves.
_SIGNAL_SOURCE = "inferred"
_METRIC_SCALE_MAX = 100.0


async def ingest_player_metrics_as_signals(
    state_engine: PersonalStateEngine,
    *,
    user_id: str,
    analysis: MatchAnalysis,
    mapping: dict[str, str],
) -> int:
    """Feeds AVAILABLE player metrics into PersonalStateEngine as
    source="inferred" signals, normalizing the pipeline's 0-100 scale to
    the 0..1 scale every NEXUS dimension uses. `mapping` (football
    metric_name -> NEXUS sports.* dimension) is config-driven
    (sports.metric_dimension_map) rather than hardcoded here, since the
    pipeline's metric set is expected to grow independently of NEXUS.
    Unavailable metrics (value=None or low_upstream_confidence) are
    skipped entirely — never coerced into a fabricated signal."""
    recorded = 0
    for metric in analysis.player_metrics:
        if not metric.is_available:
            continue
        dimension = mapping.get(metric.metric_name)
        if dimension is None:
            continue
        if not isinstance(metric.value, (int, float)):
            continue

        normalized = max(0.0, min(1.0, metric.value / _METRIC_SCALE_MAX))
        await state_engine.record_signal(
            user_id=user_id,
            dimension=dimension,
            value=normalized,
            source=_SIGNAL_SOURCE,
            note=f"football pipeline metric={metric.metric_name} player_id={metric.player_id}",
        )
        recorded += 1
    return recorded
