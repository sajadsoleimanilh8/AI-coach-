from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from nexus.health.analyzer import HealthAnalyzer, HealthPattern
from nexus.personal.profile import ProfileStore
from nexus.personal.state import PersonalState, PersonalStateEngine
from nexus.personal.weakness import Weakness, WeaknessEngine

_DEFAULT_INTENSITY = 0.7
_MIN_INTENSITY = 0.2
_MAX_INTENSITY = 1.0

_RECOVERY_WEIGHT = 0.4
_FATIGUE_WEIGHT = 0.3
_STRESS_WEIGHT = 0.25
_SLEEP_WEIGHT = 0.2

_INTENSITY_FACTORS: tuple[tuple[str, float, str], ...] = (
    ("physical.recovery", _RECOVERY_WEIGHT, "low recovery"),
    ("mental.fatigue", _FATIGUE_WEIGHT, "high fatigue"),
    ("mental.stress", _STRESS_WEIGHT, "high stress"),
    ("lifestyle.sleep_quality", _SLEEP_WEIGHT, "poor sleep"),
)


@dataclass
class GenerationBrief:
    """The deterministic INPUT to generation — assembled in Python, never
    invented by the model (principle 1)."""

    user_id: str
    request_type: Literal["workout", "recovery", "study", "daily_plan", "nutrition"]
    state: PersonalState
    weaknesses: list[Weakness]
    health_patterns: list[HealthPattern]
    profile: dict[str, Any]
    constraints: dict[str, Any]
    target_intensity: float
    rationale: list[str]


class BriefBuilder:
    def __init__(
        self,
        state_engine: PersonalStateEngine,
        weakness_engine: WeaknessEngine,
        health_analyzer: HealthAnalyzer,
        profile_store: ProfileStore,
        *,
        default_intensity: float = _DEFAULT_INTENSITY,
        min_intensity: float = _MIN_INTENSITY,
        max_intensity: float = _MAX_INTENSITY,
    ) -> None:
        self._state_engine = state_engine
        self._weakness_engine = weakness_engine
        self._health_analyzer = health_analyzer
        self._profile_store = profile_store
        self._default_intensity = default_intensity
        self._min_intensity = min_intensity
        self._max_intensity = max_intensity

    async def build(
        self, *, user_id: str, request_type: str, constraints: dict[str, Any]
    ) -> GenerationBrief:
        state = await self._state_engine.get_state(user_id)
        weaknesses = await self._weakness_engine.detect(user_id)
        health_analysis = await self._health_analyzer.analyze(user_id)
        profile = await self._profile_store.get_profile(user_id)

        if not state.dimensions:
            return GenerationBrief(
                user_id=user_id,
                request_type=request_type,
                state=state,
                weaknesses=weaknesses,
                health_patterns=health_analysis.patterns,
                profile=profile,
                constraints=constraints,
                target_intensity=self._default_intensity,
                rationale=[
                    f"No personal signals recorded yet for this user — using the "
                    f"neutral default intensity ({self._default_intensity:.2f}). This "
                    f"plan is generic, not personalized to current state."
                ],
            )

        weaknesses_by_dim = {w.dimension: w for w in weaknesses}
        intensity = self._default_intensity
        rationale = [f"Starting from the default intensity of {self._default_intensity:.2f}."]

        for dimension, weight, label in _INTENSITY_FACTORS:
            weakness = weaknesses_by_dim.get(dimension)
            if weakness is None:
                continue
            adjustment = weakness.deviation * weight
            intensity -= adjustment
            rationale.append(
                f"Reduced intensity by {adjustment:.2f} due to {label} "
                f"({dimension}={weakness.current:.2f} vs. baseline "
                f"{weakness.baseline:.2f}, deviation {weakness.deviation:.2f})."
            )

        clamped = max(self._min_intensity, min(self._max_intensity, intensity))
        if clamped != intensity:
            rationale.append(
                f"Clamped intensity from {intensity:.2f} to {clamped:.2f} "
                f"(allowed range [{self._min_intensity:.2f}, {self._max_intensity:.2f}])."
            )

        return GenerationBrief(
            user_id=user_id,
            request_type=request_type,
            state=state,
            weaknesses=weaknesses,
            health_patterns=health_analysis.patterns,
            profile=profile,
            constraints=constraints,
            target_intensity=clamped,
            rationale=rationale,
        )
