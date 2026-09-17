from __future__ import annotations

DIMENSION_GROUPS: dict[str, tuple[str, ...]] = {
    "physical": (
        "physical.energy",
        "physical.recovery",
        "physical.activity",
        "physical.mobility",
        "physical.strength",
        "physical.endurance",
    ),
    "mental": (
        "mental.focus",
        "mental.stress",
        "mental.mood",
        "mental.fatigue",
    ),
    "cognitive": (
        "cognitive.working_memory",
        "cognitive.processing_speed",
        "cognitive.creativity",
    ),
    "lifestyle": (
        "lifestyle.sleep_quality",
        "lifestyle.sleep_consistency",
        "lifestyle.nutrition",
        "lifestyle.hydration",
    ),
    "sports": (
        "sports.speed",
        "sports.acceleration",
        "sports.agility",
        "sports.endurance",
        "sports.decision_making",
        "sports.positioning",
    ),
}

ALL_DIMENSIONS: tuple[str, ...] = tuple(
    dimension for members in DIMENSION_GROUPS.values() for dimension in members
)

# How much a single signal is trusted, by how it was obtained — used both to
# weight PersonalStateEngine's recency-weighted mean and to compute
# DimensionState.confidence (principle 5: never treat a guess like a fact).
SOURCE_CONFIDENCE: dict[str, float] = {
    "explicit": 0.95,
    "behavioral": 0.75,
    "temporal": 0.65,
    "inferred": 0.55,
}

# Dimensions where a HIGHER raw value is worse (more stress, more fatigue).
# Every consumer that compares "current vs. baseline" or "rising vs. falling"
# must sign-correct through this set consistently, or a worsening trend on an
# inverted dimension reads backwards (e.g. rising stress looking "improving").
INVERTED_DIMENSIONS: frozenset[str] = frozenset({"mental.stress", "mental.fatigue"})
