from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from nexus.core.exceptions import ModelNotFoundError, ProviderUnavailableError
from nexus.core.latency_tracker import LatencyTracker
from nexus.core.provider_manager import ProviderManager
from nexus.core.providers import AIProvider
from nexus.core.types import ModelInfo, RoutingPolicy, TaskType
from nexus.logging_setup.logger import get_logger
from nexus.models.registry import get_model

logger = get_logger("core.router")

# Maps a TaskType to the capability key in ModelInfo.capabilities that best
# predicts quality for that task. Anything not listed here falls back to
# "reasoning" — a deliberately data-driven default rather than a per-task
# if/else chain, so Phase 3 can extend the map without touching scoring logic.
_TASK_CAPABILITY_MAP: dict[TaskType, str] = {
    TaskType.CODING: "coding",
    TaskType.MATH: "math",
    TaskType.RESEARCH: "reasoning",
    TaskType.COMPLEX_REASONING: "reasoning",
    TaskType.TRANSLATION: "language",
    TaskType.GENERAL: "language",
    TaskType.VISION: "vision",
    TaskType.DATA_ANALYSIS: "coding",
    TaskType.PRIVATE_PERSONAL: "reasoning",
    TaskType.HEALTH: "reasoning",
    TaskType.FITNESS: "reasoning",
    TaskType.SPORTS: "reasoning",
    TaskType.PLANNING: "reasoning",
    TaskType.DOCUMENT_ANALYSIS: "language",
    # Placeholder until Phase 14 adds real audio capability scores.
    TaskType.AUDIO: "language",
}
_DEFAULT_CAPABILITY_KEY = "reasoning"


def _capability_key(task_type: TaskType) -> str:
    return _TASK_CAPABILITY_MAP.get(task_type, _DEFAULT_CAPABILITY_KEY)


def _total_cost(model: ModelInfo) -> float:
    return (model.cost_per_1k_input_tokens or 0.0) + (model.cost_per_1k_output_tokens or 0.0)


class CapabilityProvider(Protocol):
    """Structural type for whatever supplies capability scores.

    Declared as a Protocol rather than importing
    nexus.intelligence.CapabilityMatrix because that module reads
    nexus.evaluation types, and nexus/core/ must not depend on
    nexus/evaluation/. The router only ever needs this one synchronous
    method, so structural typing keeps the dependency arrow pointing the
    right way without giving up type checking.
    """

    def effective_capabilities(self, model_id: str) -> dict[str, float]: ...


@dataclass
class RoutingDecision:
    provider_name: str
    model_id: str
    task_type: TaskType
    policy: RoutingPolicy
    reason: str


class ModelRouter:
    """Scores candidate models from the registry and picks one per policy.

    `route()` resolves a single best decision; `route_with_failover()` walks
    the full ranked candidate list, health-checking each provider before
    trusting it, and guarantees a usable (provider, decision) pair by falling
    all the way back to local if every other candidate is unhealthy.
    """

    def __init__(
        self,
        provider_manager: ProviderManager,
        model_registry: list[ModelInfo],
        default_policy: RoutingPolicy = RoutingPolicy.BALANCED,
        latency_tracker: LatencyTracker | None = None,
        capability_matrix: CapabilityProvider | None = None,
    ) -> None:
        self._provider_manager = provider_manager
        self._model_registry = model_registry
        self._default_policy = default_policy
        self._latency_tracker = latency_tracker
        # None (the default) means score on ModelInfo.capabilities exactly
        # as before — every existing caller and test is unchanged.
        self._capability_matrix = capability_matrix

    def _capability_for(self, model: ModelInfo, cap_key: str) -> float:
        if self._capability_matrix is None:
            return model.capabilities.get(cap_key, 0.0)
        effective = self._capability_matrix.effective_capabilities(model.id)
        # A matrix with no measurements for this model returns the static
        # dict unchanged, so this falls through to the same number the
        # None branch above would have produced.
        return effective.get(cap_key, model.capabilities.get(cap_key, 0.0))

    def route(
        self,
        *,
        task_type: TaskType = TaskType.GENERAL,
        policy: RoutingPolicy | None = None,
        requested_model_id: str | None = None,
        require_tool_calling: bool = False,
    ) -> RoutingDecision:
        effective_policy = policy or self._default_policy

        if requested_model_id:
            # provider_for_model() validates the model exists and its provider is
            # registered; the provider *key* (not necessarily provider.name — see
            # ProviderManager's docstring) comes from the registry entry itself.
            self._provider_manager.provider_for_model(requested_model_id)
            model_info = get_model(requested_model_id)
            assert model_info is not None  # provider_for_model() already validated this
            return RoutingDecision(
                provider_name=model_info.provider,
                model_id=requested_model_id,
                task_type=task_type,
                policy=effective_policy,
                reason=(
                    f"Explicit model_id={requested_model_id} requested; "
                    f"resolved directly via provider_for_model()."
                ),
            )

        decisions = self._ranked_decisions(
            task_type=task_type, policy=effective_policy, require_tool_calling=require_tool_calling
        )
        if not decisions:
            raise ModelNotFoundError(
                f"No candidate models available for policy={effective_policy.value}, "
                f"task={task_type.value}"
                + (", require_tool_calling=True" if require_tool_calling else "")
                + "."
            )
        return decisions[0]

    def get_provider(self, provider_name: str) -> AIProvider:
        return self._provider_manager.get(provider_name)

    def ranked_candidates(
        self,
        *,
        task_type: TaskType = TaskType.GENERAL,
        policy: RoutingPolicy | None = None,
        require_tool_calling: bool = False,
    ) -> list[RoutingDecision]:
        """Public wrapper over `_ranked_decisions()` for callers outside
        this module (e.g. verification/judge.py, which needs the full
        ranked list to find a second, different model) — keeps
        `_ranked_decisions` itself private so existing internal callers
        and tests are untouched."""
        effective_policy = policy or self._default_policy
        return self._ranked_decisions(
            task_type=task_type, policy=effective_policy, require_tool_calling=require_tool_calling
        )

    async def route_with_failover(
        self,
        *,
        task_type: TaskType = TaskType.GENERAL,
        policy: RoutingPolicy | None = None,
        requested_model_id: str | None = None,
        require_tool_calling: bool = False,
    ) -> tuple[RoutingDecision, AIProvider]:
        effective_policy = policy or self._default_policy
        ranked = self._ranked_decisions(
            task_type=task_type, policy=effective_policy, require_tool_calling=require_tool_calling
        )

        if requested_model_id:
            explicit = self.route(
                task_type=task_type, policy=effective_policy, requested_model_id=requested_model_id
            )
            candidates = [explicit] + [d for d in ranked if d.model_id != explicit.model_id]
        else:
            candidates = ranked

        tried: list[str] = []
        for decision in candidates:
            try:
                provider = self._provider_manager.get(decision.provider_name)
            except ProviderUnavailableError:
                tried.append(decision.model_id)
                continue
            try:
                healthy = await provider.health_check()
            except Exception as exc:  # noqa: BLE001 - a misbehaving provider must not break failover
                logger.warning(
                    "health_check raised for provider=%s model=%s: %s",
                    decision.provider_name,
                    decision.model_id,
                    exc,
                )
                healthy = False
            if healthy:
                return decision, provider
            logger.warning(
                "Provider %s unhealthy for candidate model=%s, trying next candidate.",
                decision.provider_name,
                decision.model_id,
            )
            tried.append(decision.model_id)

        return await self._final_local_fallback(
            task_type, effective_policy, tried, require_tool_calling=require_tool_calling
        )

    async def _final_local_fallback(
        self,
        task_type: TaskType,
        policy: RoutingPolicy,
        tried: list[str],
        *,
        require_tool_calling: bool = False,
    ) -> tuple[RoutingDecision, AIProvider]:
        local_model_id = self._local_default_model_id(require_tool_calling=require_tool_calling)
        if local_model_id is None or not self._provider_manager.has_provider("local"):
            detail = (
                " (require_tool_calling=True and no local model supports tools)"
                if require_tool_calling
                else ""
            )
            raise ProviderUnavailableError(
                f"All providers unavailable and no local fallback model is registered.{detail}"
            )

        provider = self._provider_manager.get("local")
        try:
            healthy = await provider.health_check()
        except Exception as exc:  # noqa: BLE001 - health_check must never crash the request
            logger.warning("health_check raised for local fallback: %s", exc)
            healthy = False
        if not healthy:
            raise ProviderUnavailableError(
                "All providers unavailable, including the local fallback."
            )

        decision = RoutingDecision(
            provider_name="local",
            model_id=local_model_id,
            task_type=task_type,
            policy=policy,
            reason=(
                f"All ranked candidates failed health checks "
                f"({', '.join(tried) if tried else 'none registered'}); "
                f"fell back to guaranteed local model {local_model_id}."
            ),
        )
        return decision, provider

    def _local_default_model_id(self, *, require_tool_calling: bool = False) -> str | None:
        for model in self._model_registry:
            if model.provider == "local" and model.kind == "chat":
                if require_tool_calling and not model.supports_tool_calling:
                    continue
                return model.id
        return None

    def _candidate_models(
        self, policy: RoutingPolicy, *, require_tool_calling: bool = False
    ) -> list[ModelInfo]:
        chat_models = [
            model
            for model in self._model_registry
            if model.kind == "chat" and self._provider_manager.has_provider(model.provider)
        ]
        if require_tool_calling:
            chat_models = [model for model in chat_models if model.supports_tool_calling]
        if policy == RoutingPolicy.LOCAL_ONLY:
            return [model for model in chat_models if model.provider == "local"]
        return chat_models

    def _observed_p50s(self, candidates: list[ModelInfo]) -> dict[str, float | None]:
        if self._latency_tracker is None:
            return {model.id: None for model in candidates}
        return {model.id: self._latency_tracker.p50(model.provider, model.id) for model in candidates}

    def _latency_component(
        self, model: ModelInfo, p50: float | None, max_observed_p50: float, is_local: bool
    ) -> tuple[float, str]:
        """Returns (component in ~[0,1], note-for-reason-string).

        component is "higher is better" (mirrors 1 - cost_norm), so it
        drops straight into the same weighted-sum shape the cost term
        already uses. Falls back to the Phase 2 is_local heuristic per
        candidate when this model has no observed data yet — routing must
        never stall waiting for latency samples that don't exist.
        """
        if p50 is not None:
            latency_norm = (p50 / max_observed_p50) if max_observed_p50 > 0 else 0.0
            component = 1 - latency_norm
            sample_count = self._latency_tracker.sample_count(model.provider, model.id)  # type: ignore[union-attr]
            note = f"observed p50 latency-normalized {component:.2f} from {sample_count} samples"
            return component, note

        component = 1.0 if is_local else 0.0
        note = "falling back to local-preference heuristic, insufficient latency data"
        return component, note

    def _ranked_decisions(
        self, *, task_type: TaskType, policy: RoutingPolicy, require_tool_calling: bool = False
    ) -> list[RoutingDecision]:
        candidates = self._candidate_models(policy, require_tool_calling=require_tool_calling)
        if not candidates:
            return []

        cap_key = _capability_key(task_type)
        max_cost = max((_total_cost(model) for model in candidates), default=0.0)
        p50_by_model = self._observed_p50s(candidates)
        observed_p50s = [v for v in p50_by_model.values() if v is not None]
        max_observed_p50 = max(observed_p50s, default=0.0)

        scored: list[tuple[float, ModelInfo, str]] = []
        for model in candidates:
            capability = self._capability_for(model, cap_key)
            cost = _total_cost(model)
            cost_norm = (cost / max_cost) if max_cost > 0 else 0.0
            is_local = model.provider == "local"

            if policy == RoutingPolicy.MAX_QUALITY:
                score = capability
                reason = (
                    f"MAX_QUALITY policy: {model.id} scored {capability:.2f} "
                    f"{cap_key} capability for task={task_type.value}."
                )
            elif policy == RoutingPolicy.LOW_COST:
                score = -cost
                reason = (
                    f"LOW_COST policy: {model.id} costs {cost:.5f} per 1k tokens "
                    f"combined (local={is_local}) for task={task_type.value}."
                )
            elif policy == RoutingPolicy.LOW_LATENCY:
                latency_component, latency_note = self._latency_component(
                    model, p50_by_model[model.id], max_observed_p50, is_local
                )
                score = latency_component * 1_000.0 - cost
                reason = f"LOW_LATENCY policy: {model.id} {latency_note} for task={task_type.value}."
            elif policy == RoutingPolicy.LOCAL_ONLY:
                score = capability
                reason = (
                    f"LOCAL_ONLY policy: {model.id} scored {capability:.2f} "
                    f"{cap_key} capability for task={task_type.value} "
                    f"(cloud providers excluded)."
                )
            else:  # BALANCED
                latency_component, latency_note = self._latency_component(
                    model, p50_by_model[model.id], max_observed_p50, is_local
                )
                score = 0.5 * capability + 0.3 * (1 - cost_norm) + 0.2 * latency_component
                reason = (
                    f"BALANCED policy: {model.id} scored {score:.2f} "
                    f"(capability {capability:.2f}, cost-normalized {1 - cost_norm:.2f}, "
                    f"{latency_note}) for task={task_type.value}."
                )

            scored.append((score, model, reason))

        scored.sort(key=lambda item: (-item[0], item[1].id))

        return [
            RoutingDecision(
                provider_name=model.provider,
                model_id=model.id,
                task_type=task_type,
                policy=policy,
                reason=reason,
            )
            for _, model, reason in scored
        ]
