from __future__ import annotations

from nexus.agents.base import Agent, AgentContext
from nexus.core.exceptions import ProviderUnavailableError
from nexus.core.types import Message, TaskType
from nexus.sports.coach import unavailable_reason
from nexus.sports.tactical import derive_findings


class SportsAgent(Agent):
    name = "sports"
    description = "Interprets football match data via the sports adapter, never estimating unavailable metrics."
    allowed_tools = ["files"]
    task_type = TaskType.SPORTS

    def system_prompt(self, context: AgentContext) -> str:
        return (
            "You are a football (soccer) tactical analysis agent. Interpret ONLY "
            "the derived findings and match data provided in context below — "
            "never estimate, infer, or fabricate a value for any metric "
            "explicitly listed as unavailable. Always state which metrics could "
            "not be measured and why, alongside your analysis. Any pre-match "
            "readiness figures in context were computed deterministically from "
            "a player's self-reported questionnaire: report them as given, "
            "never recompute them, and treat them as a performance-readiness "
            "estimate rather than a medical assessment."
        )

    async def _prematch_context(self, context: AgentContext) -> list[Message]:
        """Optional extra context: the player's latest pre-match readiness
        assessment, when a client and a string player identifier are both
        supplied via context.extra.
        """
        client = context.extra.get("prematch_health_client")
        player_id = context.extra.get("prematch_player_id")
        if client is None or not player_id:
            return []

        try:
            assessment = await client.get_latest(player_id)
        except ProviderUnavailableError:
            return []
        if assessment is None:
            return []

        return [
            Message(
                role="system",
                content=(
                    "Pre-match readiness for this player (already computed — "
                    "report as given, do not recalculate):\n\n"
                    f"{assessment.as_prompt_context()}"
                ),
            )
        ]

    async def prepare_context(self, context: AgentContext) -> list[Message]:
        prematch_messages = await self._prematch_context(context)

        adapter = context.extra.get("sports_adapter")
        match_id = context.extra.get("match_id")
        if adapter is None or match_id is None:
            return prematch_messages

        player_id = context.extra.get("player_id")
        analysis = (
            await adapter.get_player_analysis(match_id, player_id)
            if player_id is not None
            else await adapter.get_match_analysis(match_id)
        )
        findings = derive_findings(analysis)
        unavailable = [unavailable_reason(m) for m in analysis.unavailable]

        findings_text = (
            "\n".join(f"- [{f.area}] {f.assessment}: {f.explanation}" for f in findings)
            or "No findings derived yet."
        )
        unavailable_text = "\n".join(f"- {u}" for u in unavailable) or "None."
        content = (
            f"Match {match_id} tactical data (coverage={analysis.coverage:.0%}):\n\n"
            f"Findings:\n{findings_text}\n\n"
            f"Unavailable metrics:\n{unavailable_text}"
        )
        return [Message(role="system", content=content), *prematch_messages]
