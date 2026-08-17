from __future__ import annotations

from nexus.agents.base import Agent, AgentContext
from nexus.core.types import Message, RoutingPolicy, TaskType
from nexus.health.safety import check_output


class HealthAgent(Agent):
    name = "health"
    description = (
        "Interprets deterministically-detected health patterns — never diagnoses, "
        "always defers concerning findings to a professional."
    )
    allowed_tools = ["files"]
    task_type = TaskType.HEALTH
    default_policy = RoutingPolicy.LOCAL_ONLY
    output_filter = staticmethod(check_output)

    def system_prompt(self, context: AgentContext) -> str:
        return (
            "You are a health-pattern interpretation assistant. You will be given "
            "structured, already-computed patterns, a scorecard, and a "
            "data-sufficiency rating below — interpret ONLY what is provided, and "
            "do not use any tool to look up outside medical information. Never "
            "diagnose: do not name a disease, syndrome, or condition as a "
            "conclusion, and do not suggest one is likely. Never recommend, "
            "adjust, or discuss medication or dosage. Always state sample sizes "
            "and confidence alongside any claim you make. For any pattern rated "
            "'significant', or whenever data_sufficiency is not 'adequate', "
            "explicitly recommend professional evaluation rather than "
            "reassurance — never tell the user they don't need to see a doctor."
        )

    async def prepare_context(self, context: AgentContext) -> list[Message]:
        analyzer = context.extra.get("health_analyzer")
        if analyzer is None:
            return []
        analysis = await analyzer.analyze(context.user_id)

        if analysis.data_sufficiency == "none":
            content = "No health signals have been recorded for this user yet."
        else:
            pattern_lines = (
                "\n".join(
                    f"- [{p.severity}] {p.name} (dimensions: "
                    f"{', '.join(p.dimensions_involved)}, confidence={p.confidence:.0%}, "
                    f"sample_size={p.sample_size}): {p.explanation}"
                    for p in analysis.patterns
                )
                or "No patterns detected."
            )
            scorecard_lines = (
                "\n".join(f"- {group}: {value:.2f}" for group, value in analysis.scorecard.items())
                or "No scorecard data yet."
            )
            content = (
                f"Data sufficiency: {analysis.data_sufficiency}\n\n"
                f"Detected patterns:\n{pattern_lines}\n\n"
                f"Scorecard (0..1, higher = better):\n{scorecard_lines}"
            )
        return [Message(role="system", content=content)]
