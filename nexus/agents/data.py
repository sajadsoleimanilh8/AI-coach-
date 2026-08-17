from __future__ import annotations

from nexus.agents.base import Agent, AgentContext
from nexus.core.types import TaskType


class DataAnalysisAgent(Agent):
    name = "data"
    description = "Analyzes data by inspecting and computing over it directly."
    allowed_tools = ["python", "database", "files"]
    task_type = TaskType.DATA_ANALYSIS

    def system_prompt(self, context: AgentContext) -> str:
        return (
            "You are a data analysis agent. Before analyzing anything, inspect the "
            "data's actual shape (columns, types, row counts, ranges) using the "
            "database, files, or python tools — never assume a schema or "
            "distribution you have not actually observed. Compute statistics rather "
            "than estimating them. Always state the sample size alongside any "
            "statistic you report, since a statistic without its sample size is not "
            "trustworthy."
        )
