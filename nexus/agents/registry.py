from __future__ import annotations

from nexus.agents.autonomous_research import AutonomousResearchAgent
from nexus.agents.base import Agent
from nexus.agents.coding import CodingAgent
from nexus.agents.data import DataAnalysisAgent
from nexus.agents.health import HealthAgent
from nexus.agents.orchestrator import OrchestratorAgent
from nexus.agents.planning import PlanningAgent
from nexus.agents.research import ResearchAgent
from nexus.agents.sports import SportsAgent
from nexus.core.exceptions import AgentNotFoundError

AGENT_REGISTRY: dict[str, type[Agent]] = {
    "research": ResearchAgent,
    "coding": CodingAgent,
    "data": DataAnalysisAgent,
    "planning": PlanningAgent,
    "health": HealthAgent,
    "sports": SportsAgent,
    "orchestrator": OrchestratorAgent,
    "autonomous_research": AutonomousResearchAgent,
}


def get_agent(name: str, *, enabled: list[str]) -> Agent:
    """Constructs an agent by name. Only names listed in settings.agents.enabled
    are constructible — an agent class existing in AGENT_REGISTRY is not
    enough by itself, mirroring how tools.enabled gates ToolRegistry."""
    if name not in enabled or name not in AGENT_REGISTRY:
        raise AgentNotFoundError(f"No enabled agent registered under name={name!r}.")
    return AGENT_REGISTRY[name]()
