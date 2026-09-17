from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from nexus.agents.base import AgentContext
from nexus.agents.research import ResearchAgent
from nexus.core.types import Message

CoverageTier = Literal["known", "likely", "uncertain", "unknown"]

_SUB_QUESTION_SPLIT_RE = re.compile(r"[?;]|\band\b|\balso\b|,", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MIN_SUB_QUESTION_TOKENS = 2

# Fraction of a sub-question's content words that must appear in the
# gathered evidence for it to sit in each tier. Lexical overlap is a crude
# proxy for "is this supported", but it is DETERMINISTIC — and the decision
# to keep searching has to be the code's, not the model's, or the loop
# terminates whenever the model feels finished (principle 5).
_KNOWN_THRESHOLD = 0.7
_LIKELY_THRESHOLD = 0.4
_UNCERTAIN_THRESHOLD = 0.15

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "were", "what", "when", "where", "which", "who", "why", "with",
}


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower())) - _STOPWORDS


def split_sub_questions(goal: str) -> list[str]:
    """Deterministic decomposition of a goal into the parts coverage is
    tracked against. Deliberately not a model call: the set of things that
    must be answered should not change between rounds just because a model
    re-read the goal."""
    parts = [part.strip() for part in _SUB_QUESTION_SPLIT_RE.split(goal)]
    sub_questions = [
        part for part in parts if part and len(_tokenize(part)) >= _MIN_SUB_QUESTION_TOKENS
    ]
    return sub_questions or ([goal.strip()] if goal.strip() else [])


@dataclass
class SubQuestionCoverage:
    sub_question: str
    tier: CoverageTier
    overlap: float


@dataclass
class Coverage:
    items: list[SubQuestionCoverage] = field(default_factory=list)

    def by_tier(self, tier: CoverageTier) -> list[SubQuestionCoverage]:
        return [item for item in self.items if item.tier == tier]

    @property
    def unresolved(self) -> list[SubQuestionCoverage]:
        return [item for item in self.items if item.tier in ("uncertain", "unknown")]

    @property
    def fully_covered(self) -> bool:
        return not self.unresolved


def assess_coverage(sub_questions: list[str], evidence_texts: list[str]) -> Coverage:
    """Pure function: how well does the gathered evidence cover each
    sub-question? Zero supporting evidence lands in "unknown", which is
    what keeps the loop running."""
    evidence_tokens = _tokenize(" ".join(evidence_texts))

    items: list[SubQuestionCoverage] = []
    for sub_question in sub_questions:
        question_tokens = _tokenize(sub_question)
        if not question_tokens:
            items.append(
                SubQuestionCoverage(sub_question=sub_question, tier="unknown", overlap=0.0)
            )
            continue

        overlap = len(question_tokens & evidence_tokens) / len(question_tokens)
        if overlap >= _KNOWN_THRESHOLD:
            tier: CoverageTier = "known"
        elif overlap >= _LIKELY_THRESHOLD:
            tier = "likely"
        elif overlap >= _UNCERTAIN_THRESHOLD:
            tier = "uncertain"
        else:
            tier = "unknown"
        items.append(
            SubQuestionCoverage(sub_question=sub_question, tier=tier, overlap=overlap)
        )

    return Coverage(items=items)


def format_coverage(coverage: Coverage) -> str:
    """The Known / Likely / Uncertain / Unknown breakdown appended to every
    answer. Computed from the evidence rather than asked of the model, so
    the four tiers mean the same thing on every run."""
    sections: list[str] = []
    labels: list[tuple[CoverageTier, str]] = [
        ("known", "Known (directly supported by gathered evidence)"),
        ("likely", "Likely (partially supported; reasonable inference)"),
        ("uncertain", "Uncertain (weak or fragmentary support)"),
        ("unknown", "Unknown (no supporting evidence was found)"),
    ]

    for tier, label in labels:
        items = coverage.by_tier(tier)
        if not items:
            continue
        lines = "\n".join(f"  - {item.sub_question}" for item in items)
        sections.append(f"{label}:\n{lines}")

    if not sections:
        return ""
    return "Research coverage\n\n" + "\n\n".join(sections)


class AutonomousResearchAgent(ResearchAgent):
    """Iterative research: search, assess coverage, identify what remains
    unknown, search again — up to max_rounds.

    The division of labour is the point. The LLM proposes what to search
    next; the code decides whether there is anything left worth searching
    and when to stop. Inherits verify_output=True from ResearchAgent.
    """

    name = "autonomous_research"
    description = (
        "Researches a question over multiple rounds, re-searching whatever remains "
        "unsupported after each round, and reports what is Known / Likely / Uncertain / Unknown."
    )
    max_rounds = 3

    def system_prompt(self, context: AgentContext) -> str:
        return (
            super().system_prompt(context)
            + "\n\nYou work in rounds. After each round you will be told which parts of the "
            "goal still have no supporting evidence, and you should search specifically for "
            "those rather than repeating searches that already succeeded. Do not claim "
            "coverage you do not have — an honest 'no source found' is the correct outcome "
            "for a sub-question the sources genuinely do not address."
        )

    def next_round_message(
        self, *, round_index: int, evidence_texts: list[str], context: AgentContext
    ) -> Message | None:
        """Returns None to stop. Called by AgentRuntime after each round."""
        coverage = assess_coverage(split_sub_questions(context.goal), evidence_texts)
        context.extra["research_coverage"] = coverage

        if coverage.fully_covered:
            return None

        outstanding = "\n".join(f"- {item.sub_question}" for item in coverage.unresolved)
        return Message(
            role="user",
            content=(
                f"Round {round_index + 1} found no adequate evidence for these parts of the "
                f"goal:\n{outstanding}\n\nSearch specifically for these now. If a source "
                f"genuinely does not exist, say so rather than substituting a weaker one."
            ),
        )

    def finalize_answer(
        self, answer: str, *, evidence_texts: list[str], context: AgentContext
    ) -> str:
        coverage = context.extra.get("research_coverage") or assess_coverage(
            split_sub_questions(context.goal), evidence_texts
        )
        block = format_coverage(coverage)
        return f"{answer}\n\n---\n\n{block}" if block else answer
