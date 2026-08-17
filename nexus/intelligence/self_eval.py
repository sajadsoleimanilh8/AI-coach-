from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace

from nexus.core.router import ModelRouter
from nexus.core.types import Message, TaskType, Usage
from nexus.logging_setup.logger import get_logger
from nexus.verification.types import VerificationReport

logger = get_logger("intelligence.self_eval")

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

DEFAULT_LOW_SCORE_THRESHOLD = 0.6

_SYSTEM_PROMPT = (
    "Critique the answer below against the question it was given. Judge two things only: "
    "COMPLETENESS (did it cover everything the question asked for?) and whether it ACTUALLY "
    "ADDRESSED the question rather than an adjacent one. Do NOT judge factual correctness — "
    "that is checked separately and is not your job here. Be strict: an answer that is "
    "fluent but sidesteps the question scores low. Respond with ONLY a JSON object, no prose: "
    '{"completeness": <float 0-1>, "addressed_question": <true|false>, '
    '"gaps": ["<what was missing>", ...]}'
)


@dataclass
class SelfEvalResult:
    completeness: float
    addressed_question: bool
    gaps: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    parsed: bool = True

    @property
    def is_low_scoring(self) -> bool:
        return self.completeness < DEFAULT_LOW_SCORE_THRESHOLD or not self.addressed_question


def _parse(text: str) -> tuple[float, bool, list[str]] | None:
    payload = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(text)
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = None

    if not isinstance(payload, dict):
        return None

    try:
        completeness = float(payload.get("completeness", 0.0))
    except (TypeError, ValueError):
        return None

    completeness = min(1.0, max(0.0, completeness))
    addressed = bool(payload.get("addressed_question", False))
    gaps = [str(g) for g in payload.get("gaps", []) if str(g).strip()]
    return completeness, addressed, gaps


class SelfEvaluator:
    """Asks the model to critique its own answer for completeness."""

    def __init__(
        self,
        router: ModelRouter,
        *,
        enabled: bool = False,
        low_score_threshold: float = DEFAULT_LOW_SCORE_THRESHOLD,
    ) -> None:
        self._router = router
        self._enabled = enabled
        self._low_score_threshold = low_score_threshold

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def evaluate(self, *, question: str, answer: str) -> SelfEvalResult | None:
        if not self._enabled or not answer.strip():
            return None

        decision, provider = await self._router.route_with_failover(
            task_type=TaskType.COMPLEX_REASONING
        )
        result = await provider.generate(
            [
                Message(role="system", content=_SYSTEM_PROMPT),
                Message(
                    role="user", content=f"Question:\n{question}\n\nAnswer:\n{answer}"
                ),
            ],
            model_id=decision.model_id,
            temperature=0.0,
        )

        parsed = _parse(result.content)
        if parsed is None:
            logger.warning("self-eval response could not be parsed; treating as no signal")
            return SelfEvalResult(
                completeness=0.0,
                addressed_question=True,
                gaps=[],
                usage=result.usage,
                parsed=False,
            )

        completeness, addressed, gaps = parsed
        return SelfEvalResult(
            completeness=completeness,
            addressed_question=addressed,
            gaps=gaps,
            usage=result.usage,
        )

    def annotate_report(
        self, report: VerificationReport, self_eval: SelfEvalResult | None
    ) -> VerificationReport:
        """Adds the critique to uncertainty_notes and leaves score, band,
        and checks EXACTLY as verification computed them. This asymmetry is
        the point: self-eval can tell the reader something, but it cannot
        move the number that says how well-supported the answer is."""
        if self_eval is None or not self_eval.parsed:
            return report

        notes = list(report.uncertainty_notes)
        if not self_eval.addressed_question:
            notes.append(
                "Self-evaluation: the model judged that this answer did not directly "
                "address the question asked."
            )
        if self_eval.completeness < self._low_score_threshold:
            notes.append(
                f"Self-evaluation: completeness rated {self_eval.completeness:.2f} "
                f"(below {self._low_score_threshold:.2f})."
            )
        for gap in self_eval.gaps:
            notes.append(f"Self-evaluation gap: {gap}")

        return replace(report, uncertainty_notes=notes)

    def should_mine(self, self_eval: SelfEvalResult | None) -> bool:
        """A low self-eval is the cheapest available signal that an answer
        was weak, which makes it a good candidate for the eval_failure
        training source — that is how the improvement loop closes."""
        if self_eval is None or not self_eval.parsed:
            return False
        return (
            self_eval.completeness < self._low_score_threshold
            or not self_eval.addressed_question
        )
