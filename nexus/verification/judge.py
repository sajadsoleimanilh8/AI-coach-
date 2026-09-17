from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from nexus.core.exceptions import ProviderUnavailableError
from nexus.core.router import ModelRouter
from nexus.core.types import Message, TaskType, Usage

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_JUDGE_SYSTEM_PROMPT = (
    "You are an independent reviewer checking another AI's answer to a question. Judge whether "
    "the answer is accurate and reasonable given the question, using your own knowledge and "
    "judgment — you were not involved in producing it. Respond with ONLY a JSON object, no prose: "
    '{"agrees": true|false, "disagreement_summary": "<short specific reason if agrees is false, '
    'otherwise null>"}'
)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


@dataclass
class JudgeVerdict:
    agrees: bool
    disagreement_summary: str | None
    judge_model: str
    usage: Usage


class MultiModelJudge:
    """Consults a DIFFERENT model than the one that produced the answer,
    to catch single-model failure modes (a model is a poor judge of its
    own mistakes)."""

    def __init__(self, router: ModelRouter) -> None:
        self._router = router

    async def judge(
        self, *, question: str, answer: str, original_model_id: str, task_type: TaskType
    ) -> JudgeVerdict | None:
        candidates = self._router.ranked_candidates(task_type=task_type)

        chosen: tuple[Any, Any] | None = None
        for decision in candidates:
            if decision.model_id == original_model_id:
                continue
            try:
                provider = self._router.get_provider(decision.provider_name)
            except ProviderUnavailableError:
                continue
            try:
                healthy = await provider.health_check()
            except Exception:  # noqa: BLE001 - a misbehaving provider must not break verification
                healthy = False
            if healthy:
                chosen = (decision, provider)
                break

        if chosen is None:
            # Legitimate "could not escalate" — e.g. a local-only
            # deployment with a single model. The caller (VerificationEngine)
            # must record this as an INCONCLUSIVE check, never treat
            # single-model output as if it had been cross-validated
            # (principle 1).
            return None

        decision, provider = chosen
        result = await provider.generate(
            [
                Message(role="system", content=_JUDGE_SYSTEM_PROMPT),
                Message(
                    role="user",
                    content=f"Question: {question}\n\nAnswer to review:\n{answer}",
                ),
            ],
            model_id=decision.model_id,
            temperature=0.0,
        )

        parsed = _parse_json_object(result.content)
        if parsed is None:
            # Malformed judge output is never treated as silent agreement
            # — it's surfaced as a disagreement so the engine records that
            # the judge ran but its verdict couldn't be trusted.
            return JudgeVerdict(
                agrees=False,
                disagreement_summary="Judge response could not be parsed.",
                judge_model=decision.model_id,
                usage=result.usage,
            )

        agrees = bool(parsed.get("agrees", False))
        summary = parsed.get("disagreement_summary")
        return JudgeVerdict(
            agrees=agrees,
            disagreement_summary=str(summary) if summary else None,
            judge_model=decision.model_id,
            usage=result.usage,
        )
