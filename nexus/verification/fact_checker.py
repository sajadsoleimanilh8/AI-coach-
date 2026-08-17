from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from nexus.core.router import ModelRouter
from nexus.core.types import Message, TaskType, Usage
from nexus.core.vector_store import RetrievedChunk
from nexus.verification.types import CheckResult, CheckStatus

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_MAX_NAME_CHARS = 40

_EXTRACT_SYSTEM_PROMPT = (
    "Extract up to {max_claims} discrete, individually checkable factual or numeric claims from "
    "the answer text below. Ignore hedged opinions, instructions, and vague statements. "
    "Respond with ONLY a JSON array, no prose, in this exact shape: "
    '[{{"text": "<claim text, verbatim or near-verbatim from the answer>", '
    '"kind": "numeric"|"factual"|"opinion"|"instruction"}}, ...]'
)

_CHECK_SYSTEM_PROMPT = (
    "You are a fact-checker. For each claim below, judge it against ONLY the evidence chunks "
    "provided — never use outside knowledge, and never invent support that isn't in the evidence. "
    'For each claim, respond with a verdict of "supported" (evidence confirms it), "contradicted" '
    '(evidence explicitly contradicts it), or "unsupported" (the evidence doesn\'t address it '
    "either way). Respond with ONLY a JSON array, no prose, in this exact shape: "
    '[{"claim": "<claim text>", "verdict": "supported"|"contradicted"|"unsupported", '
    '"reason": "<short reason>"}, ...]'
)


@dataclass
class ExtractedClaim:
    text: str
    is_checkable: bool
    kind: Literal["numeric", "factual", "opinion", "instruction"]


def _parse_json_array(text: str) -> list[Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_ARRAY_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class FactChecker:
    """Uses an LLM ONLY to (a) extract discrete checkable claims and (b)
    judge each claim against provided evidence. It never invents evidence:
    a claim with no supporting retrieved chunk is reported as UNSUPPORTED,
    not judged true from model priors."""

    def __init__(
        self, router: ModelRouter, *, max_claims: int = 6, pinned_model_id: str | None = None
    ) -> None:
        self._router = router
        self._max_claims = max_claims
        self._pinned_model_id = pinned_model_id

    async def extract_claims(self, answer: str) -> list[ExtractedClaim]:
        claims, _usage = await self._extract(answer)
        return claims

    async def check_claims(
        self, claims: list[ExtractedClaim], evidence: list[RetrievedChunk]
    ) -> list[CheckResult]:
        results, _usage = await self._check(claims, evidence)
        return results

    async def run(
        self, answer: str, evidence: list[RetrievedChunk] | None
    ) -> tuple[list[CheckResult], Usage]:
        """Extracts then checks claims in one call, returning the combined
        token usage. FactChecker is typically a long-lived singleton
        shared across concurrent requests, so usage is threaded through
        return values rather than mutable instance state, which would
        """
        claims, extract_usage = await self._extract(answer)
        check_results, check_usage = await self._check(claims, evidence or [])
        combined = Usage(
            prompt_tokens=extract_usage.prompt_tokens + check_usage.prompt_tokens,
            completion_tokens=extract_usage.completion_tokens + check_usage.completion_tokens,
        )
        return check_results, combined

    async def _extract(self, answer: str) -> tuple[list[ExtractedClaim], Usage]:
        decision, provider = await self._router.route_with_failover(
            task_type=TaskType.RESEARCH, requested_model_id=self._pinned_model_id
        )
        result = await provider.generate(
            [
                Message(
                    role="system",
                    content=_EXTRACT_SYSTEM_PROMPT.format(max_claims=self._max_claims),
                ),
                Message(role="user", content=answer),
            ],
            model_id=decision.model_id,
            temperature=0.0,
        )

        parsed = _parse_json_array(result.content)
        if parsed is None:
            return [], result.usage

        claims: list[ExtractedClaim] = []
        for item in parsed[: self._max_claims]:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            kind = str(item.get("kind", "factual")).lower()
            if kind not in ("numeric", "factual", "opinion", "instruction"):
                kind = "factual"
            claims.append(
                ExtractedClaim(text=text, is_checkable=kind in ("numeric", "factual"), kind=kind)
            )
        return claims, result.usage

    async def _check(
        self, claims: list[ExtractedClaim], evidence: list[RetrievedChunk]
    ) -> tuple[list[CheckResult], Usage]:
        checkable = [c for c in claims if c.is_checkable]
        if not checkable:
            return [], Usage()

        if not evidence:
            return [
                CheckResult(
                    name=f"fact_check:{c.text[:_MAX_NAME_CHARS]}",
                    status=CheckStatus.INCONCLUSIVE,
                    weight=1.0,
                    detail="No evidence was provided to check this claim against.",
                )
                for c in checkable
            ], Usage()

        decision, provider = await self._router.route_with_failover(
            task_type=TaskType.RESEARCH, requested_model_id=self._pinned_model_id
        )
        evidence_text = "\n".join(f"[{i + 1}] {chunk.chunk_text}" for i, chunk in enumerate(evidence))
        claims_text = "\n".join(f"- {c.text}" for c in checkable)
        result = await provider.generate(
            [
                Message(role="system", content=_CHECK_SYSTEM_PROMPT),
                Message(
                    role="user",
                    content=f"Evidence:\n{evidence_text}\n\nClaims:\n{claims_text}",
                ),
            ],
            model_id=decision.model_id,
            temperature=0.0,
        )

        verdicts = _parse_json_array(result.content)
        if verdicts is None:
            return [
                CheckResult(
                    name=f"fact_check:{c.text[:_MAX_NAME_CHARS]}",
                    status=CheckStatus.INCONCLUSIVE,
                    weight=1.0,
                    detail="Fact-checker response could not be parsed.",
                )
                for c in checkable
            ], result.usage

        by_claim_text = {
            str(v.get("claim", "")).strip(): v for v in verdicts if isinstance(v, dict)
        }
        results: list[CheckResult] = []
        for claim in checkable:
            name = f"fact_check:{claim.text[:_MAX_NAME_CHARS]}"
            entry = by_claim_text.get(claim.text.strip())
            if entry is None:
                results.append(
                    CheckResult(
                        name=name, status=CheckStatus.INCONCLUSIVE, weight=1.0,
                        detail="No verdict was returned for this claim.",
                    )
                )
                continue

            verdict = str(entry.get("verdict", "")).lower()
            reason = str(entry.get("reason", ""))
            if verdict == "supported":
                results.append(
                    CheckResult(
                        name=name, status=CheckStatus.PASS, weight=1.0,
                        detail=reason or "Supported by the provided evidence.",
                        evidence=[claim.text],
                    )
                )
            elif verdict == "contradicted":
                results.append(
                    CheckResult(
                        name=name, status=CheckStatus.FAIL, weight=1.0,
                        detail=reason or "Contradicted by the provided evidence.",
                        evidence=[claim.text],
                    )
                )
            else:
                results.append(
                    CheckResult(
                        name=name, status=CheckStatus.FAIL, weight=1.0,
                        detail=reason or "The provided evidence does not support this claim.",
                        evidence=[claim.text],
                    )
                )
        return results, result.usage
