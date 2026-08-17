from __future__ import annotations

import json
from typing import Any

import pytest

from nexus.core.providers import AIProvider
from nexus.core.router import RoutingDecision
from nexus.core.types import GenerationChunk, GenerationResult, ModelInfo, RoutingPolicy, Usage
from nexus.core.vector_store import RetrievedChunk
from nexus.verification.fact_checker import ExtractedClaim, FactChecker
from nexus.verification.types import CheckStatus


class _FakeProvider(AIProvider):
    name = "fake"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.call_count = 0

    async def generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        self.call_count += 1
        content = self._responses.pop(0)
        return GenerationResult(
            content=content, model_used=model_id, provider_name=self.name,
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        )

    async def stream_generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        yield GenerationChunk(delta="unused", done=True, usage=Usage())

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> bool:
        return True

    def count_tokens(self, text: str, *, model_id: str) -> int:
        return len(text)


class _FakeRouter:
    def __init__(self, provider: AIProvider) -> None:
        self._provider = provider

    async def route_with_failover(self, **kwargs):
        decision = RoutingDecision(
            provider_name=self._provider.name, model_id="fake-model",
            task_type=kwargs.get("task_type"), policy=RoutingPolicy.BALANCED, reason="fake",
        )
        return decision, self._provider


def _raising_provider() -> AIProvider:
    class _Raises(AIProvider):
        name = "should-not-be-called"

        async def generate(self, *args: Any, **kwargs: Any):
            raise AssertionError("provider.generate() must not be called")

        async def stream_generate(self, *args: Any, **kwargs: Any):
            raise AssertionError("must not be called")
            yield  # pragma: no cover

        async def list_models(self) -> list[ModelInfo]:
            return []

        async def health_check(self) -> bool:
            return True

        def count_tokens(self, text: str, *, model_id: str) -> int:
            return 0

    return _Raises()


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(doc_id="d1", source_name="doc.txt", chunk_text=text, score=0.9, cosine_score=0.9)


@pytest.mark.asyncio
async def test_extract_claims_parses_json_response() -> None:
    response = json.dumps(
        [
            {"text": "Arsenal are based in London.", "kind": "factual"},
            {"text": "2 + 2 = 4.", "kind": "numeric"},
            {"text": "This is a nice club.", "kind": "opinion"},
        ]
    )
    provider = _FakeProvider([response])
    checker = FactChecker(_FakeRouter(provider), max_claims=6)

    claims = await checker.extract_claims("Some answer text.")

    assert len(claims) == 3
    assert claims[0] == ExtractedClaim(text="Arsenal are based in London.", is_checkable=True, kind="factual")
    assert claims[2].is_checkable is False
    assert claims[2].kind == "opinion"


@pytest.mark.asyncio
async def test_extract_claims_handles_malformed_json_gracefully() -> None:
    provider = _FakeProvider(["not json at all"])
    checker = FactChecker(_FakeRouter(provider), max_claims=6)

    claims = await checker.extract_claims("Some answer.")

    assert claims == []


@pytest.mark.asyncio
async def test_check_claims_with_no_evidence_is_inconclusive_and_makes_no_llm_call() -> None:
    checker = FactChecker(_FakeRouter(_raising_provider()))
    claims = [ExtractedClaim(text="Arsenal are based in London.", is_checkable=True, kind="factual")]

    results = await checker.check_claims(claims, [])

    assert len(results) == 1
    assert results[0].status == CheckStatus.INCONCLUSIVE
    assert "no evidence" in results[0].detail.lower()


@pytest.mark.asyncio
async def test_check_claims_unsupported_with_evidence_present_is_fail() -> None:
    response = json.dumps(
        [{"claim": "Arsenal are based in Paris.", "verdict": "unsupported", "reason": "Evidence says London."}]
    )
    provider = _FakeProvider([response])
    checker = FactChecker(_FakeRouter(provider))
    claims = [ExtractedClaim(text="Arsenal are based in Paris.", is_checkable=True, kind="factual")]
    evidence = [_chunk("Arsenal are a football club based in London.")]

    results = await checker.check_claims(claims, evidence)

    assert len(results) == 1
    assert results[0].status == CheckStatus.FAIL


@pytest.mark.asyncio
async def test_check_claims_supported_is_pass() -> None:
    response = json.dumps(
        [{"claim": "Arsenal are based in London.", "verdict": "supported", "reason": "Matches evidence."}]
    )
    provider = _FakeProvider([response])
    checker = FactChecker(_FakeRouter(provider))
    claims = [ExtractedClaim(text="Arsenal are based in London.", is_checkable=True, kind="factual")]
    evidence = [_chunk("Arsenal are a football club based in London.")]

    results = await checker.check_claims(claims, evidence)

    assert results[0].status == CheckStatus.PASS


@pytest.mark.asyncio
async def test_check_claims_contradicted_is_fail() -> None:
    response = json.dumps(
        [{"claim": "Arsenal are based in Berlin.", "verdict": "contradicted", "reason": "Evidence says London."}]
    )
    provider = _FakeProvider([response])
    checker = FactChecker(_FakeRouter(provider))
    claims = [ExtractedClaim(text="Arsenal are based in Berlin.", is_checkable=True, kind="factual")]
    evidence = [_chunk("Arsenal are a football club based in London.")]

    results = await checker.check_claims(claims, evidence)

    assert results[0].status == CheckStatus.FAIL


@pytest.mark.asyncio
async def test_check_claims_malformed_response_is_inconclusive_not_dropped() -> None:
    provider = _FakeProvider(["not valid json"])
    checker = FactChecker(_FakeRouter(provider))
    claims = [ExtractedClaim(text="Arsenal are based in London.", is_checkable=True, kind="factual")]
    evidence = [_chunk("Arsenal are a football club based in London.")]

    results = await checker.check_claims(claims, evidence)

    assert len(results) == 1
    assert results[0].status == CheckStatus.INCONCLUSIVE


@pytest.mark.asyncio
async def test_non_checkable_claims_are_never_sent_for_checking() -> None:
    checker = FactChecker(_FakeRouter(_raising_provider()))
    claims = [ExtractedClaim(text="I like this club a lot.", is_checkable=False, kind="opinion")]

    results = await checker.check_claims(claims, [_chunk("Some evidence.")])

    assert results == []


@pytest.mark.asyncio
async def test_run_combines_usage_from_extract_and_check() -> None:
    extract_response = json.dumps([{"text": "Arsenal are based in London.", "kind": "factual"}])
    check_response = json.dumps(
        [{"claim": "Arsenal are based in London.", "verdict": "supported", "reason": "ok"}]
    )
    provider = _FakeProvider([extract_response, check_response])
    checker = FactChecker(_FakeRouter(provider))
    evidence = [_chunk("Arsenal are a football club based in London.")]

    results, usage = await checker.run("Arsenal are based in London.", evidence)

    assert provider.call_count == 2
    assert usage.prompt_tokens == 20
    assert usage.completion_tokens == 10
    assert results[0].status == CheckStatus.PASS


@pytest.mark.asyncio
async def test_run_with_no_checkable_claims_makes_only_the_extract_call() -> None:
    extract_response = json.dumps([{"text": "I really like this.", "kind": "opinion"}])
    provider = _FakeProvider([extract_response])
    checker = FactChecker(_FakeRouter(provider))

    results, usage = await checker.run("I really like this.", [_chunk("Some evidence.")])

    assert provider.call_count == 1
    assert results == []
    assert usage.prompt_tokens == 10
