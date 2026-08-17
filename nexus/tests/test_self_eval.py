from __future__ import annotations

import pytest

from nexus.core.providers import AIProvider
from nexus.core.router import RoutingDecision
from nexus.core.types import (
    GenerationChunk,
    GenerationResult,
    ModelInfo,
    RoutingPolicy,
    TaskType,
    Usage,
)
from nexus.intelligence.self_eval import SelfEvaluator
from nexus.verification.types import CheckResult, CheckStatus, ConfidenceBand, VerificationReport


class _ScriptedProvider(AIProvider):
    name = "fake"

    def __init__(self, content: str) -> None:
        self._content = content
        self.call_count = 0

    async def generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        self.call_count += 1
        return GenerationResult(
            content=self._content, model_used=model_id, provider_name=self.name,
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        )

    async def stream_generate(self, messages, *, model_id, temperature=0.7, max_tokens=None, tools=None):
        yield GenerationChunk(delta="", done=True, usage=Usage())

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
        return (
            RoutingDecision(
                provider_name=self._provider.name, model_id="fake-model",
                task_type=kwargs.get("task_type", TaskType.GENERAL),
                policy=RoutingPolicy.BALANCED, reason="fake",
            ),
            self._provider,
        )


def _report(score: float = 0.9, band: ConfidenceBand = ConfidenceBand.HIGH) -> VerificationReport:
    return VerificationReport(
        checks=[CheckResult(name="c", status=CheckStatus.PASS, weight=1.0, detail="ok")],
        score=score, band=band, summary="verified", uncertainty_notes=["existing note"],
    )


@pytest.mark.asyncio
async def test_produces_a_critique() -> None:
    provider = _ScriptedProvider(
        '{"completeness": 0.4, "addressed_question": false, "gaps": ["ignored the second half"]}'
    )
    evaluator = SelfEvaluator(_FakeRouter(provider), enabled=True)

    result = await evaluator.evaluate(question="Compare A and B.", answer="A is nice.")

    assert result is not None
    assert result.completeness == pytest.approx(0.4)
    assert result.addressed_question is False
    assert result.gaps == ["ignored the second half"]
    assert result.usage.total_tokens > 0


@pytest.mark.asyncio
async def test_disabled_makes_no_provider_call() -> None:
    provider = _ScriptedProvider('{"completeness": 1.0, "addressed_question": true, "gaps": []}')
    evaluator = SelfEvaluator(_FakeRouter(provider), enabled=False)

    assert await evaluator.evaluate(question="q", answer="a") is None
    assert provider.call_count == 0


@pytest.mark.asyncio
async def test_empty_answer_is_not_critiqued() -> None:
    provider = _ScriptedProvider("{}")
    evaluator = SelfEvaluator(_FakeRouter(provider), enabled=True)

    assert await evaluator.evaluate(question="q", answer="   ") is None
    assert provider.call_count == 0


@pytest.mark.asyncio
async def test_completeness_is_clamped_to_the_unit_interval() -> None:
    provider = _ScriptedProvider('{"completeness": 5.0, "addressed_question": true, "gaps": []}')
    evaluator = SelfEvaluator(_FakeRouter(provider), enabled=True)

    result = await evaluator.evaluate(question="q", answer="a")

    assert result.completeness == 1.0


@pytest.mark.asyncio
async def test_unparseable_critique_is_a_missing_measurement_not_a_bad_answer() -> None:
    """Scoring an unparseable response as completeness 0.0 would flag good
    answers for retraining, so it is marked unparsed and ignored instead."""
    provider = _ScriptedProvider("I think the answer was pretty good, honestly.")
    evaluator = SelfEvaluator(_FakeRouter(provider), enabled=True)

    result = await evaluator.evaluate(question="q", answer="a")

    assert result.parsed is False
    assert evaluator.should_mine(result) is False


@pytest.mark.asyncio
async def test_never_overrides_a_verification_verdict() -> None:
    """THE constraint. A model grading its own homework must not be able to
    move the number that says how well-supported its answer is."""
    provider = _ScriptedProvider('{"completeness": 0.1, "addressed_question": false, "gaps": ["everything"]}')
    evaluator = SelfEvaluator(_FakeRouter(provider), enabled=True)
    self_eval = await evaluator.evaluate(question="q", answer="a")
    original = _report(score=0.9, band=ConfidenceBand.HIGH)

    annotated = evaluator.annotate_report(original, self_eval)

    assert annotated.score == 0.9
    assert annotated.band is ConfidenceBand.HIGH
    assert annotated.checks == original.checks


@pytest.mark.asyncio
async def test_cannot_raise_a_low_verification_score_either() -> None:
    provider = _ScriptedProvider('{"completeness": 1.0, "addressed_question": true, "gaps": []}')
    evaluator = SelfEvaluator(_FakeRouter(provider), enabled=True)
    self_eval = await evaluator.evaluate(question="q", answer="a")
    original = _report(score=0.2, band=ConfidenceBand.UNCERTAIN)

    annotated = evaluator.annotate_report(original, self_eval)

    assert annotated.score == 0.2
    assert annotated.band is ConfidenceBand.UNCERTAIN


@pytest.mark.asyncio
async def test_findings_are_added_as_additive_notes() -> None:
    provider = _ScriptedProvider(
        '{"completeness": 0.3, "addressed_question": false, "gaps": ["no numbers given"]}'
    )
    evaluator = SelfEvaluator(_FakeRouter(provider), enabled=True)
    self_eval = await evaluator.evaluate(question="q", answer="a")

    annotated = evaluator.annotate_report(_report(), self_eval)

    assert "existing note" in annotated.uncertainty_notes
    assert any("did not directly address" in n for n in annotated.uncertainty_notes)
    assert any("no numbers given" in n for n in annotated.uncertainty_notes)


@pytest.mark.asyncio
async def test_a_good_self_eval_adds_nothing() -> None:
    provider = _ScriptedProvider('{"completeness": 0.95, "addressed_question": true, "gaps": []}')
    evaluator = SelfEvaluator(_FakeRouter(provider), enabled=True)
    self_eval = await evaluator.evaluate(question="q", answer="a")
    original = _report()

    annotated = evaluator.annotate_report(original, self_eval)

    assert annotated.uncertainty_notes == original.uncertainty_notes


def test_annotate_with_no_self_eval_returns_the_report_unchanged() -> None:
    evaluator = SelfEvaluator(_FakeRouter(_ScriptedProvider("{}")), enabled=True)
    original = _report()

    assert evaluator.annotate_report(original, None) is original


@pytest.mark.asyncio
async def test_low_scores_are_flagged_for_dataset_mining() -> None:
    """This is how the improvement loop closes: a weak answer becomes an
    eval_failure training candidate."""
    provider = _ScriptedProvider('{"completeness": 0.2, "addressed_question": true, "gaps": ["thin"]}')
    evaluator = SelfEvaluator(_FakeRouter(provider), enabled=True)

    self_eval = await evaluator.evaluate(question="q", answer="a")

    assert self_eval.is_low_scoring is True
    assert evaluator.should_mine(self_eval) is True


@pytest.mark.asyncio
async def test_an_answer_that_missed_the_question_is_mined_even_at_high_completeness() -> None:
    provider = _ScriptedProvider('{"completeness": 0.95, "addressed_question": false, "gaps": []}')
    evaluator = SelfEvaluator(_FakeRouter(provider), enabled=True)

    self_eval = await evaluator.evaluate(question="q", answer="a")

    assert evaluator.should_mine(self_eval) is True


@pytest.mark.asyncio
async def test_a_strong_answer_is_not_mined() -> None:
    provider = _ScriptedProvider('{"completeness": 0.9, "addressed_question": true, "gaps": []}')
    evaluator = SelfEvaluator(_FakeRouter(provider), enabled=True)

    assert evaluator.should_mine(await evaluator.evaluate(question="q", answer="a")) is False


def test_should_mine_with_no_self_eval_is_false() -> None:
    evaluator = SelfEvaluator(_FakeRouter(_ScriptedProvider("{}")), enabled=True)

    assert evaluator.should_mine(None) is False
