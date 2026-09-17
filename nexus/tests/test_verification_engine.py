from __future__ import annotations

import pytest

from nexus.core.types import TaskType, Usage
from nexus.verification.engine import VerificationEngine
from nexus.verification.judge import JudgeVerdict
from nexus.verification.types import CheckResult, CheckStatus, ConfidenceBand


class _FakeFactChecker:
    def __init__(self, results: list[CheckResult], usage: Usage | None = None) -> None:
        self._results = results
        self._usage = usage or Usage()
        self.called = False

    async def run(self, answer: str, evidence):
        self.called = True
        return self._results, self._usage


class _FakeJudge:
    def __init__(self, verdict: JudgeVerdict | None) -> None:
        self._verdict = verdict
        self.called = False

    async def judge(self, **kwargs):
        self.called = True
        return self._verdict


def _engine(*, fact_checker=None, judge=None, escalate_below=0.65, enable_fact_check=False, enable_judge=True) -> VerificationEngine:
    return VerificationEngine(
        None,  # router is unused directly by the engine — fact_checker/judge own their own routing
        fact_checker or _FakeFactChecker([]),
        judge or _FakeJudge(None),
        escalate_below=escalate_below,
        enable_fact_check=enable_fact_check,
        enable_judge=enable_judge,
    )


@pytest.mark.asyncio
async def test_escalation_triggers_below_threshold() -> None:
    judge = _FakeJudge(JudgeVerdict(agrees=True, disagreement_summary=None, judge_model="m2", usage=Usage()))
    engine = _engine(judge=judge, escalate_below=0.65, enable_judge=True)

    report = await engine.verify(
        question="q", answer="2 + 2 = 5.", model_id="m1", task_type=TaskType.GENERAL
    )

    assert report.escalated is True
    assert judge.called is True


@pytest.mark.asyncio
async def test_no_escalation_above_threshold() -> None:
    judge = _FakeJudge(None)
    engine = _engine(judge=judge, escalate_below=0.65, enable_judge=True)

    report = await engine.verify(
        question="q", answer="2 + 2 = 4.", model_id="m1", task_type=TaskType.GENERAL
    )

    assert report.escalated is False
    assert judge.called is False
    assert report.band == ConfidenceBand.HIGH


@pytest.mark.asyncio
async def test_unverified_score_still_triggers_escalation() -> None:
    judge = _FakeJudge(JudgeVerdict(agrees=True, disagreement_summary=None, judge_model="m2", usage=Usage()))
    engine = _engine(judge=judge, escalate_below=0.65, enable_judge=True)

    report = await engine.verify(
        question="q", answer="This is a plain sentence with nothing checkable in it.",
        model_id="m1", task_type=TaskType.GENERAL,
    )

    assert report.escalated is True
    assert judge.called is True


@pytest.mark.asyncio
async def test_enable_judge_false_never_escalates_regardless_of_score() -> None:
    judge = _FakeJudge(JudgeVerdict(agrees=True, disagreement_summary=None, judge_model="m2", usage=Usage()))
    engine = _engine(judge=judge, escalate_below=0.65, enable_judge=False)

    report = await engine.verify(
        question="q", answer="2 + 2 = 5.", model_id="m1", task_type=TaskType.GENERAL
    )

    assert report.escalated is False
    assert judge.called is False


@pytest.mark.asyncio
async def test_unavailable_judge_adds_inconclusive_never_a_pass() -> None:
    judge = _FakeJudge(None)
    engine = _engine(judge=judge, escalate_below=0.65, enable_judge=True)

    report = await engine.verify(
        question="q", answer="2 + 2 = 5.", model_id="m1", task_type=TaskType.GENERAL
    )

    judge_checks = [c for c in report.checks if c.name == "multi_model_judge"]
    assert len(judge_checks) == 1
    assert judge_checks[0].status == CheckStatus.INCONCLUSIVE


@pytest.mark.asyncio
async def test_disagreeing_judge_adds_fail_check_and_uncertainty_note() -> None:
    judge = _FakeJudge(
        JudgeVerdict(agrees=False, disagreement_summary="Wrong date given.", judge_model="m2", usage=Usage())
    )
    engine = _engine(judge=judge, escalate_below=0.65, enable_judge=True)

    report = await engine.verify(
        question="q", answer="2 + 2 = 5.", model_id="m1", task_type=TaskType.GENERAL
    )

    judge_checks = [c for c in report.checks if c.name == "multi_model_judge"]
    assert judge_checks[0].status == CheckStatus.FAIL
    assert any("Wrong date given." in note for note in report.uncertainty_notes)


@pytest.mark.asyncio
async def test_agreeing_judge_adds_a_pass_check() -> None:
    judge = _FakeJudge(
        JudgeVerdict(agrees=True, disagreement_summary=None, judge_model="m2", usage=Usage())
    )
    engine = _engine(judge=judge, escalate_below=0.65, enable_judge=True)

    report = await engine.verify(
        question="q", answer="2 + 2 = 5.", model_id="m1", task_type=TaskType.GENERAL
    )

    judge_checks = [c for c in report.checks if c.name == "multi_model_judge"]
    assert judge_checks[0].status == CheckStatus.PASS


@pytest.mark.asyncio
async def test_extra_usage_accumulates_from_fact_check_and_judge() -> None:
    fact_checker = _FakeFactChecker(
        [CheckResult(name="fact_check:x", status=CheckStatus.PASS, weight=1.0, detail="ok")],
        usage=Usage(prompt_tokens=100, completion_tokens=50),
    )
    judge = _FakeJudge(
        JudgeVerdict(agrees=False, disagreement_summary="nope", judge_model="m2",
                     usage=Usage(prompt_tokens=20, completion_tokens=10))
    )
    engine = _engine(fact_checker=fact_checker, judge=judge, escalate_below=0.99, enable_fact_check=True, enable_judge=True)

    report = await engine.verify(
        question="q", answer="2 + 2 = 5.", model_id="m1", task_type=TaskType.GENERAL, evidence=None
    )

    assert fact_checker.called is True
    assert judge.called is True
    assert report.extra_usage.prompt_tokens == 120
    assert report.extra_usage.completion_tokens == 60


@pytest.mark.asyncio
async def test_deterministic_checks_always_run_even_with_fact_check_and_judge_off() -> None:
    engine = _engine(enable_fact_check=False, enable_judge=False)

    report = await engine.verify(
        question="q", answer="2 + 2 = 5.", model_id="m1", task_type=TaskType.GENERAL
    )

    assert any(c.name == "arithmetic" for c in report.checks)
    assert report.score == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_uncertainty_notes_include_every_fail_and_inconclusive() -> None:
    engine = _engine(enable_fact_check=False, enable_judge=False)

    report = await engine.verify(
        question="q", answer="2 + 2 = 5.", model_id="m1", task_type=TaskType.GENERAL
    )

    fail_and_inconclusive = [c for c in report.checks if c.status != CheckStatus.PASS]
    assert len(report.uncertainty_notes) == len(fail_and_inconclusive)
