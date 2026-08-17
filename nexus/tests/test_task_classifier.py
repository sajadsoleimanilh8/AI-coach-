from __future__ import annotations

import pytest

from nexus.core.types import TaskType
from nexus.intelligence.task_classifier import TaskClassifier

_LONG_COMPLEX_QUERY = (
    "I understand the basic mechanism here, however I keep running into edge "
    "cases where the behavior seems inconsistent across environments, and on "
    "the other hand the documentation doesn't clarify what happens when "
    "multiple concurrent requests hit the same resource, so what determines "
    "precedence in that scenario given all of these constraints and caveats"
)

_CASES: list[tuple[str, TaskType]] = [
    ("I'm getting a stack trace when I run this function, help me debug the bug", TaskType.CODING),
    ("def parse_input(): please refactor this class to fix a syntax error", TaskType.CODING),
    ("Can you help me solve this equation for x?", TaskType.MATH),
    ("What's the derivative of this formula, and calculate the probability", TaskType.MATH),
    (
        "What is the latest research on this? Studies show conflicting results, can you compare them?",
        TaskType.RESEARCH,
    ),
    ("What is the state of the art here, do you have a citation?", TaskType.RESEARCH),
    (
        "Can you summarize this document, this pdf, for me?",
        TaskType.DOCUMENT_ANALYSIS,
    ),
    (
        "I've attached this pdf, can you look at the attached file and summarize this file?",
        TaskType.DOCUMENT_ANALYSIS,
    ),
    ("Can you describe this image and what's happening in the picture?", TaskType.VISION),
    ("That's a nice photo, can you take a screenshot of this and describe it?", TaskType.VISION),
    ("Can you transcribe this audio for me?", TaskType.AUDIO),
    ("I have a recording I need transcribed, can you transcribe it?", TaskType.AUDIO),
    ("Can you translate this phrase, tell me how to say it in Spanish?", TaskType.TRANSLATION),
    ("How do I say hello in French, can you translate it?", TaskType.TRANSLATION),
    ("I have a symptom of pain, and I need to see a doctor about my medication", TaskType.HEALTH),
    ("I've been so tired lately and my sleep has been terrible, should I see a doctor", TaskType.HEALTH),
    ("Can you build me a workout and training plan for cardio?", TaskType.FITNESS),
    ("How many reps of this exercise should I do in my workout?", TaskType.FITNESS),
    ("What formation should the team use against this opponent in the match?", TaskType.SPORTS),
    ("Analyze the possession stats and tactics from that match", TaskType.SPORTS),
    ("Can you analyze this data in the dataset and find the correlation?", TaskType.DATA_ANALYSIS),
    ("I have a csv file, can you load it into a dataframe?", TaskType.DATA_ANALYSIS),
    ("Help me plan my week with a clear roadmap", TaskType.PLANNING),
    ("What's the timeline and deadline for this schedule?", TaskType.PLANNING),
    ("Can you automate this and do this for me?", TaskType.AGENT_EXECUTION),
    (
        "Please execute the following steps and automate the whole workflow",
        TaskType.AGENT_EXECUTION,
    ),
    ("Can you show me my profile and my personal info?", TaskType.PRIVATE_PERSONAL),
    ("I need help accessing my account and my private files, my ssn", TaskType.PRIVATE_PERSONAL),
    (
        "Why does this happen? Can you explain in depth the trade-offs step by step?",
        TaskType.COMPLEX_REASONING,
    ),
    (_LONG_COMPLEX_QUERY, TaskType.COMPLEX_REASONING),
]


@pytest.mark.parametrize("query,expected_task_type", _CASES)
def test_classify_matches_expected_task_type(query: str, expected_task_type: TaskType) -> None:
    classifier = TaskClassifier()
    result = classifier.classify(query)

    assert result.task_type == expected_task_type
    assert result.matched_signals
    assert result.confidence >= 0.3
    assert result.reason


def test_ambiguous_short_query_falls_back_to_general() -> None:
    classifier = TaskClassifier()
    result = classifier.classify("Hey, how's it going today?")

    assert result.task_type == TaskType.GENERAL
    assert result.matched_signals == []
    assert "No keyword/pattern signals matched" in result.reason


def test_long_context_token_threshold_overrides_keyword_match() -> None:
    classifier = TaskClassifier(long_context_token_threshold=50)
    query = "please refactor this function and fix the bug " + ("padding word " * 60)

    result = classifier.classify(query)

    assert result.task_type == TaskType.LONG_CONTEXT
    assert result.confidence == 1.0
    assert "structural override" in result.reason


def test_long_context_threshold_considers_extra_char_count() -> None:
    classifier = TaskClassifier(long_context_token_threshold=100)
    short_query = "What do you think?"

    below_threshold = classifier.classify(short_query, extra_char_count=0)
    above_threshold = classifier.classify(short_query, extra_char_count=1000)

    assert below_threshold.task_type != TaskType.LONG_CONTEXT
    assert above_threshold.task_type == TaskType.LONG_CONTEXT


def test_min_confidence_forces_general_fallback() -> None:
    classifier = TaskClassifier(min_confidence=0.9)
    result = classifier.classify("Can you help me solve this equation?")

    assert result.task_type == TaskType.GENERAL
    assert result.matched_signals
    assert "defaulting to GENERAL" in result.reason
