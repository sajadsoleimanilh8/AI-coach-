from __future__ import annotations

import re
from dataclasses import dataclass, field

from nexus.core.types import TaskType
from nexus.logging_setup.logger import get_logger

logger = get_logger("intelligence.task_classifier")

# Same rough heuristic OllamaRuntime and the cloud providers already use
# for context-window checks — kept here too so the classifier stays a pure,
# synchronous, dependency-free function (no tokenizer, no provider call).
_CHARS_PER_TOKEN_ESTIMATE = 4

_DEFAULT_MIN_CONFIDENCE = 0.3
_DEFAULT_LONG_CONTEXT_TOKEN_THRESHOLD = 6000

# confidence = min(1.0, 0.3 + 0.15 * matched_count): one matched signal is
# "plausible" (0.45), four or more is "confident" (1.0) — simple, monotonic,
# and easy to reason about when tuning min_confidence in nexus.yaml.
_CONFIDENCE_BASE = 0.3
_CONFIDENCE_PER_SIGNAL = 0.15

_COMPLEX_REASONING_CONJUNCTIONS = ("however", "although", "trade-off", "trade off", "on the other hand")
_COMPLEX_REASONING_MIN_WORDS = 40


def _default_signal_map() -> dict[TaskType, list[str]]:
    """Built-in keyword/regex signals per TaskType.

    Every entry is compiled as a case-insensitive regex (re.search), so
    plain words work as substrings and operators can supply real regexes
    via nexus.yaml's routing.classification.extra_signals without any code
    change. LONG_CONTEXT and GENERAL are intentionally absent: LONG_CONTEXT
    is a structural (token-count) override, not a topical one, and GENERAL
    is the no-signal fallback.
    """
    return {
        TaskType.CODING: [
            r"\bfunction\b",
            r"\bbug\b",
            r"\berror\b",
            r"stack trace",
            r"\brefactor\b",
            r"\bcompile\b",
            r"\bsyntax\b",
            r"def ",
            r"class ",
            r"\bimport\s+\w+",
            r"\bpython\b",
            r"\bjavascript\b",
            r"\btypescript\b",
            r"\bgolang\b",
            r"\brust\b",
            r"\bexception\b",
            r"\bnull ?pointer\b",
        ],
        TaskType.MATH: [
            r"\bequation\b",
            r"\bsolve\b",
            r"\bderivative\b",
            r"\bintegral\b",
            r"\bprobability\b",
            r"\bstatistics\b",
            r"\bcalculate\b",
            r"\bformula\b",
        ],
        TaskType.RESEARCH: [
            r"\blatest\b",
            r"\bresearch\b",
            r"studies show",
            r"\bcompare\b",
            r"what is the state of",
            r"\bcitation\b",
            r"\bcite\b",
        ],
        TaskType.DOCUMENT_ANALYSIS: [
            r"this document",
            r"this pdf",
            r"summarize this file",
            r"\battached\b",
        ],
        TaskType.VISION: [
            r"this image",
            r"in the picture",
            r"\bscreenshot\b",
            r"\bphoto\b",
        ],
        TaskType.AUDIO: [
            r"this audio",
            r"\btranscribe\b",
            r"\brecording\b",
        ],
        TaskType.TRANSLATION: [
            r"\btranslate\b",
            r"in french",
            r"in spanish",
            r"in german",
            r"from english to",
            r"into english",
        ],
        TaskType.HEALTH: [
            r"\bsymptom",
            r"\bdiagnos",
            r"\btired\b",
            r"\bsleep\b",
            r"\bpain\b",
            r"\bdoctor\b",
            r"\bmedication\b",
        ],
        TaskType.FITNESS: [
            r"\bworkout\b",
            r"\bexercise\b",
            r"training plan",
            r"\breps\b",
            r"\bcardio\b",
        ],
        TaskType.SPORTS: [
            r"\bmatch\b",
            r"\bformation\b",
            r"\bpossession\b",
            r"\btactics\b",
            r"\bopponent\b",
            r"\bteam\b",
        ],
        TaskType.DATA_ANALYSIS: [
            r"\bdataset\b",
            r"\bcsv\b",
            r"analyze this data",
            r"\bcorrelation\b",
            r"\bdataframe\b",
        ],
        TaskType.PLANNING: [
            r"plan my",
            r"\bschedule\b",
            r"\broadmap\b",
            r"\btimeline\b",
            r"\bdeadline\b",
        ],
        TaskType.AGENT_EXECUTION: [
            r"do this for me",
            r"\bautomate\b",
            r"execute the following steps",
        ],
        TaskType.PRIVATE_PERSONAL: [
            r"my profile",
            r"my personal info",
            r"my ssn",
            r"my account",
            r"my private",
        ],
        TaskType.COMPLEX_REASONING: [
            r"\bwhy\b",
            r"explain in depth",
            r"trade-offs?",
            r"step by step",
        ],
    }


def merge_signal_map(
    extra_signals: dict[TaskType, list[str]] | None,
) -> dict[TaskType, list[str]]:
    """Built-in defaults + config-supplied additions (additive, never replaces)."""
    merged = _default_signal_map()
    if not extra_signals:
        return merged
    for task_type, extra_patterns in extra_signals.items():
        merged[task_type] = merged.get(task_type, []) + list(extra_patterns)
    return merged


def build_signal_map_from_names(
    extra_signals_by_name: dict[str, list[str]] | None,
) -> dict[TaskType, list[str]]:
    """Config-facing wrapper: nexus.yaml keys extra_signals by TaskType.value
    strings (e.g. "coding") since YAML has no enum keys. Unknown names are
    dropped with a warning instead of failing boot over a config typo.
    """
    converted: dict[TaskType, list[str]] = {}
    for name, patterns in (extra_signals_by_name or {}).items():
        try:
            converted[TaskType(name)] = list(patterns)
        except ValueError:
            logger.warning("Ignoring unknown task type in routing.classification.extra_signals: %s", name)
    return merge_signal_map(converted)


@dataclass
class TaskClassification:
    task_type: TaskType
    confidence: float
    matched_signals: list[str] = field(default_factory=list)
    reason: str = ""


class TaskClassifier:
    """Deterministic keyword/pattern-based classifier.

    Scores every TaskType by counting matched signals against the query
    text, picks the highest-scoring type above `min_confidence`, and falls
    back to TaskType.GENERAL (with a low confidence reflecting "no strong
    signal") when nothing clears the threshold.
    """

    def __init__(
        self,
        signal_map: dict[TaskType, list[str]] | None = None,
        *,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
        long_context_token_threshold: int = _DEFAULT_LONG_CONTEXT_TOKEN_THRESHOLD,
    ) -> None:
        source = _default_signal_map() if signal_map is None else signal_map
        self._compiled: dict[TaskType, list[tuple[str, re.Pattern[str]]]] = {
            task_type: [(raw, re.compile(raw, re.IGNORECASE)) for raw in patterns]
            for task_type, patterns in source.items()
        }
        self._min_confidence = min_confidence
        self._long_context_token_threshold = long_context_token_threshold

    def classify(self, query: str, *, extra_char_count: int = 0) -> TaskClassification:
        # Structural override checked first and unconditionally: an
        # oversized prompt needs long-context routing regardless of what
        # it's topically about, so it must win over any keyword match.
        estimated_tokens = (len(query) + max(extra_char_count, 0)) // _CHARS_PER_TOKEN_ESTIMATE
        if estimated_tokens > self._long_context_token_threshold:
            return TaskClassification(
                task_type=TaskType.LONG_CONTEXT,
                confidence=1.0,
                matched_signals=[f"structural:estimated_tokens={estimated_tokens}"],
                reason=(
                    f"Estimated {estimated_tokens} tokens (query+context) exceeds "
                    f"long_context_token_threshold={self._long_context_token_threshold}; "
                    f"structural override to LONG_CONTEXT regardless of keyword signals."
                ),
            )

        scores: dict[TaskType, list[str]] = {}
        for task_type, patterns in self._compiled.items():
            matches = [f"keyword:{raw}" for raw, compiled in patterns if compiled.search(query)]
            if task_type == TaskType.COMPLEX_REASONING:
                matches += self._complex_reasoning_structural_signals(query)
            if matches:
                scores[task_type] = matches

        if not scores:
            return TaskClassification(
                task_type=TaskType.GENERAL,
                confidence=_CONFIDENCE_BASE,
                matched_signals=[],
                reason="No keyword/pattern signals matched; defaulting to GENERAL.",
            )

        best_type, best_matches = max(scores.items(), key=lambda item: (len(item[1]), item[0].value))
        confidence = min(1.0, _CONFIDENCE_BASE + _CONFIDENCE_PER_SIGNAL * len(best_matches))

        if confidence < self._min_confidence:
            return TaskClassification(
                task_type=TaskType.GENERAL,
                confidence=confidence,
                matched_signals=best_matches,
                reason=(
                    f"Best candidate {best_type.value} only reached confidence "
                    f"{confidence:.2f} (< min_confidence={self._min_confidence}); "
                    f"defaulting to GENERAL."
                ),
            )

        return TaskClassification(
            task_type=best_type,
            confidence=confidence,
            matched_signals=best_matches,
            reason=(
                f"Classified as {best_type.value} via {len(best_matches)} matched "
                f"signal(s): {', '.join(best_matches)}."
            ),
        )

    @staticmethod
    def _complex_reasoning_structural_signals(query: str) -> list[str]:
        signals: list[str] = []
        if query.count("?") >= 2:
            signals.append("structural:multiple_question_marks")
        if len(query.split()) >= _COMPLEX_REASONING_MIN_WORDS:
            signals.append("structural:long_query")
        lowered = query.lower()
        if any(conjunction in lowered for conjunction in _COMPLEX_REASONING_CONJUNCTIONS):
            signals.append("structural:conjunction")
        return signals
