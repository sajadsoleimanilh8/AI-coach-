from __future__ import annotations

from nexus.core.vector_store import RetrievedChunk
from nexus.verification.checks import (
    CheckContext,
    check_arithmetic,
    check_citation_support,
    check_internal_contradiction,
    check_refusal_consistency,
    check_unsupported_certainty,
)
from nexus.verification.types import CheckStatus


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(doc_id="d1", source_name="doc.txt", chunk_text=text, score=0.9, cosine_score=0.9)


# ---------------------------------------------------------------------- #
# check_arithmetic
# ---------------------------------------------------------------------- #

def test_arithmetic_pass() -> None:
    result = check_arithmetic(CheckContext(text="We know that 12 + 8 = 20, so the total works out."))
    assert result.status == CheckStatus.PASS
    assert result.evidence


def test_arithmetic_fail() -> None:
    result = check_arithmetic(CheckContext(text="We know that 12 + 8 = 25, so the total works out."))
    assert result.status == CheckStatus.FAIL
    assert result.evidence


def test_arithmetic_inconclusive_with_no_assertion() -> None:
    result = check_arithmetic(CheckContext(text="The weather today is pleasant and mild."))
    assert result.status == CheckStatus.INCONCLUSIVE


def test_arithmetic_percentage_pass() -> None:
    result = check_arithmetic(CheckContext(text="20% of 50 is 10, which matches expectations."))
    assert result.status == CheckStatus.PASS


def test_arithmetic_tolerates_small_rounding() -> None:
    result = check_arithmetic(CheckContext(text="1 / 3 is 0.33, roughly speaking."))
    assert result.status == CheckStatus.PASS


# ---------------------------------------------------------------------- #
# check_citation_support
# ---------------------------------------------------------------------- #

def test_citation_support_pass() -> None:
    evidence = [_chunk("Arsenal are a football club based in London, founded in 1886.")]
    result = check_citation_support(
        CheckContext(text="Arsenal are a football club based in London.", evidence=evidence)
    )
    assert result.status == CheckStatus.PASS


def test_citation_support_fail() -> None:
    evidence = [_chunk("Arsenal are a football club based in London, founded in 1886.")]
    result = check_citation_support(
        CheckContext(
            text="The moon landing was staged by lizard people wearing tinfoil hats.",
            evidence=evidence,
        )
    )
    assert result.status == CheckStatus.FAIL
    assert result.evidence


def test_citation_support_inconclusive_with_no_evidence() -> None:
    result = check_citation_support(CheckContext(text="Some claim with no evidence attached."))
    assert result.status == CheckStatus.INCONCLUSIVE


# ---------------------------------------------------------------------- #
# check_internal_contradiction
# ---------------------------------------------------------------------- #

def test_internal_contradiction_pass() -> None:
    result = check_internal_contradiction(
        CheckContext(text="The score is 5. Later in the match, the score is 5 again.")
    )
    assert result.status == CheckStatus.PASS


def test_internal_contradiction_fail_on_conflicting_numbers() -> None:
    result = check_internal_contradiction(
        CheckContext(text="The score is 5. Later in the match, the score is 9.")
    )
    assert result.status == CheckStatus.FAIL
    assert result.evidence


def test_internal_contradiction_fail_on_negation_pair() -> None:
    result = check_internal_contradiction(
        CheckContext(text="The recovery is good. However, the recovery is not good.")
    )
    assert result.status == CheckStatus.FAIL


def test_internal_contradiction_inconclusive_with_nothing_repeatable() -> None:
    result = check_internal_contradiction(CheckContext(text="This is a plain, unremarkable sentence."))
    assert result.status == CheckStatus.INCONCLUSIVE


# ---------------------------------------------------------------------- #
# check_unsupported_certainty
# ---------------------------------------------------------------------- #

def test_unsupported_certainty_pass_with_reasoning() -> None:
    result = check_unsupported_certainty(
        CheckContext(text="This is definitely correct because the underlying data confirms it.")
    )
    assert result.status == CheckStatus.PASS


def test_unsupported_certainty_fail_with_no_backing() -> None:
    result = check_unsupported_certainty(
        CheckContext(text="This is definitely true, without a doubt.")
    )
    assert result.status == CheckStatus.FAIL
    assert result.evidence


def test_unsupported_certainty_inconclusive_with_no_certainty_language() -> None:
    result = check_unsupported_certainty(CheckContext(text="This might be the case, roughly."))
    assert result.status == CheckStatus.INCONCLUSIVE


# ---------------------------------------------------------------------- #
# check_refusal_consistency
# ---------------------------------------------------------------------- #

def test_refusal_consistency_pass() -> None:
    result = check_refusal_consistency(
        CheckContext(text="I cannot determine the exact cause without more data.")
    )
    assert result.status == CheckStatus.PASS


def test_refusal_consistency_fail() -> None:
    result = check_refusal_consistency(
        CheckContext(
            text="I cannot determine the exact score. Actually, the exact score is 3-1."
        )
    )
    assert result.status == CheckStatus.FAIL
    assert result.evidence


def test_refusal_consistency_inconclusive_with_no_refusal() -> None:
    result = check_refusal_consistency(CheckContext(text="Everything here is straightforward."))
    assert result.status == CheckStatus.INCONCLUSIVE
