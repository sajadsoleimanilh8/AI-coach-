from __future__ import annotations

from nexus.core.types import Usage
from nexus.verification.presenter import apply_verification
from nexus.verification.types import ConfidenceBand, VerificationReport


def _report(band: ConfidenceBand, score: float | None, notes: list[str] | None = None) -> VerificationReport:
    return VerificationReport(
        checks=[], score=score, band=band, summary="summary text",
        uncertainty_notes=notes or [], escalated=False, extra_usage=Usage(),
    )


def test_high_band_appends_a_single_short_line() -> None:
    report = _report(ConfidenceBand.HIGH, 0.9)
    result = apply_verification("The answer.", report)

    assert result.startswith("The answer.")
    added = result[len("The answer."):]
    assert added.count("\n") <= 3  # short footer, not a wall of text
    assert "High confidence" in added


def test_unverified_visibly_states_uncertainty_and_lists_notes() -> None:
    notes = ["No arithmetic could be checked.", "No RAG evidence was provided."]
    report = _report(ConfidenceBand.UNVERIFIED, None, notes)
    result = apply_verification("The answer.", report)

    assert "not been fully verified" in result.lower() or "unverified" in result.lower()
    for note in notes:
        assert note in result


def test_uncertain_visibly_states_uncertainty_and_lists_notes() -> None:
    notes = ["The arithmetic in the answer does not check out."]
    report = _report(ConfidenceBand.UNCERTAIN, 0.2, notes)
    result = apply_verification("The answer.", report)

    assert "not been fully verified" in result.lower()
    assert notes[0] in result


def test_unverified_with_no_notes_still_says_so_explicitly() -> None:
    report = _report(ConfidenceBand.UNVERIFIED, None, [])
    result = apply_verification("The answer.", report)

    assert "no checks produced a usable signal" in result.lower()


def test_medium_band_is_brief_but_mentions_unresolved_count() -> None:
    notes = ["One item unclear."]
    report = _report(ConfidenceBand.MEDIUM, 0.7, notes)
    result = apply_verification("The answer.", report)

    assert "Medium confidence" in result
    assert "1 item(s)" in result
    assert notes[0] not in result


def test_original_answer_text_is_always_preserved_as_a_prefix() -> None:
    for band, score in [
        (ConfidenceBand.HIGH, 0.9), (ConfidenceBand.MEDIUM, 0.7),
        (ConfidenceBand.LOW, 0.5), (ConfidenceBand.UNCERTAIN, 0.2),
        (ConfidenceBand.UNVERIFIED, None),
    ]:
        report = _report(band, score)
        result = apply_verification("Original answer text.", report)
        assert result.startswith("Original answer text.")
