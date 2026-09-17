from __future__ import annotations

from nexus.verification.types import ConfidenceBand, VerificationReport

_BAND_LABELS: dict[ConfidenceBand, str] = {
    ConfidenceBand.HIGH: "High confidence",
    ConfidenceBand.MEDIUM: "Medium confidence",
    ConfidenceBand.LOW: "Low confidence",
    ConfidenceBand.UNCERTAIN: "Uncertain",
    ConfidenceBand.UNVERIFIED: "Unverified",
}

def apply_verification(answer: str, report: VerificationReport) -> str:
    """Appends an honest, compact confidence footer to the answer. HIGH
    gets a single short line (principle 1's honesty is about never
    overclaiming, not about padding every response with a wall of text);
    MEDIUM adds a short count; LOW/UNCERTAIN/UNVERIFIED all visibly
    disclose uncertainty and list the specific unresolved notes, not just
    a number."""
    label = _BAND_LABELS[report.band]
    score_text = f" ({report.score:.0%})" if report.score is not None else ""

    if report.band == ConfidenceBand.HIGH:
        return f"{answer}\n\n---\n{label}{score_text}."

    if report.band == ConfidenceBand.MEDIUM:
        footer = f"\n\n---\n{label}{score_text}."
        if report.uncertainty_notes:
            footer += f" {len(report.uncertainty_notes)} item(s) could not be fully confirmed."
        return answer + footer

    lines = [f"\n\n---\n{label}{score_text} — this answer has NOT been fully verified."]
    if report.uncertainty_notes:
        lines.append("Specifically unsure about:")
        lines.extend(f"  - {note}" for note in report.uncertainty_notes)
    else:
        lines.append("No checks produced a usable signal for this answer.")
    return answer + "\n".join(lines)
