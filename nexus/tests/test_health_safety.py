from __future__ import annotations

import pytest

from nexus.health.safety import (
    MEDICAL_DISCLAIMER,
    URGENT_REFERRAL_MESSAGE,
    check_input,
    check_output,
)

RED_FLAG_INPUTS = [
    "I've been having severe chest pain for the last hour.",
    "I can't breathe properly and it's getting worse.",
    "I just fainted in the kitchen and feel dizzy.",
    "I don't want to be here anymore, I've been thinking about suicide.",
    "I want to kill myself.",
    "This is the worst pain of my life, it came on suddenly.",
    "I'm bleeding heavily and it won't stop.",
    "My face is drooping on one side and my speech is slurred.",
    "I'm pregnant and bleeding heavily.",
]

SAFE_INPUTS = [
    "How has my sleep quality trended this month?",
    "I felt a bit more tired than usual after training yesterday.",
    "What does my recovery score mean?",
    "I have a mild headache, is that reflected in my stress data?",
]

PROHIBITED_OUTPUTS = [
    "Based on these patterns, you have an anxiety disorder.",
    "This is clearly diabetes given your fatigue levels.",
    "You are suffering from depression.",
    "You should take 400mg of ibuprofen twice a day.",
    "Stop taking your prescribed medication immediately.",
    "Take 2 acetaminophen tablets every 4 hours.",
    "There's no need to see a doctor about this.",
    "You don't need to see a doctor, this will resolve on its own.",
]

SAFE_OUTPUTS = [
    "Your recovery has been below baseline for the last 5 days, which is worth monitoring.",
    "This pattern (high activity, low recovery) is rated 'notable' at 72% confidence.",
    "Your stress signal has been trending upward over the recent window.",
    "Given the significant pattern detected, please consult a healthcare professional.",
]


@pytest.mark.parametrize("text", RED_FLAG_INPUTS)
def test_red_flag_inputs_short_circuit_to_referral(text: str) -> None:
    verdict = check_input(text)

    assert verdict.allowed is False
    assert verdict.requires_professional_referral is True
    assert verdict.rewritten_text == URGENT_REFERRAL_MESSAGE
    assert len(verdict.triggered_rules) >= 1


@pytest.mark.parametrize("text", SAFE_INPUTS)
def test_ordinary_inputs_pass_through(text: str) -> None:
    verdict = check_input(text)

    assert verdict.allowed is True
    assert verdict.rewritten_text is None
    assert verdict.triggered_rules == []
    assert verdict.requires_professional_referral is False


def test_red_flag_input_names_specific_rules() -> None:
    verdict = check_input("I've had chest pain and shortness of breath for an hour.")

    assert "chest_pain" in verdict.triggered_rules
    assert "breathing_difficulty" in verdict.triggered_rules


@pytest.mark.parametrize("text", PROHIBITED_OUTPUTS)
def test_prohibited_outputs_are_blocked_and_rewritten(text: str) -> None:
    verdict = check_output(text)

    assert verdict.allowed is False
    assert verdict.rewritten_text is not None
    assert verdict.rewritten_text != text
    assert verdict.requires_professional_referral is True
    assert len(verdict.triggered_rules) >= 1


@pytest.mark.parametrize("text", SAFE_OUTPUTS)
def test_safe_outputs_pass_through_unblocked(text: str) -> None:
    verdict = check_output(text)

    assert verdict.allowed is True
    assert verdict.rewritten_text is None
    assert verdict.triggered_rules == []


def test_diagnostic_claim_is_specifically_flagged() -> None:
    verdict = check_output("You have an anxiety disorder based on these readings.")
    assert "diagnostic_claim" in verdict.triggered_rules


def test_medication_dosage_is_specifically_flagged() -> None:
    verdict = check_output("Take 500mg of ibuprofen every six hours.")
    assert "dosage_with_drug" in verdict.triggered_rules or "medication_instruction" in verdict.triggered_rules


def test_discourages_care_is_specifically_flagged() -> None:
    verdict = check_output("There's no need to see a doctor about this.")
    assert "discourages_care" in verdict.triggered_rules


def test_blocked_output_rewrite_uses_supplied_patterns_summary() -> None:
    summary = "Detected pattern: recovery_debt (notable, 80% confidence, n=6)."
    verdict = check_output("You have chronic fatigue syndrome.", patterns_summary=summary)

    assert verdict.allowed is False
    assert summary in verdict.rewritten_text


def test_blocked_output_rewrite_falls_back_without_patterns_summary() -> None:
    verdict = check_output("You have chronic fatigue syndrome.")

    assert verdict.allowed is False
    assert verdict.rewritten_text is not None
    assert MEDICAL_DISCLAIMER in verdict.rewritten_text


def test_disclaimer_is_a_stable_module_constant() -> None:
    assert isinstance(MEDICAL_DISCLAIMER, str)
    assert len(MEDICAL_DISCLAIMER) > 0
    assert "not medical advice" in MEDICAL_DISCLAIMER.lower()
