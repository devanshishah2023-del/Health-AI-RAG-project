"""Tests for the emergency safety guardrail."""

from app import guardrails


def test_chest_pain_triggers():
    result = guardrails.evaluate("I have sudden chest pain and feel awful")
    assert result.triggered is True
    assert result.guardrail_type == "emergency_escalation"
    assert any("chest pain" in t.lower() for t in result.matched_terms)


def test_severe_shortness_of_breath_triggers():
    assert guardrails.check_high_risk_query(
        "I have severe shortness of breath right now"
    )


def test_fainting_triggers():
    assert guardrails.check_high_risk_query("I keep fainting today")


def test_routine_question_does_not_trigger():
    result = guardrails.evaluate(
        "What foods are low in sodium for someone with HFpEF?"
    )
    assert result.triggered is False
    assert result.matched_terms == []


def test_negated_symptom_does_not_trigger():
    # "no chest pain" should not be treated as an emergency.
    result = guardrails.evaluate("I have no chest pain, just a mild cough")
    assert result.triggered is False


def test_message_is_present_when_triggered():
    result = guardrails.evaluate("I think I'm having a heart attack")
    assert result.triggered is True
    assert result.message and "emergency" in result.message.lower()
