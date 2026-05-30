from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

# Curated emergency-symptom patterns. Word boundaries avoid matching substrings
# inside unrelated words.
EMERGENCY_PATTERNS: List[str] = [
    r"chest\s+pain",
    r"chest\s+pressure",
    r"chest\s+tightness",
    r"tightness\s+in\s+(my\s+)?chest",
    r"pressure\s+in\s+(my\s+)?chest",
    r"severe\s+shortness\s+of\s+breath",
    r"sudden\s+shortness\s+of\s+breath",
    r"can'?t\s+breathe",
    r"cannot\s+breathe",
    r"trouble\s+breathing",
    r"struggling\s+to\s+breathe",
    r"gasping\s+for\s+(air|breath)",
    r"faint(ing|ed)?",
    r"passed\s+out",
    r"pass(ing)?\s+out",
    r"black(ed|ing)?\s+out",
    r"lost\s+consciousness",
    r"unconscious",
    r"severe\s+dizziness",
    r"severely\s+dizzy",
    r"sudden\s+dizziness",
    r"coughing\s+up\s+(blood|pink|foam)",
    r"(blue|bluish)\s+lips",
    r"heart\s+(is\s+)?racing",
    r"heart\s+palpitations",
    r"irregular\s+heartbeat",
    r"slurred\s+speech",
    r"numbness\s+on\s+one\s+side",
    r"weakness\s+on\s+one\s+side",
    r"face\s+drooping",
    r"stroke",
    r"heart\s+attack",
]

# Negation cues that, when they immediately precede a matched symptom, suggest
# the user is *reporting the absence* of the symptom.
_NEGATION_CUES = (
    "no",
    "not",
    "without",
    "don't have",
    "dont have",
    "do not have",
    "didn't have",
    "didnt have",
    "never had",
    "free of",
)

SAFETY_MESSAGE = (
    "This may be a medical emergency. The symptoms you described can be signs of "
    "a serious, potentially life-threatening condition. Please stop using this "
    "tool and seek emergency help now: call your local emergency number "
    "(911 in Canada and the U.S.) or go to the nearest emergency department. "
    "If you are with someone who is unresponsive or struggling to breathe, call "
    "for emergency help immediately. This educational assistant cannot assess or "
    "treat emergencies and must not be used to decide against seeking urgent care."
)

GUARDRAIL_TYPE = "emergency_escalation"

_COMPILED = [re.compile(p, re.IGNORECASE) for p in EMERGENCY_PATTERNS]


@dataclass
class GuardrailResult:
    triggered: bool
    guardrail_type: Optional[str]
    matched_terms: List[str]
    message: Optional[str]


def _is_negated(text: str, match_start: int) -> bool:
    """Heuristic: was the matched symptom negated just before it?"""
    window = text[max(0, match_start - 30) : match_start].lower()
    return any(cue in window for cue in _NEGATION_CUES)


def evaluate(question: str) -> GuardrailResult:
    """Evaluate a question for emergency content."""
    text = question or ""
    matched: List[str] = []
    for pattern in _COMPILED:
        for m in pattern.finditer(text):
            if _is_negated(text, m.start()):
                continue
            matched.append(m.group(0).strip())

    if matched:
        # De-duplicate while preserving order.
        seen = set()
        unique = [t for t in matched if not (t.lower() in seen or seen.add(t.lower()))]
        return GuardrailResult(
            triggered=True,
            guardrail_type=GUARDRAIL_TYPE,
            matched_terms=unique,
            message=SAFETY_MESSAGE,
        )
    return GuardrailResult(
        triggered=False, guardrail_type=None, matched_terms=[], message=None
    )


# Backwards-compatible convenience wrapper.
def check_high_risk_query(question: str) -> bool:
    return evaluate(question).triggered
