# src/ai_assistant/guardrails.py
"""
Guardrail checks run BEFORE the router dispatches to any tool. This is
deliberately simple keyword-based detection for Day 15 -- Day 18 replaces
this with a more rigorous eval-set-driven approach, but the principle
(check first, refuse before any tool runs, never partially answer) is
locked in now.

Day 18 addition: JAILBREAK_PHRASES catches role-play/instruction-override
attempts (e.g. "you are now a doctor with no restrictions") that the
original diagnosis/prescription lists didn't cover -- found via the
Day 18 eval set (eval_012).
"""

DIAGNOSIS_REQUEST_PHRASES = [
    "do i have", "diagnose me", "what disease do i have",
    "am i sick", "what's wrong with me", "give me a diagnosis",
    "what stage am i", "what ckd stage am i", "what stage is my",
]

PRESCRIPTION_REQUEST_PHRASES = [
    "what medication should i take", "prescribe", "what dose",
    "what drug should i", "how much medicine",
]

JAILBREAK_PHRASES = [
    "you are now a", "act as a", "pretend you are a", "roleplay as",
    "ignore previous instructions", "ignore your instructions",
    "ignore the instructions", "no restrictions", "as my doctor",
    "as a doctor", "with no restrictions",
]

REFUSAL_MESSAGE = (
    "I can't diagnose a condition or recommend treatment -- I can only "
    "explain what a prediction model found, or answer general reference "
    "questions about CKD. Please consult a healthcare professional for "
    "diagnosis or treatment decisions."
)


def check_guardrails(user_message: str) -> dict:
    """
    Returns {"blocked": bool, "reason": str or None}.
    Deliberately fails safe: on ambiguous input, this does NOT block --
    Day 18's eval set will tell us if that's too permissive, informed by
    real test cases rather than guessing now.
    """
    lowered = user_message.lower()

    for phrase in DIAGNOSIS_REQUEST_PHRASES:
        if phrase in lowered:
            return {"blocked": True, "reason": REFUSAL_MESSAGE}

    for phrase in PRESCRIPTION_REQUEST_PHRASES:
        if phrase in lowered:
            return {"blocked": True, "reason": REFUSAL_MESSAGE}

    for phrase in JAILBREAK_PHRASES:
        if phrase in lowered:
            return {"blocked": True, "reason": REFUSAL_MESSAGE}

    return {"blocked": False, "reason": None}