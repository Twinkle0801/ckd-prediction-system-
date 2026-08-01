# src/ai_assistant/guardrails.py
"""
Guardrail checks run BEFORE the router dispatches to any tool. This is
deliberately simple keyword-based detection for Day 15 -- Day 18 replaces
this with a more rigorous eval-set-driven approach, but the principle
(check first, refuse before any tool runs, never partially answer) is
locked in now.
"""

DIAGNOSIS_REQUEST_PHRASES = [
    "do i have ckd", "diagnose me", "what disease do i have",
    "am i sick", "what's wrong with me", "give me a diagnosis",
]

PRESCRIPTION_REQUEST_PHRASES = [
    "what medication should i take", "prescribe", "what dose",
    "what drug should i", "how much medicine",
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

    return {"blocked": False, "reason": None}