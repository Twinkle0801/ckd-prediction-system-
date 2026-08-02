# src/ai_assistant/grounding.py
"""
Lightweight automated check: does a generated explanation only reference
numbers that were actually provided as grounding data? This can't catch
every form of hallucination (e.g. inventing a qualitative claim with no
number attached, or citing a real number under the wrong feature name),
but it catches the most common and most dangerous failure mode -- a
fabricated lab value or percentage.
"""

import re

HARMLESS_COUNTING_NUMBERS = {"0.0", "1.0", "2.0", "3.0", "4.0", "5.0"}


def extract_numbers(text: str) -> set:
    """Pull every standalone number out of a string, normalized to strings
    with trailing zeros stripped for comparison (e.g. '15.40' -> '15.4')."""
    raw_matches = re.findall(r"-?\d+\.?\d*", text)
    normalized = set()
    for m in raw_matches:
        try:
            normalized.add(str(float(m)))
        except ValueError:
            continue
    return normalized


def check_grounding(explanation_text: str, shap_explanation: dict, prediction_result: dict) -> dict:
    """
    Returns {"grounded": bool, "ungrounded_numbers": list}.
    Compares every number mentioned in the explanation against the actual
    source numbers it was given (SHAP values, patient values, probability).

    FIX (found via Day 16 grounding edge-case testing): an earlier version
    used "float(n) >= 2" to exclude harmless small counts like "3-4
    sentences", but this silently exempted almost all real clinical values
    too (SHAP contributions, creatinine, potassium are typically well under
    2), meaning fabricated small decimals (a fake reference range, a fake
    combined SHAP total) were never even checked. Now only exact small
    whole-number counts (0-5) are exempted -- any decimal, or any integer
    outside that range, is checked regardless of magnitude.
    """
    allowed_numbers = set()

    allowed_numbers.add(str(float(prediction_result["probability"])))

    for item in shap_explanation["top_contributions"]:
        try:
            allowed_numbers.add(str(float(item["value"])))
        except (ValueError, TypeError):
            pass  # categorical values like "yes"/"normal" -- not numeric, skip
        allowed_numbers.add(str(round(float(item["shap_contribution"]), 3)))
        allowed_numbers.add(str(round(float(item["shap_contribution"]), 2)))
        allowed_numbers.add(str(round(float(item["shap_contribution"]), 1)))

    mentioned_numbers = extract_numbers(explanation_text)

    suspicious = {
        n for n in mentioned_numbers
        if n not in HARMLESS_COUNTING_NUMBERS and n not in allowed_numbers
    }

    return {
        "grounded": len(suspicious) == 0,
        "ungrounded_numbers": sorted(suspicious),
    }