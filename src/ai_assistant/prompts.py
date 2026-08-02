# src/ai_assistant/prompts.py
"""
Prompt templates for the AI assistant. Kept separate from tools.py so
prompt wording can be iterated on without touching the calling logic --
prompt engineering changes shouldn't require re-reading the tool dispatch
code every time.
"""

DISCLAIMER = (
    "This is decision support only and not a medical diagnosis. "
    "Please consult a healthcare professional."
)


def build_explainer_prompt(prediction_result: dict, shap_explanation: dict) -> str:
    """
    Builds a tightly-grounded prompt: every number the LLM is allowed to
    reference is explicitly listed. The instruction to use ONLY these
    numbers is repeated (top and bottom) since models are more reliable
    at following a constraint stated multiple times in different words.
    """
    label = prediction_result["prediction_label"]
    probability = prediction_result["probability"]

    contributions = shap_explanation["top_contributions"][:5]
    factors_text = "\n".join(
        f"- {item['feature']}: value={item['value']}, "
        f"shap_contribution={item['shap_contribution']:.3f} "
        f"({'increases' if item['shap_contribution'] >= 0 else 'decreases'} CKD risk)"
        for item in contributions
    )

    prompt = f"""You are explaining a machine learning model's prediction to a patient or clinician.

Prediction: {'CKD' if label == 'ckd' else 'Not CKD'}
Probability of CKD: {probability:.2f}

Top contributing factors (the ONLY numbers you are allowed to reference):
{factors_text}

Instructions:
- Explain in 3-4 plain-English sentences why these factors drove this prediction.
- Use ONLY the numbers listed above. Do not invent, estimate, round to a
  different value, or reference any lab value not explicitly listed.
- Do not state a diagnosis. Frame this as "factors the model weighted,"
  not medical fact.
- Do not recommend any treatment, medication, or dosage.
- Remember: use ONLY the exact numbers given above, nothing else.
"""
    return prompt