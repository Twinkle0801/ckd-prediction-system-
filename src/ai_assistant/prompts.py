# src/ai_assistant/prompts.py
"""
Prompt templates for the AI assistant.

Kept separate from tools.py so prompt wording can be iterated on without
touching the calling logic. Prompt engineering changes shouldn't require
re-reading the tool dispatch code every time.
"""


DISCLAIMER = (
    "This is decision support only and not a medical diagnosis. "
    "Please consult a healthcare professional."
)

# Maximum vector distance considered relevant for RAG retrieval.
# Chunks with a larger distance will be ignored.
RELEVANCE_DISTANCE_THRESHOLD = 1.0


def build_explainer_prompt(
    prediction_result: dict,
    shap_explanation: dict,
) -> str:
    """
    Build a tightly-grounded explanation prompt for the prediction.

    Every numeric value the LLM is allowed to mention is explicitly listed
    so it cannot invent or hallucinate laboratory values.
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

Prediction: {"CKD" if label == "ckd" else "Not CKD"}
Probability of CKD: {probability:.2f}

Top contributing factors (the ONLY numbers you are allowed to reference):
{factors_text}

Instructions:
- Explain in 3-4 plain-English sentences why these factors drove this prediction.
- Use ONLY the numbers listed above.
- Do not invent, estimate, round to a different value, or reference any lab value not explicitly listed.
- Do not state a diagnosis.
- Frame this as "factors the model weighted," not medical fact.
- Do not recommend any treatment, medication, or dosage.
- Remember: use ONLY the exact numbers given above, nothing else.
"""

    return prompt


def build_rag_prompt(
    user_question: str,
    retrieved_chunks: list,
) -> tuple[str, bool]:
    """
    Build a grounded RAG prompt.

    Returns:
        (prompt, has_relevant_context)

    If no retrieved chunk is sufficiently relevant, the prompt instructs
    the LLM to explicitly say it lacks grounded information rather than
    answering from general knowledge.
    """

    relevant_chunks = [
        chunk
        for chunk in retrieved_chunks
        if chunk["distance"] <= RELEVANCE_DISTANCE_THRESHOLD
    ]

    if not relevant_chunks:
        prompt = f"""The user asked:

"{user_question}"

No sufficiently relevant reference material was found for this question.

Instructions:
- Clearly state that you do not have enough grounded reference material to answer.
- Do NOT answer from your own general medical knowledge.
- Do NOT guess.
- Recommend consulting a healthcare professional.
"""

        return prompt, False

    context_text = "\n\n---\n\n".join(
        f"[Source: {chunk['source']}]\n{chunk['text']}"
        for chunk in relevant_chunks
    )

    prompt = f"""You are answering a medical reference question using ONLY the retrieved context below.

Retrieved Context
=================
{context_text}

=================

User Question:
"{user_question}"

Instructions:
- Answer ONLY using the retrieved context.
- Do not use your own medical knowledge.
- If the retrieved context is incomplete, explicitly say so.
- Cite the source document(s) you used (for example: "According to lab_normal_ranges.md...").
- Do not diagnose any individual.
- Keep the answer concise (2-4 sentences).
"""

    return prompt, True

