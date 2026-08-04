# src/ai_assistant/router.py

"""
Router: decides which AI assistant tool should handle a user's message
before any tool actually runs.

Routing priority:
1. Safety Guardrails
2. SHAP explanation requests
3. RAG knowledge-base questions
4. No matching tool
"""

from src.ai_assistant.tools import (
    call_shap_explainer_tool,
    call_rag_tool,
)

from src.ai_assistant.guardrails import check_guardrails


# Questions asking why a prediction was made
EXPLAIN_KEYWORDS = [
    "why",
    "explain this prediction",
    "what factors",
]


def route_message(user_message: str, context: dict = None) -> dict:
    """
    Routes a user's message to the appropriate assistant tool.
    """

    # ---------------------------------------------------------
    # 1. Safety Guardrails
    # ---------------------------------------------------------
    guardrail_result = check_guardrails(user_message)

    if guardrail_result["blocked"]:
        return {
            "tool": "guardrail_refusal",
            "message": guardrail_result["reason"],
        }

    lowered = user_message.lower()

    # ---------------------------------------------------------
    # 2. SHAP Prediction Explanation
    # ---------------------------------------------------------
    if (
        context
        and context.get("last_prediction")
        and any(keyword in lowered for keyword in EXPLAIN_KEYWORDS)
    ):
        prediction_result = context["last_prediction"]
        shap_explanation = context["last_shap_explanation"]

        tool_result = call_shap_explainer_tool(
            prediction_result,
            shap_explanation,
        )

        return {
            "tool": "shap_explainer",
            **tool_result,
        }

    # ---------------------------------------------------------
    # 3. RAG Knowledge Base
    # ---------------------------------------------------------
    QUESTION_STARTERS = (
        "what",
        "how",
        "why",
        "when",
        "where",
        "which",
        "who",
        "can",
        "does",
        "is",
        "are",
    )

    CKD_TERMS = (
        "ckd",
        "kidney",
        "renal",
        "creatinine",
        "gfr",
        "egfr",
        "dialysis",
        "albumin",
        "hemoglobin",
        "blood urea",
        "protein",
        "hypertension",
        "diabetes",
        "anemia",
        "blood pressure",
    )

    if (
        lowered.startswith(QUESTION_STARTERS)
        or any(term in lowered for term in CKD_TERMS)
    ):
        tool_result = call_rag_tool(user_message)

        return {
            "tool": "rag",
            **tool_result,
        }

    # ---------------------------------------------------------
    # 4. No Tool Matched
    # ---------------------------------------------------------
    return {
        "tool": "none_matched",
        "status": "no_tool_confidently_matched",
    }