# src/ai_assistant/router.py
"""
Router: decides which AI assistant tool should handle a user's message
before any tool actually runs.

Current routing strategy:
1. Guardrails (highest priority)
2. SHAP explanation requests
3. RAG knowledge-base questions
4. No matching tool

This is intentionally simple keyword-based routing. It can later be
replaced with LLM function calling/tool calling if desired.
"""

from src.ai_assistant.tools import (
    call_prediction_tool,
    call_shap_explainer_tool,
    call_rag_tool,
)

from src.ai_assistant.guardrails import check_guardrails


# Questions that should be answered using the knowledge base (RAG)
RAG_KEYWORDS = [
    "what does",
    "normal range",
    "typically indicate",
    "stage",
    "means",
]

# Questions asking why a prediction was made
EXPLAIN_KEYWORDS = [
    "why",
    "explain this prediction",
    "what factors",
]


def route_message(user_message: str, context: dict = None) -> dict:
    """
    Routes a user's message to the appropriate assistant tool.

    Parameters
    ----------
    user_message : str
        The user's input message.

    context : dict, optional
        Session context containing the latest prediction and SHAP
        explanation for follow-up questions.

    Returns
    -------
    dict
        Response from the selected tool.
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
    if any(keyword in lowered for keyword in RAG_KEYWORDS):
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