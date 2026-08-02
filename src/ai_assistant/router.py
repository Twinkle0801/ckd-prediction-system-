# src/ai_assistant/router.py
"""
Router: decides which of the three tool types a user's message needs,
before any tool actually runs. This is intentionally simple keyword-based
routing for now -- Day 18 may replace this with the LLM's own native
tool-use/function-calling if that proves more reliable in practice.
"""

from src.ai_assistant.tools import (
    call_prediction_tool,
    call_shap_explainer_tool,
    call_rag_tool,
)
from src.ai_assistant.guardrails import check_guardrails

RAG_KEYWORDS = ["what does", "normal range", "typically indicate", "stage", "means"]
EXPLAIN_KEYWORDS = ["why", "explain this prediction", "what factors"]


def route_message(user_message: str, context: dict = None) -> dict:
    """
    context can carry the current session's last prediction result / SHAP
    values, if the user is asking a follow-up about a prediction they just
    got, rather than a general reference question.

    Returns a dict describing what happened. Shape varies by tool:
      - guardrail_refusal: {"tool": "guardrail_refusal", "message": str}
      - shap_explainer:    {"tool": "shap_explainer", "explanation": str,
                             "grounded": bool, "ungrounded_numbers": list}
      - rag:               {"tool": "rag", "status": "not_yet_implemented"}
      - none_matched:      {"tool": "none_matched", "status": str}
    """
    guardrail_result = check_guardrails(user_message)
    if guardrail_result["blocked"]:
        return {"tool": "guardrail_refusal", "message": guardrail_result["reason"]}

    lowered = user_message.lower()

    if context and context.get("last_prediction") and any(k in lowered for k in EXPLAIN_KEYWORDS):
        prediction_result = context["last_prediction"]
        shap_explanation = context["last_shap_explanation"]
        tool_result = call_shap_explainer_tool(prediction_result, shap_explanation)
        return {"tool": "shap_explainer", **tool_result}

    if any(k in lowered for k in RAG_KEYWORDS):
        return {"tool": "rag", "status": "not_yet_implemented"}

    return {"tool": "none_matched", "status": "no_tool_confidently_matched"}