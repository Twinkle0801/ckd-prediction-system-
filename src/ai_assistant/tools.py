# src/ai_assistant/tools.py
"""
The three tool types the router can dispatch to.

Tool 1 (prediction call) and Tool 3 (SHAP explainer) are fully implemented
here as of Day 16. Tool 2 (RAG) remains stubbed until Day 17.
"""

from src.inference import predict_sample, preprocess_input
from src.evaluation import explain_model, explain_single_prediction
from src.ai_assistant.prompts import build_explainer_prompt, DISCLAIMER
from src.ai_assistant.llm_client import simple_completion
from src.ai_assistant.grounding import check_grounding
from src.ai_assistant.rag.retriever import retrieve_relevant_chunks
from src.ai_assistant.prompts import build_rag_prompt
from src.ai_assistant.llm_client import completion_with_usage
from src.ai_assistant.cost_tracker import log_query_cost
from src.ai_assistant.cache import get_cached, set_cached


def call_prediction_tool(patient_input: dict, bundle: dict) -> dict:
    """
    Tool type 1: run a real prediction + SHAP explanation for one patient.
    Reuses the exact same functions api/main.py and app.py already call --
    no duplicated prediction logic, so results are guaranteed consistent
    across the API, dashboard, and this AI assistant.
    """
    result = predict_sample(patient_input, bundle=bundle)
    processed = preprocess_input(patient_input, bundle)
    explainer, shap_values = explain_model(bundle["model"], processed)
    explanation = explain_single_prediction(bundle["model"], explainer, shap_values, processed, 0)

    return {
        "prediction_result": result,
        "shap_explanation": explanation,
    }


def call_shap_explainer_tool(prediction_result: dict, shap_explanation: dict) -> dict:
    """
    Tool type 3: turn a prediction + SHAP contributions into a plain-English
    explanation via the LLM, strictly grounded in the given numbers.

    Returns a dict (not just a string) so callers can see whether the
    explanation passed the grounding check.

    The disclaimer is appended in CODE, not trusted to the LLM's own output --
    defense in depth. Even if the model forgets to include it (or the mock
    fallback fires), the disclaimer is guaranteed present.
    """
    prompt = build_explainer_prompt(prediction_result, shap_explanation)
    raw_explanation = simple_completion(prompt, max_tokens=300)

    if DISCLAIMER not in raw_explanation:
        raw_explanation = f"{raw_explanation.strip()}\n\n{DISCLAIMER}"

    grounding_result = check_grounding(raw_explanation, shap_explanation, prediction_result)

    return {
        "explanation": raw_explanation,
        "grounded": grounding_result["grounded"],
        "ungrounded_numbers": grounding_result["ungrounded_numbers"],
    }


def call_rag_tool(user_question: str) -> dict:
    """
    Tool type 2: retrieve relevant reference chunks, build a grounded
    prompt, and get an LLM answer with citations. Checks cache first;
    logs token cost on real (non-cached) calls.
    """
    cached = get_cached(user_question)
    if cached is not None:
        result = cached.copy()
        result["cache_hit"] = True
        return result

    chunks = retrieve_relevant_chunks(user_question, top_k=3)
    prompt, has_context = build_rag_prompt(user_question, chunks)

    completion = completion_with_usage(prompt, max_tokens=250)
    answer = completion["text"]

    log_query_cost(
        user_question,
        completion["input_tokens"],
        completion["output_tokens"],
        cache_hit=False,
    )

    if DISCLAIMER not in answer:
        answer = f"{answer.strip()}\n\n{DISCLAIMER}"

    sources = sorted(set(c["source"] for c in chunks if c["distance"] <= 1.0)) if has_context else []

    result = {
        "answer": answer,
        "has_grounded_context": has_context,
        "sources": sources,
        "cache_hit": False,
    }
    set_cached(user_question, result)
    return result