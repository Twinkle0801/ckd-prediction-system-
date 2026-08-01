# src/ai_assistant/tools.py
"""
The three tool types the router can dispatch to. Each is stubbed here --
real implementations land in Day 16 (predict/explain) and Day 17 (RAG).
Keeping them as separate functions with a consistent signature means the
router doesn't need to change when each is filled in later.
"""


def call_prediction_tool(patient_input: dict) -> dict:
    """
    Tool type 1: direct call into the Phase 10 prediction function.
    Day 16 will wire this to src.inference.predict_sample +
    src.evaluation.explain_single_prediction.
    """
    raise NotImplementedError("Wired in Day 16 (AI Health Assistant)")


def call_shap_explainer_tool(prediction_result: dict, shap_explanation: dict) -> str:
    """
    Tool type 3: turn a prediction + SHAP contributions into a plain-English
    explanation. Day 16 will wire this to an actual LLM prompt, grounded
    strictly in the numbers passed in.
    """
    raise NotImplementedError("Wired in Day 16 (AI Health Assistant)")


def call_rag_tool(user_question: str) -> dict:
    """
    Tool type 2: RAG over CKD reference documents. Day 17 will wire this to
    a ChromaDB/FAISS retrieval pipeline plus a cited LLM answer.
    """
    raise NotImplementedError("Wired in Day 17 (Document Intelligence / RAG)")