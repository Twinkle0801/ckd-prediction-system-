# AI Layer Architecture

## Flow

Streamlit Chat UI
    -> Guardrail check (src/ai_assistant/guardrails.py)
        -> if blocked: fixed refusal message, no tool runs
        -> if allowed: Router (src/ai_assistant/router.py)
            -> Tool 1: Prediction call (src/ai_assistant/tools.py -> src.inference)
            -> Tool 2: RAG over CKD reference docs (Day 17)
            -> Tool 3: Plain-English SHAP explanation (Day 16)

## Design principles
1. Guardrail checks run BEFORE routing, not after -- a blocked message
   never reaches a tool.
2. Every tool receives only grounding data it was explicitly given
   (SHAP values, retrieved document chunks) -- never asked to rely on the
   LLM's own general medical knowledge.
3. Router and tools are decoupled -- swapping the LLM provider only
   requires changing src/ai_assistant/llm_client.py.

## Current status (Day 15)
- Router and guardrails: fully implemented and tested (keyword-based)
- Tool 1 (prediction call): stubbed, wires to src.inference in Day 16
- Tool 2 (RAG): stubbed, wires to a ChromaDB/FAISS pipeline in Day 17
- Tool 3 (SHAP explainer): stubbed, wires to an LLM prompt in Day 16
- LLM client: implemented with mock-mode fallback (no active API credits
  yet -- real calls return a labeled [MOCK] response instead of crashing)