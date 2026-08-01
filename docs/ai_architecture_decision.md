# AI Layer: Provider Decision

**Chosen provider:** Anthropic Claude API

**Why:** Native tool-use support, strong grounding/instruction-following for the
"explain-not-diagnose" guardrail this project depends on, and straightforward
integration with a RAG pipeline (ChromaDB/FAISS) for Day 17.

**Alternative considered:** OpenAI GPT-4o — also has mature tool-use and would
work equally well architecturally. Not chosen for this build, but the router
design below is provider-agnostic enough to swap later if needed.

**Decision date:** August 1, 2026