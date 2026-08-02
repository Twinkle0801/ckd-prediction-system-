"""
eval_rag_questions.py

Manual eval exercise for the RAG tool (Day 17, Segment 2 item 2). Not a
pytest suite -- this is exploratory: run it, read the actual answers and
sources, and manually judge whether retrieval/routing behaved sensibly.
Findings get logged into docs/rag_eval_notes.md, not asserted here.

Run with: python eval_rag_questions.py
"""

from src.ai_assistant.router import route_message

# Questions expected to be ANSWERABLE from the 3 reference docs
ANSWERABLE_QUESTIONS = [
    "What does a serum creatinine of 3.8 typically indicate?",
    "What is a normal range for potassium?",
    "What does high blood urea indicate?",
    "What are risk factors for CKD?",
    "Does diabetes increase CKD risk?",
    "What is Stage 3 CKD?",
    "What does albumin in urine typically indicate?",
    "What normal range is expected for hemoglobin?",
    "How does hypertension relate to kidney disease?",
    "What does a low eGFR mean?",
]

# Questions expected to be DECLINED (no relevant reference material exists)
OFF_TOPIC_QUESTIONS = [
    "What does love mean?",
    "What is the recipe for chocolate cake?",
    "What does stock market volatility typically indicate?",
]


def run_eval():
    print("=" * 70)
    print("ANSWERABLE QUESTIONS (expect: tool=rag, has_grounded_context=True)")
    print("=" * 70)

    for q in ANSWERABLE_QUESTIONS:
        result = route_message(q)
        tool = result.get("tool")
        has_context = result.get("has_grounded_context", "N/A")
        sources = result.get("sources", [])
        print(f"\nQ: {q}")
        print(f"   tool={tool} | has_context={has_context} | sources={sources}")

    print("\n" + "=" * 70)
    print("OFF-TOPIC QUESTIONS (expect: has_grounded_context=False, sources=[])")
    print("=" * 70)

    for q in OFF_TOPIC_QUESTIONS:
        result = route_message(q)
        tool = result.get("tool")
        has_context = result.get("has_grounded_context", "N/A")
        sources = result.get("sources", [])
        print(f"\nQ: {q}")
        print(f"   tool={tool} | has_context={has_context} | sources={sources}")


if __name__ == "__main__":
    run_eval()