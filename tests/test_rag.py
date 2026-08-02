# tests/test_rag.py
from src.ai_assistant.rag.ingest import chunk_text
from src.ai_assistant.rag.retriever import retrieve_relevant_chunks
from src.ai_assistant.prompts import build_rag_prompt, RELEVANCE_DISTANCE_THRESHOLD
from src.ai_assistant.tools import call_rag_tool


def test_chunk_text_splits_on_paragraphs():
    text = "Para one.\n\nPara two.\n\nPara three."
    chunks = chunk_text(text, chunk_size=15)
    assert len(chunks) >= 2  # small chunk_size forces a split across these paragraphs


def test_retrieve_relevant_chunks_returns_ranked_results():
    chunks = retrieve_relevant_chunks("What does a serum creatinine of 3.8 typically indicate?", top_k=3)
    assert len(chunks) == 3
    assert chunks[0]["distance"] <= chunks[1]["distance"] <= chunks[2]["distance"]  # sorted by relevance
    assert any(c["source"] == "ckd_stages.md" for c in chunks)


def test_build_rag_prompt_flags_irrelevant_query():
    fake_chunks = [
        {
            "text": "irrelevant",
            "source": "x.md",
            "distance": 999.0,
        }
    ]

    prompt, has_context = build_rag_prompt(
        "What is the capital of France?",
        fake_chunks,
    )

    assert has_context is False
    assert "grounded reference material" in prompt
    assert "consulting a healthcare professional" in prompt.lower()


def test_build_rag_prompt_includes_relevant_chunks():
    real_chunks = [{"text": "Serum Creatinine: 0.6-1.3 mg/dL...", "source": "lab_normal_ranges.md", "distance": 0.3}]
    prompt, has_context = build_rag_prompt("What is normal creatinine?", real_chunks)
    assert has_context is True
    assert "lab_normal_ranges.md" in prompt
    assert "0.6-1.3" in prompt


def test_call_rag_tool_returns_sources_for_relevant_question():
    result = call_rag_tool("What does a serum creatinine of 3.8 typically indicate?")
    assert result["has_grounded_context"] is True
    assert "ckd_stages.md" in result["sources"]


def test_call_rag_tool_declines_for_irrelevant_question():
    result = call_rag_tool("What is the capital of France?")
    assert result["has_grounded_context"] is False
    assert result["sources"] == []