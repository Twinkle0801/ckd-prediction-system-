# src/ai_assistant/rag/retriever.py
"""
Query-time retrieval: embed a user question and pull the top-k most
similar chunks from the persistent Chroma collection built by ingest.py.
"""

import chromadb
from sentence_transformers import SentenceTransformer

from src.ai_assistant.rag.ingest import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME

_model = None  # lazy-loaded singleton -- loading the embedding model is slow, do it once


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def retrieve_relevant_chunks(query: str, top_k: int = 3) -> list:
    """
    Returns a list of {"text": str, "source": str, "distance": float},
    ordered by relevance (lowest distance = most similar).
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    model = _get_model()
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i],
        })

    return chunks