"""
Query-time retrieval: embed a user question and retrieve the most
relevant chunks from the persistent ChromaDB collection.

If the collection does not exist (e.g., first deployment on Streamlit
Cloud), automatically build it from the reference documents.
"""

import chromadb
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

from src.ai_assistant.rag.ingest import (
    build_index,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)

# Lazy-loaded embedding model
_model = None


def _get_model():
    """
    Load the embedding model only once.
    """
    global _model

    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _model


def _get_collection():
    """
    Return the ChromaDB collection.

    If the collection does not exist (first deployment),
    automatically build the vector database.
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    try:
        return client.get_collection(COLLECTION_NAME)

    except NotFoundError:
        print("Chroma collection not found.")
        print("Building vector database...")

        # Build embeddings + create collection
        build_index()

        # Create a fresh client after building
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        return client.get_collection(COLLECTION_NAME)


def retrieve_relevant_chunks(query: str, top_k: int = 3) -> list:
    """
    Retrieve the top-k most relevant chunks for a user query.

    Returns:
        [
            {
                "text": "...",
                "source": "...",
                "distance": 0.12
            },
            ...
        ]
    """

    collection = _get_collection()

    model = _get_model()

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    chunks = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, metadata, distance in zip(documents, metadatas, distances):
        chunks.append(
            {
                "text": doc,
                "source": metadata.get("source", "Unknown"),
                "distance": distance,
            }
        )

    return chunks