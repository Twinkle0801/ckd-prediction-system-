# src/ai_assistant/rag/ingest.py
"""
One-time (or re-run-when-docs-change) ingestion: chunk the reference
documents and store their embeddings in a persistent ChromaDB collection.

Run directly with: python -m src.ai_assistant.rag.ingest
"""

import os
import glob

import chromadb
from sentence_transformers import SentenceTransformer

REFERENCE_DOCS_DIR = "data/reference_docs"
CHROMA_PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME = "ckd_reference_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, runs locally -- no API cost


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Simple paragraph-aware chunking: split on blank lines first (natural
    paragraph boundaries in the reference docs), then merge small
    paragraphs up to chunk_size characters, with a little overlap so a
    fact split across a boundary isn't lost entirely from either chunk.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) <= chunk_size:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    return chunks


def build_index():
    """
    Reads every .md file in REFERENCE_DOCS_DIR, chunks it, embeds each
    chunk, and stores it in a persistent Chroma collection (reset from
    scratch each run -- simplest correct behavior for a small, infrequently
    updated reference set).
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Reset the collection each run so re-ingesting never duplicates chunks
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet -- fine on first run

    collection = client.create_collection(COLLECTION_NAME)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    doc_paths = glob.glob(os.path.join(REFERENCE_DOCS_DIR, "*.md"))
    if not doc_paths:
        raise RuntimeError(f"No .md files found in {REFERENCE_DOCS_DIR} -- nothing to ingest.")

    all_chunks, all_ids, all_metadatas = [], [], []

    for path in doc_paths:
        source_name = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{source_name}::chunk_{i}")
            all_metadatas.append({"source": source_name, "chunk_index": i})

    embeddings = model.encode(all_chunks).tolist()

    collection.add(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_chunks,
        metadatas=all_metadatas,
    )

    print(f"Ingested {len(all_chunks)} chunks from {len(doc_paths)} documents into '{COLLECTION_NAME}'.")
    return collection


if __name__ == "__main__":
    build_index()