"""
retriever.py — Semantic retrieval pipeline
"""

from __future__ import annotations
from src.embeddings import embed_query, embed_texts, get_embedding_dim
from src.vector_store import create_store
from src.chunking import chunk_documents, get_chunk_stats
from src.ingestion import load_all_documents


def build_index(
    docs: dict[str, str],
    chunk_size: int,
    overlap: int,
    model_name: str,
    store_type: str,
) -> tuple:
    """
    Full pipeline: chunk → embed → index.

    Returns:
        (store, chunks, stats)
    """
    # 1. Chunk
    chunks = chunk_documents(docs, chunk_size=chunk_size, overlap=overlap)
    stats = get_chunk_stats(chunks)

    if not chunks:
        return None, [], stats

    # 2. Embed
    texts = [c["text"] for c in chunks]
    dim = get_embedding_dim(model_name)
    embeddings = embed_texts(texts, model_name=model_name, show_progress=False)

    # 3. Store
    store = create_store(store_type, dim=dim)
    store.add(embeddings, chunks)

    return store, chunks, stats


def retrieve(
    query: str,
    store,
    model_name: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve top-k relevant chunks for a query.

    Returns list of chunk dicts with added 'score' field.
    """
    if store is None or store.total == 0:
        return []

    query_emb = embed_query(query, model_name=model_name)
    results = store.search(query_emb, top_k=top_k)
    return results
