"""
retriever.py — Full RAG retrieval pipeline: chunk → embed → index → search.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.chunking import chunk_documents, get_chunk_stats
from src.embeddings import embed_query, embed_texts, get_embedding_dim
from src.ingestion import load_all_documents
from src.vector_store import create_store

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Index Building
# ══════════════════════════════════════════════════════════════════════════════

def build_index(
    docs: dict[str, str],
    chunk_size: int,
    overlap: int,
    model_name: str,
    store_type: str,
) -> tuple:
    """
    Full pipeline: chunk → embed → store.

    Args:
        docs:       {filename: text} mapping.
        chunk_size: Words per chunk.
        overlap:    Word overlap between consecutive chunks.
        model_name: Embedding model key (from AVAILABLE_MODELS).
        store_type: 'FAISS' or 'ChromaDB'.

    Returns:
        (store, chunks, stats) — or (None, [], stats) if no chunks produced.

    Raises:
        ValueError:   On invalid configuration.
        RuntimeError: On embedding or indexing failure.
    """
    if not docs:
        logger.warning("build_index called with empty docs — nothing to index.")
        empty_stats = {"total": 0, "avg_words": 0, "min_words": 0, "max_words": 0}
        return None, [], empty_stats

    # 1. Chunk
    logger.info("Chunking %d document(s) (size=%d, overlap=%d)…", len(docs), chunk_size, overlap)
    chunks = chunk_documents(docs, chunk_size=chunk_size, overlap=overlap)
    stats = get_chunk_stats(chunks)

    if not chunks:
        logger.warning("No chunks produced — documents may be empty or too short.")
        return None, [], stats

    # 2. Embed
    logger.info("Embedding %d chunks with '%s'…", len(chunks), model_name)
    texts = [c["text"] for c in chunks]
    dim = get_embedding_dim(model_name)
    embeddings = embed_texts(texts, model_name=model_name, show_progress=False)

    # 3. Store
    logger.info("Indexing into %s…", store_type)
    store = create_store(store_type, dim=dim)
    store.add(embeddings, chunks)

    logger.info(
        "Index built: %d chunks, dim=%d, store=%s.",
        store.total, dim, store_type,
    )
    return store, chunks, stats


# ══════════════════════════════════════════════════════════════════════════════
#  Retrieval
# ══════════════════════════════════════════════════════════════════════════════

def retrieve(
    query: str,
    store,
    model_name: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve top-k relevant chunks for a query.

    Args:
        query:      The search query string.
        store:      A FAISSStore or ChromaStore instance.
        model_name: Embedding model to use for the query.
        top_k:      Number of results to return.

    Returns:
        List of chunk dicts with an added 'score' field.
        Returns [] on any error or if the store is empty.
    """
    if not query or not query.strip():
        logger.warning("retrieve called with empty query — returning [].")
        return []

    if store is None:
        logger.warning("retrieve called with store=None — returning [].")
        return []

    try:
        if store.total == 0:
            logger.info("retrieve: store is empty — no results.")
            return []
    except Exception as e:
        logger.error("Cannot determine store size: %s", e)
        return []

    try:
        query_emb = embed_query(query.strip(), model_name=model_name)
        results = store.search(query_emb, top_k=top_k)
        logger.debug("retrieve: '%s' → %d results.", query[:60], len(results))
        return results
    except ValueError as e:
        logger.error("Invalid query or embedding config: %s", e)
        return []
    except RuntimeError as e:
        logger.error("Retrieval failed: %s", e)
        return []
    except Exception as e:
        logger.error("Unexpected retrieval error: %s", e)
        return []
