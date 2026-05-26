"""
vector_store.py — FAISS (in-memory) and ChromaDB (persistent) vector store wrappers.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import CHROMA_DIR

logger = logging.getLogger(__name__)

# ── Optional dependency guards ────────────────────────────────────────────────
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not found — FAISS store will be unavailable.")

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("chromadb not found — ChromaDB store will be unavailable.")


# ══════════════════════════════════════════════════════════════════════════════
#  FAISS Store
# ══════════════════════════════════════════════════════════════════════════════

class FAISSStore:
    """In-memory FAISS flat index with metadata storage."""

    def __init__(self, dim: int) -> None:
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss-cpu is not installed. Run: pip install faiss-cpu"
            )
        if dim <= 0:
            raise ValueError(f"Embedding dimension must be > 0, got {dim}.")
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine with normalised vecs)
        self.chunks: list[dict] = []
        logger.debug("FAISSStore created (dim=%d).", dim)

    def add(self, embeddings: np.ndarray, chunks: list[dict]) -> None:
        """
        Add embeddings and associated chunk metadata.

        Raises:
            ValueError: On shape mismatch or empty inputs.
            RuntimeError: On FAISS internal error.
        """
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("embeddings must be a non-empty array.")
        if not chunks:
            raise ValueError("chunks list must not be empty.")
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"embeddings length ({len(embeddings)}) != chunks length ({len(chunks)})."
            )
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dim mismatch: store expects {self.dim}, got {embeddings.shape[1]}."
            )

        try:
            embeddings = embeddings.astype("float32")
            self.index.add(embeddings)
            self.chunks.extend(chunks)
            logger.debug("FAISSStore: added %d vectors (total=%d).", len(chunks), self.total)
        except Exception as e:
            raise RuntimeError(f"FAISS add failed: {e}") from e

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Search for the top-k most similar chunks.

        Returns:
            List of chunk dicts with an added 'score' field.
            Returns [] if the index is empty or top_k < 1.
        """
        if self.total == 0:
            logger.debug("FAISSStore.search called on empty index.")
            return []
        if top_k < 1:
            logger.warning("top_k=%d < 1 — returning empty results.", top_k)
            return []

        try:
            query = query_embedding.astype("float32").reshape(1, -1)
            if query.shape[1] != self.dim:
                raise ValueError(
                    f"Query dim {query.shape[1]} != store dim {self.dim}."
                )
            k = min(top_k, self.total)
            scores, indices = self.index.search(query, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue  # FAISS uses -1 for padding
                chunk = dict(self.chunks[idx])
                chunk["score"] = float(score)
                results.append(chunk)

            logger.debug("FAISSStore.search: top_k=%d → %d results.", top_k, len(results))
            return results

        except ValueError:
            raise
        except Exception as e:
            logger.error("FAISS search error: %s", e)
            return []

    def clear(self) -> None:
        """Reset the index and metadata."""
        try:
            self.index.reset()
            self.chunks.clear()
            logger.debug("FAISSStore cleared.")
        except Exception as e:
            logger.error("Failed to clear FAISSStore: %s", e)

    @property
    def total(self) -> int:
        """Number of indexed vectors."""
        return self.index.ntotal


# ══════════════════════════════════════════════════════════════════════════════
#  ChromaDB Store
# ══════════════════════════════════════════════════════════════════════════════

class ChromaStore:
    """Persistent ChromaDB vector store."""

    _COLLECTION_NAME = "course_docs"

    def __init__(self, collection_name: str = _COLLECTION_NAME) -> None:
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "chromadb is not installed. Run: pip install chromadb==0.4.24"
            )

        self._collection_name = collection_name

        try:
            self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialise ChromaDB client at '{CHROMA_DIR}': {e}\n"
                "Try deleting the data/chroma_db/ folder and restarting."
            ) from e

        # Always start fresh for the current session
        self._reset_collection()
        logger.debug("ChromaStore initialised (collection='%s').", collection_name)

    def _reset_collection(self) -> None:
        """Delete and recreate the collection (fresh session start)."""
        try:
            self.client.delete_collection(self._collection_name)
        except Exception:
            pass  # Collection may not exist yet — that's fine

        try:
            self.collection = self.client.create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create ChromaDB collection '{self._collection_name}': {e}"
            ) from e

    def add(self, embeddings: np.ndarray, chunks: list[dict]) -> None:
        """
        Add embeddings and chunk metadata to ChromaDB.

        Raises:
            ValueError:   On input shape mismatch.
            RuntimeError: On ChromaDB insertion failure.
        """
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("embeddings must be a non-empty array.")
        if not chunks:
            raise ValueError("chunks list must not be empty.")
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"embeddings length ({len(embeddings)}) != chunks length ({len(chunks)})."
            )

        try:
            ids = [str(uuid.uuid4()) for _ in chunks]
            documents = [c.get("text", "") for c in chunks]
            metadatas = [
                {
                    "source": str(c.get("source", "unknown")),
                    "chunk_id": str(c.get("chunk_id", 0)),
                    "word_count": str(c.get("word_count", 0)),
                    "start_word": str(c.get("start_word", 0)),
                }
                for c in chunks
            ]
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
            )
            logger.debug("ChromaStore: added %d vectors (total=%d).", len(chunks), self.total)
        except Exception as e:
            raise RuntimeError(f"ChromaDB add failed: {e}") from e

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Search for top-k most similar chunks.

        Returns:
            List of chunk dicts with 'score' field. Empty list if store is empty.
        """
        count = self.total
        if count == 0:
            logger.debug("ChromaStore.search called on empty collection.")
            return []
        if top_k < 1:
            logger.warning("top_k=%d < 1 — returning empty results.", top_k)
            return []

        n_results = min(top_k, count)
        try:
            result = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
            )
        except Exception as e:
            logger.error("ChromaDB query failed: %s", e)
            return []

        chunks = []
        try:
            for doc, meta, dist in zip(
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            ):
                chunks.append(
                    {
                        "text": doc,
                        "source": meta.get("source", "unknown"),
                        "chunk_id": int(meta.get("chunk_id", 0)),
                        "word_count": int(meta.get("word_count", 0)),
                        "start_word": int(meta.get("start_word", 0)),
                        "score": max(0.0, 1.0 - float(dist)),  # distance → similarity, clamp ≥ 0
                    }
                )
        except (KeyError, IndexError, TypeError) as e:
            logger.error("Failed to parse ChromaDB results: %s", e)
            return []

        logger.debug("ChromaStore.search: top_k=%d → %d results.", top_k, len(chunks))
        return chunks

    def clear(self) -> None:
        """Delete and recreate the collection."""
        try:
            self._reset_collection()
            logger.debug("ChromaStore cleared.")
        except Exception as e:
            logger.error("Failed to clear ChromaStore: %s", e)

    @property
    def total(self) -> int:
        """Number of stored vectors."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error("Failed to get ChromaStore count: %s", e)
            return 0


# ══════════════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════════════

def create_store(store_type: str, dim: int) -> FAISSStore | ChromaStore:
    """
    Instantiate the requested vector store.

    Args:
        store_type: 'FAISS' or 'ChromaDB'.
        dim:        Embedding dimension (used by FAISS; ignored by ChromaDB).

    Returns:
        FAISSStore or ChromaStore instance.

    Raises:
        ValueError:   For unknown store_type.
        ImportError:  If the required library is not installed.
        RuntimeError: If initialisation fails.
    """
    if store_type == "FAISS":
        return FAISSStore(dim=dim)
    elif store_type == "ChromaDB":
        return ChromaStore()
    else:
        raise ValueError(
            f"Unknown store type: '{store_type}'. Valid options: 'FAISS', 'ChromaDB'."
        )
