"""
vector_store.py — FAISS and ChromaDB vector store wrappers
"""

from __future__ import annotations
import os
import uuid
import numpy as np
from pathlib import Path

# ── FAISS ─────────────────────────────────────────────────────────────────────
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# ── ChromaDB ──────────────────────────────────────────────────────────────────
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  FAISS Store
# ══════════════════════════════════════════════════════════════════════════════

class FAISSStore:
    """In-memory FAISS flat index with metadata storage."""

    def __init__(self, dim: int):
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu is not installed.")
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine with normed vecs)
        self.chunks: list[dict] = []

    def add(self, embeddings: np.ndarray, chunks: list[dict]):
        """Add embeddings and associated chunk metadata."""
        embeddings = embeddings.astype("float32")
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """Search for top-k most similar chunks."""
        query = query_embedding.astype("float32").reshape(1, -1)
        k = min(top_k, self.index.ntotal)
        if k == 0:
            return []
        scores, indices = self.index.search(query, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    def clear(self):
        self.index.reset()
        self.chunks.clear()

    @property
    def total(self) -> int:
        return self.index.ntotal


# ══════════════════════════════════════════════════════════════════════════════
#  ChromaDB Store
# ══════════════════════════════════════════════════════════════════════════════

class ChromaStore:
    """Persistent ChromaDB vector store."""

    def __init__(self, collection_name: str = "course_docs"):
        if not CHROMA_AVAILABLE:
            raise ImportError("chromadb is not installed.")
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        # Always start fresh to stay consistent with current session
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, embeddings: np.ndarray, chunks: list[dict]):
        """Add embeddings and chunk metadata to ChromaDB."""
        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "source": c.get("source", "unknown"),
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

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """Search for top-k most similar chunks."""
        n_results = min(top_k, self.collection.count())
        if n_results == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
        )
        chunks = []
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
                    "score": 1 - dist,  # convert distance to similarity
                }
            )
        return chunks

    def clear(self):
        try:
            self.client.delete_collection("course_docs")
            self.collection = self.client.create_collection(
                name="course_docs",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            pass

    @property
    def total(self) -> int:
        return self.collection.count()


# ══════════════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════════════

def create_store(store_type: str, dim: int):
    """Return a FAISSStore or ChromaStore."""
    if store_type == "FAISS":
        return FAISSStore(dim=dim)
    elif store_type == "ChromaDB":
        return ChromaStore()
    else:
        raise ValueError(f"Unknown store type: {store_type}")
