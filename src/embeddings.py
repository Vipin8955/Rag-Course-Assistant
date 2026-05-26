"""
embeddings.py — Sentence embedding models wrapper
Supports: all-MiniLM-L6-v2 and BAAI/bge-base-en-v1.5
"""

from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer

# Cache loaded models to avoid re-loading on every call
_model_cache: dict[str, SentenceTransformer] = {}

AVAILABLE_MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
}


def get_model(model_name: str) -> SentenceTransformer:
    """Load (or return cached) SentenceTransformer model."""
    if model_name not in _model_cache:
        model_id = AVAILABLE_MODELS.get(model_name, model_name)
        _model_cache[model_name] = SentenceTransformer(model_id)
    return _model_cache[model_name]


def embed_texts(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Compute embeddings for a list of texts.

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    model = get_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalise for cosine sim via dot product
    )
    return embeddings


def embed_query(query: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed a single query string."""
    return embed_texts([query], model_name=model_name)[0]


def get_embedding_dim(model_name: str) -> int:
    """Return the embedding dimension for a model."""
    model = get_model(model_name)
    return model.get_sentence_embedding_dimension()
