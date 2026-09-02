# src/embedder.py
"""Query embedding using the same model as document embedding."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the model once (singleton)."""
    return SentenceTransformer(MODEL_NAME)


def embed_query(text: str) -> list[float]:
    """Embed a single query. Returns a normalized 768-dim vector."""
    model = get_model()
    vector = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vector.tolist()
