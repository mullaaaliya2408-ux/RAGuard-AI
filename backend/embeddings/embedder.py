"""
Wraps Sentence Transformers behind a simple interface. The model is loaded
once and cached, since loading it per-request would be slow and wasteful.

We use all-MiniLM-L6-v2: a small, fast model (384-dim vectors) that offers
a strong accuracy/speed tradeoff for chunk-level semantic search.
"""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("raguard.embedder")

_model: SentenceTransformer | None = None
EMBEDDING_DIM = 384


def get_embedding_model() -> SentenceTransformer:
    """Return a cached embedding model, loading it on first use."""
    global _model
    if _model is None:
        logger.info("Loading embedding model (first use only)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a batch of texts and L2-normalize the vectors, so that a simple
    dot product between vectors equals cosine similarity. Normalizing once
    here means the vector store doesn't need to worry about it later.
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype="float32")

    model = get_embedding_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8  # avoid divide-by-zero on empty/degenerate text
    return (vectors / norms).astype("float32")


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string, returned as a (1, dim) array for FAISS."""
    return embed_texts([query])