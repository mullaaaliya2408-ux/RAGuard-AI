"""
FAISS-backed vector store with incremental indexing and disk persistence.

Uses IndexFlatIP (inner product) over L2-normalized vectors, which is
mathematically equivalent to cosine similarity search — simple, exact,
and fast enough for the chunk volumes this project targets. A production
system with millions of vectors would swap this for IndexIVFFlat or
similar, but exact search keeps behavior predictable and easy to explain.
"""

import logging
import os
import pickle

import faiss
import numpy as np

from embeddings.embedder import EMBEDDING_DIM, embed_texts
from models.schemas import Chunk

logger = logging.getLogger("raguard.faiss_store")

INDEX_PATH = "vectorstore/data/index.faiss"
ID_MAP_PATH = "vectorstore/data/id_map.pkl"


class FaissStore:
    def __init__(self):
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        # Maps FAISS's internal sequential position -> our chunk_id,
        # since FAISS itself only knows about row numbers, not our IDs.
        self.id_map: list[str] = []
        self._load()

    def _load(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(ID_MAP_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(ID_MAP_PATH, "rb") as f:
                self.id_map = pickle.load(f)
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")

    def save(self):
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(ID_MAP_PATH, "wb") as f:
            pickle.dump(self.id_map, f)

    def add_chunks(self, chunks: list[Chunk]):
        """Embed and add new chunks. Assumes duplicate-checking already
        happened upstream (in the indexing service) — this always inserts."""
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vectors = embed_texts(texts)
        self.index.add(vectors)
        self.id_map.extend([c.chunk_id for c in chunks])
        self.save()
        logger.info(f"Added {len(chunks)} vectors, index now has {self.index.ntotal}")

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return [(chunk_id, similarity_score), ...] for the top_k nearest chunks."""
        if self.index.ntotal == 0:
            return []

        query_vector = embed_texts([query])
        top_k = min(top_k, self.index.ntotal)
        scores, positions = self.index.search(query_vector, top_k)

        results = []
        for score, position in zip(scores[0], positions[0]):
            if position == -1:
                continue
            results.append((self.id_map[position], float(score)))
        return results


faiss_store = FaissStore()