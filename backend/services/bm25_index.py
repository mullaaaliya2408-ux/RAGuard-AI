"""
BM25 keyword index, complementing semantic search: catches exact-term
matches (names, codes, numbers) that embeddings sometimes miss or dilute.
Persisted to disk alongside the FAISS index so it survives restarts.
"""

import logging
import os
import pickle
import re

from rank_bm25 import BM25Okapi

from models.schemas import Chunk

logger = logging.getLogger("raguard.bm25")

BM25_PATH = "vectorstore/data/bm25.pkl"


def tokenize(text: str) -> list[str]:
    """Simple lowercase word tokenizer — BM25 doesn't need anything fancier."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    def __init__(self):
        self.chunk_ids: list[str] = []
        self.bm25: BM25Okapi | None = None
        self._load()

    def _load(self):
        if os.path.exists(BM25_PATH):
            with open(BM25_PATH, "rb") as f:
                data = pickle.load(f)
            self.chunk_ids = data["chunk_ids"]
            self.bm25 = data["bm25"]
            logger.info(f"Loaded BM25 index with {len(self.chunk_ids)} documents")

    def save(self):
        os.makedirs(os.path.dirname(BM25_PATH), exist_ok=True)
        with open(BM25_PATH, "wb") as f:
            pickle.dump({"chunk_ids": self.chunk_ids, "bm25": self.bm25}, f)

    def rebuild(self, all_chunks: list[Chunk]):
        """
        BM25Okapi doesn't support incremental insertion, so we rebuild the
        whole index from the current full chunk set. This is cheap enough
        at this project's scale (thousands, not millions, of chunks).
        """
        self.chunk_ids = [c.chunk_id for c in all_chunks]
        tokenized_corpus = [tokenize(c.text) for c in all_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        self.save()
        logger.info(f"Rebuilt BM25 index with {len(self.chunk_ids)} chunks")

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self.bm25 is None:
            return []

        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: x[1], reverse=True)
        # Filter out zero-score matches — a zero means no keyword overlap at all
        return [(cid, score) for cid, score in ranked[:top_k] if score > 0]


bm25_index = BM25Index()