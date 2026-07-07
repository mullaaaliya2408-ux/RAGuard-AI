"""
Stores full Chunk objects keyed by chunk_id, and tracks which document_ids
have already been indexed to prevent duplicate embeddings on re-upload.

FAISS only stores vectors + an integer position -> we need this separate
store to map back from a FAISS result position to the actual Chunk content.
"""

import json
import logging
import os

from models.schemas import Chunk

logger = logging.getLogger("raguard.chunk_store")

STORE_PATH = "vectorstore/data/chunks.json"


class ChunkStore:
    def __init__(self, path: str = STORE_PATH):
        self.path = path
        self.chunks_by_id: dict[str, Chunk] = {}
        self.indexed_document_ids: set[str] = set()
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, "r") as f:
            raw = json.load(f)
        self.chunks_by_id = {cid: Chunk(**data) for cid, data in raw["chunks"].items()}
        self.indexed_document_ids = set(raw["indexed_document_ids"])
        logger.info(f"Loaded {len(self.chunks_by_id)} chunks from disk")

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        raw = {
            "chunks": {cid: chunk.model_dump() for cid, chunk in self.chunks_by_id.items()},
            "indexed_document_ids": list(self.indexed_document_ids),
        }
        with open(self.path, "w") as f:
            json.dump(raw, f)

    def is_document_indexed(self, document_id: str) -> bool:
        """Used to skip re-embedding a document that's already been indexed."""
        return document_id in self.indexed_document_ids

    def add_chunks(self, chunks: list[Chunk]):
        for chunk in chunks:
            self.chunks_by_id[chunk.chunk_id] = chunk
        if chunks:
            self.indexed_document_ids.add(chunks[0].document_id)
        self.save()

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self.chunks_by_id.get(chunk_id)

    def all_chunks(self) -> list[Chunk]:
        return list(self.chunks_by_id.values())


# Single shared instance used across the app (simple singleton pattern —
# fine for this project's scope; a multi-process deployment would use a
# real database instead).
chunk_store = ChunkStore()