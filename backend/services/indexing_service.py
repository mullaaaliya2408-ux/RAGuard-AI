"""
Orchestrates indexing a document's chunks into both the FAISS vector store
and the BM25 keyword index, while avoiding duplicate embedding work if a
document has already been indexed.
"""

import logging

from models.schemas import Chunk
from vectorstore.chunk_store import chunk_store
from vectorstore.faiss_store import faiss_store
from services.bm25_index import bm25_index

logger = logging.getLogger("raguard.indexing")


def index_chunks(chunks: list[Chunk]) -> bool:
    """
    Index a document's chunks. Returns False (and skips work) if this
    document_id has already been indexed, preventing duplicate embeddings
    on repeated uploads of the same file.
    """
    if not chunks:
        return False

    document_id = chunks[0].document_id
    if chunk_store.is_document_indexed(document_id):
        logger.info(f"Document {document_id} already indexed, skipping")
        return False

    chunk_store.add_chunks(chunks)
    faiss_store.add_chunks(chunks)
    bm25_index.rebuild(chunk_store.all_chunks())  # BM25 requires full rebuild

    logger.info(f"Indexed {len(chunks)} chunks for document {document_id}")
    return True