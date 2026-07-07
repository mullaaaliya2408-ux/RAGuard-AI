"""
Combines semantic search (FAISS), keyword search (BM25), and metadata
matching into one ranked result list.

Merge strategy:
- Run semantic + BM25 search for each expanded query variant.
- Normalize each score type to a 0-1 range so they're comparable.
- Combine into a weighted score per chunk (semantic weighted higher, since
  it generalizes better than exact keyword overlap for natural questions).
- A chunk found by multiple methods/variants gets boosted, not double-counted.
"""

import logging

from models.schemas import RetrievedChunk, RetrievalResult
from vectorstore.chunk_store import chunk_store
from vectorstore.faiss_store import faiss_store
from services.bm25_index import bm25_index
from services.query_expansion import generate_query_variants

logger = logging.getLogger("raguard.hybrid_retriever")

SEMANTIC_WEIGHT = 0.6
BM25_WEIGHT = 0.3
METADATA_WEIGHT = 0.1

TOP_K_PER_METHOD = 8
FINAL_TOP_K = 10


def normalize_scores(scored: list[tuple[str, float]]) -> dict[str, float]:
    """Min-max normalize scores to 0-1 so semantic/BM25 scales become comparable."""
    if not scored:
        return {}
    values = [s for _, s in scored]
    lo, hi = min(values), max(values)
    if hi == lo:
        return {cid: 1.0 for cid, _ in scored}
    return {cid: (s - lo) / (hi - lo) for cid, s in scored}


def compute_metadata_score(query: str, chunk) -> float:
    """
    Simple metadata match: does the query mention the document name or
    section heading? A hit here is a strong relevance signal even without
    semantic or keyword overlap on the body text.
    """
    query_lower = query.lower()
    score = 0.0
    if chunk.section and chunk.section.lower() in query_lower:
        score += 0.7
    if chunk.document_name.lower().replace(".pdf", "") in query_lower:
        score += 0.3
    return min(score, 1.0)


def hybrid_search(query: str) -> RetrievalResult:
    """Main entry point: expand query, run both search methods, merge, re-rank."""
    variants = generate_query_variants(query)

    semantic_hits: dict[str, float] = {}
    bm25_hits: dict[str, float] = {}
    matched_via: dict[str, set[str]] = {}

    for variant in variants:
        for chunk_id, score in faiss_store.search(variant, top_k=TOP_K_PER_METHOD):
            semantic_hits[chunk_id] = max(semantic_hits.get(chunk_id, 0), score)
            matched_via.setdefault(chunk_id, set()).add("semantic")

        for chunk_id, score in bm25_index.search(variant, top_k=TOP_K_PER_METHOD):
            bm25_hits[chunk_id] = max(bm25_hits.get(chunk_id, 0), score)
            matched_via.setdefault(chunk_id, set()).add("bm25")

    normalized_semantic = normalize_scores(list(semantic_hits.items()))
    normalized_bm25 = normalize_scores(list(bm25_hits.items()))

    all_chunk_ids = set(semantic_hits) | set(bm25_hits)
    retrieved: list[RetrievedChunk] = []

    for chunk_id in all_chunk_ids:
        chunk = chunk_store.get_chunk(chunk_id)
        if chunk is None:
            continue  # defensive: index and store should always agree, but don't crash if not

        sem_score = normalized_semantic.get(chunk_id, 0.0)
        bm25_score = normalized_bm25.get(chunk_id, 0.0)
        meta_score = compute_metadata_score(query, chunk)

        combined = (
            SEMANTIC_WEIGHT * sem_score
            + BM25_WEIGHT * bm25_score
            + METADATA_WEIGHT * meta_score
        )

        retrieved.append(
            RetrievedChunk(
                chunk=chunk,
                semantic_score=round(sem_score, 3),
                bm25_score=round(bm25_score, 3),
                metadata_score=round(meta_score, 3),
                combined_score=round(combined, 3),
                matched_via=sorted(matched_via.get(chunk_id, [])),
            )
        )

    retrieved.sort(key=lambda r: r.combined_score, reverse=True)
    top_results = retrieved[:FINAL_TOP_K]

    average_similarity = (
        sum(r.semantic_score for r in top_results) / len(top_results)
        if top_results else 0.0
    )

    return RetrievalResult(
        query=query,
        expanded_queries=variants,
        results=top_results,
        average_similarity=round(average_similarity, 3),
    )