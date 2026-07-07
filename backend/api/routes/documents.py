"""
Read-only endpoints over already-indexed documents/chunks, used by the
Evidence Viewer page. No new storage needed -- this reads from the same
ChunkStore built in Milestone 4.
"""

from fastapi import APIRouter, HTTPException

from vectorstore.chunk_store import chunk_store

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("/list")
def list_documents():
    """
    Group all indexed chunks by document_id and return a summary per
    document -- name, chunk count, page range, average OCR confidence.
    """
    chunks = chunk_store.all_chunks()
    grouped: dict[str, list] = {}

    for chunk in chunks:
        grouped.setdefault(chunk.document_id, []).append(chunk)

    summaries = []
    for document_id, doc_chunks in grouped.items():
        ocr_scores = [c.ocr_confidence for c in doc_chunks if c.ocr_confidence is not None]
        summaries.append({
            "document_id": document_id,
            "document_name": doc_chunks[0].document_name,
            "chunk_count": len(doc_chunks),
            "page_count": len({c.page_number for c in doc_chunks}),
            "average_ocr_confidence": round(sum(ocr_scores) / len(ocr_scores), 3) if ocr_scores else None,
        })

    return {"documents": summaries}


@router.get("/{document_id}/chunks")
def get_document_chunks(document_id: str):
    """Return all chunks belonging to a single document, ordered by position."""
    chunks = [c for c in chunk_store.all_chunks() if c.document_id == document_id]
    if not chunks:
        raise HTTPException(status_code=404, detail="Document not found or has no indexed chunks")

    chunks.sort(key=lambda c: c.chunk_index)
    return {"document_id": document_id, "chunks": [c.model_dump() for c in chunks]}