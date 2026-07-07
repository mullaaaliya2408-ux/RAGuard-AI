"""
Query endpoints. /retrieve exposes raw hybrid search (kept for debugging,
from Milestone 4). /ask runs the full self-correction loop — this is what
the frontend's Ask page will call once Answer Generation exists.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from services.hybrid_retriever import hybrid_search
from services.self_correction_loop import run_self_correction_loop
from services.pipeline import run_pipeline

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    query: str


@router.post("/retrieve")
def retrieve(request: QueryRequest):
    """Raw single-pass hybrid retrieval — useful for debugging retrieval alone."""
    result = hybrid_search(request.query)
    return result.model_dump()


@router.post("/ask")
def ask(request: QueryRequest):
    """
    Full pipeline: query understanding -> retrieve -> quality check ->
    verify -> reflect -> rewrite/retry as needed -> generate final answer.
    Returns the complete structured answer plus the full reasoning trail.
    """
    result = run_pipeline(request.query)
    return result.model_dump()