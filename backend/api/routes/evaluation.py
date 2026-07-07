"""
Evaluation endpoint: runs the full 15-question benchmark suite and returns
aggregate metrics for the Evaluation page. This is a synchronous, blocking
call (15 questions x multiple LLM calls each) -- expect this to take a
minute or two, which the frontend surfaces as a clear loading state.
"""

from fastapi import APIRouter

from evaluation.evaluator import run_full_evaluation

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/run")
def run_evaluation():
    """Run all benchmark questions through the pipeline and return metrics."""
    summary = run_full_evaluation()
    return summary.model_dump()