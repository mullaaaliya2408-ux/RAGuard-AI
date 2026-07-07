"""
Runs the full benchmark suite through the pipeline and computes aggregate
metrics (spec: EVALUATION PAGE -- Accuracy, Hallucination Rate, Precision,
Recall, Latency, Confidence Distribution).

"was_answered" is determined by checking whether the final answer matches
the standard refusal string from answer_generator.py -- a simple but
reliable signal, since that string is only ever returned by the
hallucination-prevention gate.
"""

import logging
import time

from evaluation.benchmark_questions import BENCHMARK_QUESTIONS
from models.schemas import BenchmarkResult, EvaluationSummary
from services.pipeline import run_pipeline
from services.answer_generator import NO_EVIDENCE_ANSWER

logger = logging.getLogger("raguard.evaluator")


def run_single_benchmark(question_id: int, question: str, expected_answerable: bool) -> BenchmarkResult:
    start = time.perf_counter()
    result = run_pipeline(question)
    latency_ms = (time.perf_counter() - start) * 1000

    was_answered = result.final_answer.final_answer.strip() != NO_EVIDENCE_ANSWER
    is_correct = was_answered == expected_answerable

    return BenchmarkResult(
        question_id=question_id,
        question=question,
        expected_answerable=expected_answerable,
        was_answered=was_answered,
        is_correct=is_correct,
        confidence_score=result.final_answer.confidence.score,
        confidence_label=result.final_answer.confidence.label,
        latency_ms=round(latency_ms, 1),
        attempts_used=len(result.self_correction.attempts),
    )


def run_full_evaluation() -> EvaluationSummary:
    """Run all benchmark questions and compute aggregate metrics."""
    results = [
        run_single_benchmark(q.id, q.question, q.expected_answerable)
        for q in BENCHMARK_QUESTIONS
    ]

    total = len(results)
    correct = sum(1 for r in results if r.is_correct)

    expected_true = [r for r in results if r.expected_answerable]
    expected_false = [r for r in results if not r.expected_answerable]

    hallucinations = sum(1 for r in expected_false if r.was_answered)
    hallucination_rate = hallucinations / len(expected_false) if expected_false else 0.0

    answered = [r for r in results if r.was_answered]
    true_positives = sum(1 for r in answered if r.expected_answerable)
    precision = true_positives / len(answered) if answered else 0.0
    recall = true_positives / len(expected_true) if expected_true else 0.0

    confidence_distribution = {"High": 0, "Medium": 0, "Low": 0}
    for r in results:
        confidence_distribution[r.confidence_label] += 1

    logger.info(f"Evaluation complete: accuracy={correct/total:.2f}, hallucination_rate={hallucination_rate:.2f}")

    return EvaluationSummary(
        total_questions=total,
        accuracy=round(correct / total, 3),
        hallucination_rate=round(hallucination_rate, 3),
        precision=round(precision, 3),
        recall=round(recall, 3),
        average_latency_ms=round(sum(r.latency_ms for r in results) / total, 1),
        average_confidence=round(sum(r.confidence_score for r in results) / total, 3),
        confidence_distribution=confidence_distribution,
        results=results,
    )