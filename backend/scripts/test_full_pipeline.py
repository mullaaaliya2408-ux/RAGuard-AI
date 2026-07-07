"""
End-to-end test of the complete pipeline: understanding, retrieval,
quality checks, verification, reflection, and final answer generation.

Usage:
    python scripts/test_full_pipeline.py "your question"
"""

import sys

from services.pipeline import run_pipeline


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/test_full_pipeline.py "your question"')
        sys.exit(1)

    query = sys.argv[1]
    result = run_pipeline(query)

    print(f"\n{'='*60}")
    print(f"QUESTION: {query}")
    print(f"{'='*60}\n")

    print("FINAL ANSWER:")
    print(result.final_answer.final_answer)
    print(f"\nConfidence: {result.final_answer.confidence_score} ({result.final_answer.confidence_label})")

    print(f"\nEvidence Sources: {result.final_answer.evidence_sources}")
    print(f"Reasoning Summary: {result.final_answer.reasoning_summary}")

    if result.final_answer.warnings:
        print(f"\nWarnings:")
        for w in result.final_answer.warnings:
            print(f"  ⚠️  {w}")

    if result.final_answer.suggested_follow_up:
        print(f"\nSuggested follow-up: {result.final_answer.suggested_follow_up}")

    print(f"\n--- Pipeline trace ---")
    print(f"Self-correction succeeded: {result.self_correction.succeeded}")
    print(f"Attempts: {len(result.self_correction.attempts)}")
    if result.reflection:
        print(f"Reflection satisfied: {result.reflection.is_satisfied} | {result.reflection.reasoning}")
    if result.self_correction.conflict_report and result.self_correction.conflict_report.has_conflict:
        print(f"⚠️  Contradictions detected: {len(result.self_correction.conflict_report.conflicts)}")


if __name__ == "__main__":
    main()