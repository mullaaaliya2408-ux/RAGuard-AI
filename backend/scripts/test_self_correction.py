"""
Standalone script to observe the self-correction loop in action —
especially useful for testing a deliberately vague query to see the
rewrite chain kick in.

Usage:
    python scripts/test_self_correction.py "what is attendance?"
"""

import sys

from services.self_correction_loop import run_self_correction_loop


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/test_self_correction.py "your question"')
        sys.exit(1)

    query = sys.argv[1]
    result = run_self_correction_loop(query)

    print(f"\nOriginal query: {result.original_query}")
    print(f"Intent: {result.query_analysis.intent} | Question type: {result.query_analysis.question_type}")
    print(f"Ambiguous: {result.query_analysis.is_ambiguous}")
    print(f"Succeeded: {result.succeeded}\n")

    print("--- Attempts ---")
    for i, attempt in enumerate(result.attempts, 1):
        tag = "(rewrite)" if attempt.was_rewrite else "(original)"
        print(f"{i}. {tag} '{attempt.query_used}' -> overall_score={attempt.quality.overall_score} "
              f"passed={attempt.quality.passed}")
        if attempt.quality.reasons:
            print(f"   reasons: {attempt.quality.reasons}")

    print(f"\n--- Final Retrieval ({len(result.final_retrieval.results)} chunks) ---")
    for r in result.final_retrieval.results[:3]:
        print(f"[{r.combined_score}] Page {r.chunk.page_number}: {r.chunk.text[:120]}...")


if __name__ == "__main__":
    main()