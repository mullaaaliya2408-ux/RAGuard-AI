"""
Standalone script to test hybrid retrieval quality directly against
whatever has already been indexed (via prior uploads).

Usage:
    python scripts/test_retrieval.py "What is the attendance policy?"
"""

import sys

from services.hybrid_retriever import hybrid_search


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/test_retrieval.py "your question"')
        sys.exit(1)

    query = sys.argv[1]
    result = hybrid_search(query)

    print(f"\nQuery: {result.query}")
    print(f"Expanded queries: {result.expanded_queries}")
    print(f"Average similarity: {result.average_similarity}")
    print(f"Results: {len(result.results)}\n")

    for r in result.results:
        print(f"[{r.combined_score}] (sem={r.semantic_score} bm25={r.bm25_score} meta={r.metadata_score}) "
              f"via={r.matched_via}")
        print(f"  Page {r.chunk.page_number} | {r.chunk.document_name} | Section: {r.chunk.section}")
        print(f"  {r.chunk.text[:150]}...\n")


if __name__ == "__main__":
    main()