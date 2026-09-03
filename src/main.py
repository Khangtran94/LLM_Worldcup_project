# src/main.py
"""Simple CLI to test the RAG pipeline."""

from __future__ import annotations

import argparse

from rag import answer
from monitoring import log_feedback


def ask_for_feedback(query_id: int) -> None:
    raw = input("Was this helpful? (y/n/skip): ").strip().lower()
    if raw in ("y", "yes"):
        log_feedback(query_id, True)
    elif raw in ("n", "no"):
        comment = input("Optional comment (enter to skip): ").strip() or None
        log_feedback(query_id, False, comment)
    # anything else (including blank / "skip") -> no feedback logged


def main() -> None:
    parser = argparse.ArgumentParser(description="World Cup RAG CLI")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print retrieved chunks before answering",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of chunks to retrieve (default: 12)",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Skip the feedback prompt after each answer",
    )
    args = parser.parse_args()

    print("World Cup RAG — type a question (or 'quit' to exit)")
    if args.debug:
        print("(debug mode ON — retrieved chunks will be shown)\n")
    else:
        print()

    while True:
        question = input("Question: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue

        print("\nThinking...")
        result, query_id, _meta = answer(question, top_k=args.top_k, debug=args.debug)
        print(f"\nAnswer: {result}\n")

        if not args.no_feedback:
            ask_for_feedback(query_id)

        print("-" * 60)


if __name__ == "__main__":
    main()
