# src/main.py
"""Simple CLI to test the RAG pipeline."""

from __future__ import annotations

import argparse

from rag import answer


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
        result = answer(question, top_k=args.top_k, debug=args.debug)
        print(f"\nAnswer: {result}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
