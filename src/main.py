# src/main.py
"""Simple CLI to test the RAG pipeline."""

from rag import answer


def main() -> None:
    print("World Cup RAG — type a question (or 'quit' to exit)\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue

        print("\nThinking...")
        result = answer(question)
        print(f"\nAnswer: {result}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
