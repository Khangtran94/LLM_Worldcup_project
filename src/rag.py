# src/rag.py
"""Main RAG pipeline: retrieve → prompt → LLM."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

from prompt import build_messages
from retrieve import retrieve

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"


def answer(question: str, top_k: int = 12, debug: bool = False) -> str:
    """
    Full RAG flow.
    Returns the model's answer (no sources in the final text).
    Set debug=True to print retrieved chunks.
    """
    contexts = retrieve(question, top_k=top_k)

    if debug:
        print("\n=== Retrieved chunks ===")
        for i, c in enumerate(contexts, 1):
            preview = c["text"][:130].replace("\n", " ")
            print(
                f"{i:2}. [{c['chunk_type']:12}] score={c['score']:.3f} "
                f"year={c.get('year')} final={c.get('is_final')} | {preview}..."
            )
        print("========================\n")

    messages = build_messages(question, contexts)

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,
    )

    return response.choices[0].message.content.strip()
