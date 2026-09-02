# src/prompt.py
"""Build the grounded prompt for the LLM."""

from __future__ import annotations

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about FIFA World Cup history using only the provided context.

Rules:
1. Use ONLY the information in the context below. Do not invent facts from outside knowledge.
2. You may combine facts from multiple context blocks.
3. If the context clearly contains the answer (winner, score, year, number of matches in a specific round, etc.), give a short direct answer.
4. A World Cup final is one match. If the question asks how many matches are in a specific final and a final is present in the context, the answer is 1.
5. For player totals (how many goals / matches overall), only answer if you can reliably count from the given contexts or the total is explicitly stated. Otherwise reply exactly:
   "I don't know based on the available information."
6. Be concise and factual."""


def build_messages(question: str, contexts: list[dict]) -> list[dict[str, str]]:
    """Create the messages list for the OpenAI Chat Completions API."""
    context_block = "\n\n".join(
        f"[Context {i+1} | {c.get('chunk_type', '?')} | year={c.get('year')} | final={c.get('is_final')} | score={c.get('score', 0):.3f}]\n{c['text']}"
        for i, c in enumerate(contexts)
    )

    user_content = f"""Context:
{context_block}

Question: {question}

Answer:"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
