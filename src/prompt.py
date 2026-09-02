# src/prompt.py
"""Build the grounded prompt for the LLM."""

from __future__ import annotations

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about FIFA World Cup history using only the provided context.

Rules:
1. Use ONLY the information in the context below. Do not invent facts.
2. You may combine facts from multiple context blocks.
3. If a context clearly states the winner, score, or a fact, use it.
4. If the context does not contain enough information to answer confidently, reply exactly:
   "I don't know based on the available information."
5. Be concise. Prefer short factual answers (e.g. "Argentina won the 2022 World Cup final.").
6. For totals (how many goals / matches a player has), only answer if the total is explicitly stated or you can safely count from the given contexts. Otherwise say you don't know."""


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
