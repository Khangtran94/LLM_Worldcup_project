# src/prompt.py
"""Build the grounded prompt for the LLM."""

from __future__ import annotations

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about FIFA World Cup history using only the provided context.

Rules:
1. Use ONLY the information in the context below. Do not use external knowledge.
2. You may combine facts from multiple context blocks if they help answer the question.
3. If the context does not contain enough information to answer confidently, reply exactly:
   "I don't know based on the available information."
4. Be concise and accurate. Prefer short factual answers.
5. When the question asks for a number (goals, matches, etc.) and the context only shows individual matches, say you don't know the total unless the total is explicitly stated."""


def build_messages(question: str, contexts: list[dict]) -> list[dict[str, str]]:
    """Create the messages list for the OpenAI Chat Completions API."""
    context_block = "\n\n".join(
        f"[Context {i+1} | {c.get('chunk_type', '?')} | score={c.get('score', 0):.3f}]\n{c['text']}"
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
