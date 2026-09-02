# src/prompt.py
"""Build the grounded prompt for the LLM."""

from __future__ import annotations

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about FIFA World Cup history.
Answer ONLY using the information provided in the context below.
If the context does not contain enough information to answer the question, reply exactly:
"I don't know based on the available information."
Do not use any external knowledge. Be concise and accurate."""


def build_messages(question: str, contexts: list[str]) -> list[dict[str, str]]:
    """Create the messages list for the OpenAI Chat Completions API."""
    context_block = "\n\n".join(
        f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)
    )

    user_content = f"""Context:
{context_block}

Question: {question}

Answer:"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
