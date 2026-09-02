# src/retrieve.py
"""Retrieve top-k relevant chunks from Postgres (pgvector)."""

from __future__ import annotations

from db.connection import get_connection
from embedder import embed_query


def retrieve(question: str, top_k: int = 8) -> list[str]:
    """
    Embed the question and return the text of the top-k most similar chunks.
    """
    query_embedding = embed_query(question)
    vector_literal = "[" + ",".join(f"{x:.8f}" for x in query_embedding) + "]"

    sql = """
        SELECT text
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (vector_literal, top_k))
            rows = cur.fetchall()

    return [row[0] for row in rows]
