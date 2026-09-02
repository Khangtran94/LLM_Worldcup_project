# src/retrieve.py
"""Retrieve top-k relevant chunks from Postgres (pgvector)."""

from __future__ import annotations

from db.connection import get_connection
from embedder import embed_query


def retrieve(
    question: str,
    top_k: int = 12,
    prefer_types: tuple[str, ...] = ("overview", "goals", "parent", "lineup"),
) -> list[dict]:
    """
    Embed the question and return the top-k most similar chunks.

    Returns a list of dicts:
        {"text": str, "chunk_type": str, "score": float, "match_id": str, "year": int | None}
    """
    query_embedding = embed_query(question)
    vector_literal = "[" + ",".join(f"{x:.8f}" for x in query_embedding) + "]"

    # Prefer more informative chunk types when scores are close
    type_priority = {t: i for i, t in enumerate(prefer_types)}

    sql = """
        SELECT
            text,
            chunk_type,
            match_id,
            year,
            1 - (embedding <=> %s::vector) AS score
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (vector_literal, vector_literal, top_k * 2))
            rows = cur.fetchall()

    results = [
        {
            "text": row[0],
            "chunk_type": row[1],
            "match_id": row[2],
            "year": row[3],
            "score": float(row[4]),
        }
        for row in rows
    ]

    # Re-rank slightly: boost preferred chunk types
    def sort_key(item: dict) -> tuple:
        priority = type_priority.get(item["chunk_type"], len(prefer_types))
        return (-item["score"], priority)

    results.sort(key=sort_key)
    return results[:top_k]
