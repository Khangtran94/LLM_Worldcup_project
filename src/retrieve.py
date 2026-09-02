# src/retrieve.py
"""Retrieve top-k relevant chunks from Postgres (pgvector) with light metadata boosting."""

from __future__ import annotations

import re
from typing import Any

from db.connection import get_connection
from embedder import embed_query


def _extract_year(question: str) -> int | None:
    """Return the first 4-digit year that looks like a World Cup year."""
    match = re.search(r"\b(19[3-9]\d|20[0-2]\d)\b", question)
    if not match:
        return None
    year = int(match.group(1))
    if 1930 <= year <= 2030:
        return year
    return None


def _wants_final(question: str) -> bool:
    q = question.lower()
    return "final" in q and "semi" not in q and "quarter" not in q


def retrieve(
    question: str,
    top_k: int = 12,
    prefer_types: tuple[str, ...] = ("overview", "goals", "parent", "lineup"),
) -> list[dict[str, Any]]:
    """
    Embed the question and return the top-k most similar chunks.

    Light metadata boosting:
    - Prefer chunks from the year mentioned in the question
    - Prefer is_final=true when the question asks about a final
    - Prefer overview / goals / parent chunk types
    """
    query_embedding = embed_query(question)
    vector_literal = "[" + ",".join(f"{x:.8f}" for x in query_embedding) + "]"

    year = _extract_year(question)
    wants_final = _wants_final(question)
    type_priority = {t: i for i, t in enumerate(prefer_types)}

    # Fetch a larger candidate pool, then re-rank with metadata boosts
    sql = """
        SELECT
            text,
            chunk_type,
            match_id,
            year,
            is_final,
            1 - (embedding <=> %s::vector) AS score
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (vector_literal, vector_literal, top_k * 3))
            rows = cur.fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        text, chunk_type, match_id, row_year, is_final, score = row
        results.append(
            {
                "text": text,
                "chunk_type": chunk_type,
                "match_id": match_id,
                "year": row_year,
                "is_final": bool(is_final) if is_final is not None else False,
                "score": float(score),
            }
        )

    def sort_key(item: dict[str, Any]) -> tuple:
        boosted = item["score"]

        # Strong boost for matching year
        if year is not None and item["year"] == year:
            boosted += 0.12

        # Strong boost when user asks for the final
        if wants_final and item["is_final"]:
            boosted += 0.15

        # Mild preference for informative chunk types
        priority = type_priority.get(item["chunk_type"], len(prefer_types))

        return (-boosted, priority)

    results.sort(key=sort_key)
    return results[:top_k]
