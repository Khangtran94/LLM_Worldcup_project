# src/retrieve.py
"""Retrieve top-k relevant chunks from Postgres (pgvector).

Change from the original version: year/"final" intent is now a HARD SQL
WHERE filter instead of a soft +0.12 / +0.15 score boost. The boost
approach wasn't reliably surfacing the correct match in testing (e.g.
"2026 world cup final" returned a mix of 2022/2026 chunks from unrelated
rounds) — a hard filter guarantees the right rows are even in the
candidate pool, rather than hoping the nudge outweighs whatever the raw
cosine similarity happened to be (which is often nearly flat across
chunks, since most match documents share almost identical boilerplate).
"""

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


def _run_query(
    vector_literal: str,
    limit: int,
    year: int | None,
    final_only: bool,
) -> list[tuple]:
    where_clauses: list[str] = []
    params: list[Any] = []

    if year is not None:
        where_clauses.append("year = %s")
        params.append(year)
    if final_only:
        where_clauses.append("is_final = true")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    sql = f"""
        SELECT
            text,
            chunk_type,
            match_id,
            year,
            is_final,
            1 - (embedding <=> %s::vector) AS score
        FROM chunks
        {where_sql}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    query_params = [vector_literal, *params, vector_literal, limit]

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, query_params)
            return cur.fetchall()


def retrieve(
    question: str,
    top_k: int = 12,
    prefer_types: tuple[str, ...] = ("overview", "goals", "parent", "lineup"),
) -> list[dict[str, Any]]:
    """
    Embed the question and return the top-k most similar chunks.

    Strategy:
    - If the question names a specific year and/or clearly wants "the final",
      apply those as a HARD SQL filter first (guarantees relevant rows are
      in the pool).
    - If that filtered query comes back empty or too small (e.g. the year
      regex misfired, or metadata is missing for some rows), fall back to
      an unfiltered semantic search so we never return zero results.
    - Mild type-priority tie-breaking is still applied after retrieval.
    """
    query_embedding = embed_query(question)
    vector_literal = "[" + ",".join(f"{x:.8f}" for x in query_embedding) + "]"

    year = _extract_year(question)
    final_only = _wants_final(question)
    type_priority = {t: i for i, t in enumerate(prefer_types)}

    rows: list[tuple] = []
    if year is not None or final_only:
        rows = _run_query(vector_literal, top_k * 3, year, final_only)

    # Fallback: filter was too narrow (or metadata gap) -> go unfiltered
    if len(rows) < top_k:
        fallback_rows = _run_query(vector_literal, top_k * 3, None, False)
        # Merge, preferring filtered rows first, de-duped by match_id+chunk_type+text
        seen = set()
        merged: list[tuple] = []
        for row in rows + fallback_rows:
            key = (row[2], row[1], row[0][:60])  # match_id, chunk_type, text prefix
            if key not in seen:
                seen.add(key)
                merged.append(row)
        rows = merged

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
        # Small residual boosts just for tie-breaking within the (already
        # filtered) pool — the heavy lifting is done by the SQL WHERE clause.
        boosted = item["score"]
        if year is not None and item["year"] == year:
            boosted += 0.05
        if final_only and item["is_final"]:
            boosted += 0.05
        priority = type_priority.get(item["chunk_type"], len(prefer_types))
        return (-boosted, priority)

    results.sort(key=sort_key)
    return results[:top_k]
