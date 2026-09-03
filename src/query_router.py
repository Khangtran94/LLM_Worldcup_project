# src/query_router.py
"""
Deterministic router for question types that RAG handles poorly.

Two categories of questions consistently failed in testing:

1. Aggregate/count questions ("how many goals has Messi scored",
   "how many matches in the 2022 final") — these need an EXHAUSTIVE
   answer, but top-k similarity search only returns a random sample
   of relevant chunks. No amount of better chunking fixes this; you
   need a full-table SQL aggregate.

2. Exact-match lookups where the question names a specific year and/or
   round ("2026 world cup final winner") — these should be a hard SQL
   filter, not a soft embedding-score boost. Soft boosts are easy to
   get wrong (unit mismatch, boolean stored as text, etc.) and hard to
   debug because nothing errors, it just silently doesn't reorder.

If this router recognizes the question and can answer it confidently,
it returns a string. Otherwise it returns None and the caller should
fall through to normal RAG.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db.connection import get_connection
from chunk_matches import extract_goal_records  # reuse the same parser used at index time


# ---------------------------------------------------------------------------
# Shared extraction helpers (mirrors retrieve.py's heuristics)
# ---------------------------------------------------------------------------

def extract_year(question: str) -> int | None:
    match = re.search(r"\b(19[3-9]\d|20[0-2]\d)\b", question)
    if not match:
        return None
    year = int(match.group(1))
    return year if 1930 <= year <= 2030 else None


def wants_final(question: str) -> bool:
    q = question.lower()
    return "final" in q and "semi" not in q and "quarter" not in q


def extract_player_name(question: str) -> str | None:
    """
    Heuristic: pull out a run of 2+ Title-Case words, e.g. "Lionel Messi".
    Not perfect (misses single-word nicknames like "Pele", "Ronaldo" used
    alone, or ALL-CAPS surnames), but catches the common "First Last" case
    cleanly and avoids false positives on generic question words.
    """
    candidates = re.findall(r"\b[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)+\b", question)
    # Filter out things like "World Cup", "How Many" if question was oddly cased
    stopword_phrases = {"world cup", "how many", "fifa world cup"}
    for c in candidates:
        if c.lower() not in stopword_phrases:
            return c
    return None


# ---------------------------------------------------------------------------
# Aggregate: total goals / matches scored in, for a named player
# ---------------------------------------------------------------------------

def count_player_goals(player_name: str) -> dict[str, Any]:
    """
    Pull every 'goals' chunk that mentions the player, re-parse it with the
    same extract_goal_records() used at index time, and count individual
    goal lines (not just distinct matches) so multi-goal games are counted
    correctly.
    """
    name_lower = player_name.lower()

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT match_id, year, text
                FROM chunks
                WHERE chunk_type = 'goals' AND text ILIKE %s
                """,
                (f"%{player_name}%",),
            )
            rows = cur.fetchall()

    total_goals = 0
    matches_scored: set[str] = set()
    by_year: dict[int, int] = {}

    for match_id, year, text in rows:
        records = extract_goal_records(text.splitlines())
        for g in records:
            if name_lower in g["player"].lower():
                total_goals += 1
                matches_scored.add(match_id)
                if year is not None:
                    by_year[year] = by_year.get(year, 0) + 1

    return {
        "total_goals": total_goals,
        "matches": len(matches_scored),
        "by_year": by_year,
    }


# ---------------------------------------------------------------------------
# Aggregate: how many matches in a given round / year
# ---------------------------------------------------------------------------

def count_matches(year: int | None, round_is_final: bool) -> int:
    where = ["chunk_type = 'parent'"]
    params: list[Any] = []

    if year is not None:
        where.append("year = %s")
        params.append(year)

    if round_is_final:
        where.append("is_final = true")

    sql = f"SELECT COUNT(*) FROM chunks WHERE {' AND '.join(where)}"

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Router entry point
# ---------------------------------------------------------------------------

def try_direct_answer(question: str) -> str | None:
    q = question.lower()

    # --- "how many goals has X scored" / "how many matches did X score in" ---
    if re.search(r"how many", q) and ("goal" in q or "score" in q):
        name = extract_player_name(question)
        if name:
            result = count_player_goals(name)
            if result["total_goals"] > 0:
                return (
                    f"{name} scored {result['total_goals']} goal(s) "
                    f"in {result['matches']} match(es) in the World Cup dataset."
                )
            # No goals found for this name — don't fabricate; let RAG/LLM
            # try, or fall through to the "I don't know" behavior downstream.
            return None

    # --- "how many matches in the <year> final" / "...in the final" ---
    if "how many match" in q:
        year = extract_year(question)
        final = wants_final(question)
        if year is not None or final:
            count = count_matches(year, final)
            label_parts = []
            if year:
                label_parts.append(str(year))
            if final:
                label_parts.append("final")
            label = " ".join(label_parts) if label_parts else "matches"
            plural = "es" if count != 1 else ""
            verb = "were" if count != 1 else "was"
            return f"There {verb} {count} match{plural} for the {label}."

    return None
