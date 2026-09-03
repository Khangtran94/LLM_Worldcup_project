# src/query_router.py
"""
Deterministic router for question types that RAG handles poorly.

Four categories of questions consistently failed in testing:

1. Aggregate/count questions ("how many goals has Messi scored",
   "how many matches in the 2022 final") — these need an EXHAUSTIVE
   answer, but top-k similarity search only returns a random sample
   of relevant chunks. No amount of better chunking fixes this; you
   need a full-table SQL aggregate.

2. Exact-match lookups where the question names a specific year and/or
   round ("2026 world cup final winner") — these should be a hard SQL
   filter, not a soft embedding-score boost.

3. Team win counts ("how many World Cups has Brazil won", "which team
   has won the most World Cups") — same exhaustiveness problem as (1):
   counting every final a team won requires scanning every parent chunk
   with is_final=true, not a top-k sample.

4. Total tournament count ("how many World Cups have been held") — this
   isn't about any single match at all, so similarity search has
   nothing good to retrieve; it's a COUNT(DISTINCT year) over completed
   finals.

If this router recognizes the question and can answer it confidently,
it returns a string. Otherwise it returns None and the caller should
fall through to normal RAG.
"""

from __future__ import annotations

import re
import sys
from functools import lru_cache
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


@lru_cache(maxsize=1)
def get_known_teams() -> tuple[str, ...]:
    """
    Every team name that appears in the dataset, longest-first so a
    substring match tries "West Germany" before "Germany" and doesn't
    stop short.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT team1 FROM chunks WHERE team1 IS NOT NULL
                UNION
                SELECT team2 FROM chunks WHERE team2 IS NOT NULL
                """
            )
            teams = {row[0] for row in cur.fetchall()}
    return tuple(sorted(teams, key=len, reverse=True))


def extract_team_name(question: str) -> str | None:
    """Match a known team name in the question, whole-word, longest name first."""
    q_lower = question.lower()
    for team in get_known_teams():
        if re.search(rf"\b{re.escape(team.lower())}\b", q_lower):
            return team
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
# Team win counts: "how many World Cups has X won" / "who's won the most"
# ---------------------------------------------------------------------------

def count_team_wins(team_name: str) -> int:
    """Exhaustive count of finals a given team won."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(DISTINCT match_id)
                FROM chunks
                WHERE chunk_type = 'parent' AND is_final = true AND winner ILIKE %s
                """,
                (team_name,),
            )
            return cur.fetchone()[0]


def team_win_years(team_name: str) -> list[int]:
    """Every year a given team won the final, ascending."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT year
                FROM chunks
                WHERE chunk_type = 'parent' AND is_final = true AND winner ILIKE %s
                ORDER BY year
                """,
                (team_name,),
            )
            return [row[0] for row in cur.fetchall() if row[0] is not None]


def wants_win_years(question: str) -> bool:
    """
    Distinguishes "which year(s) did X win" from "how many times did X
    win" — same team+win intent, different shape of answer expected.
    """
    q = question.lower()
    if "how many" in q:
        return False
    return "year" in q or "when" in q


def top_winning_teams(limit: int = 5) -> list[tuple[str, int]]:
    """Leaderboard of most World Cup titles won, most first."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT winner, COUNT(DISTINCT match_id) AS wins
                FROM chunks
                WHERE chunk_type = 'parent' AND is_final = true AND winner IS NOT NULL
                GROUP BY winner
                ORDER BY wins DESC, winner ASC
                LIMIT %s
                """,
                (limit,),
            )
            return cur.fetchall()


def wants_team_win_query(question: str) -> tuple[str | None, bool]:
    """
    Returns (team_name, is_leaderboard_query).
      - team_name set        -> "how many World Cups has <team> won"
      - is_leaderboard=True  -> "which team has won the most World Cups"
      - both None/False      -> not this kind of question
    """
    q = question.lower()
    if "world cup" not in q:
        return None, False
    if not any(w in q for w in ("won", "win", "title", "champion")):
        return None, False

    team = extract_team_name(question)
    if team:
        return team, False

    if "most" in q or "which team" in q or "who has won" in q:
        return None, True

    return None, False


# ---------------------------------------------------------------------------
# Total tournament count: "how many World Cups have been held"
# ---------------------------------------------------------------------------

def count_world_cups_held() -> int:
    """
    COUNT(DISTINCT year) over finals that have a recorded winner — i.e.
    tournaments that actually completed in the dataset, not just ones
    with fixtures loaded (e.g. an in-progress or future tournament with
    no result yet wouldn't count).
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(DISTINCT year)
                FROM chunks
                WHERE chunk_type = 'parent' AND is_final = true AND winner IS NOT NULL
                """
            )
            return cur.fetchone()[0]


def wants_world_cup_total(question: str) -> bool:
    q = question.lower()
    if "world cup" not in q or "how many" not in q:
        return False
    # exclude questions about goals/teams/wins so it doesn't collide
    # with count_player_goals / wants_team_win_query
    if any(w in q for w in ("goal", "score", "won", "win", "team")):
        return False
    return any(w in q for w in ("held", "been", "played", "so far", "total"))


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

    # --- "how many World Cups has X won" / "which team has won the most" ---
    team, is_leaderboard = wants_team_win_query(question)
    if is_leaderboard:
        board = top_winning_teams()
        if board:
            top_name, top_wins = board[0]
            others = ", ".join(f"{name} ({wins})" for name, wins in board[1:5])
            tail = f" Next: {others}." if others else ""
            return (
                f"{top_name} has won the most World Cups in the dataset, "
                f"with {top_wins} title(s).{tail}"
            )
        return None
    if team:
        if wants_win_years(question):
            years = team_win_years(team)
            if not years:
                return f"{team} has not won the World Cup in the dataset."
            years_str = ", ".join(str(y) for y in years)
            return f"{team} has won the World Cup in {len(years)} year(s): {years_str}."
        wins = count_team_wins(team)
        plural = "s" if wins != 1 else ""
        return f"{team} has won the World Cup {wins} time{plural} in the dataset."

    # --- "how many World Cups have been held" ---
    if wants_world_cup_total(question):
        count = count_world_cups_held()
        plural = "s" if count != 1 else ""
        return (
            f"There {'have' if count != 1 else 'has'} been {count} completed "
            f"FIFA World Cup{plural} in the dataset (tournaments with a recorded "
            f"final result)."
        )

    return None
