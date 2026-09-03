# src/chunk_matches.py
"""
Structure-aware chunking for FIFA World Cup match documents.

Reads  data/processed/matches.jsonl
Writes data/processed/match_chunks.jsonl

Design
------
Parent–child structure-aware strategy:

  parent  → full match document (chunk_type="parent")
  children → one chunk per natural section:
               overview | goals | lineup | substitutions |
               bookings | penalties | referees

This preserves the inherent structure of each match while still
allowing precise retrieval on individual sections.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "matches.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "match_chunks.jsonl"

# Sections that appear in the transformed match text
SECTION_NAMES = {
    "Score",
    "Goals",
    "Lineups",
    "Substitutions",
    "Bookings",
    "Penalty shootout",
    "Referees",
}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Structure parsing
# ---------------------------------------------------------------------------

def parse_sections(text: str) -> dict[str, list[str]]:
    """Split match text into its named structural sections."""
    lines = [ln.strip() for ln in text.splitlines()]
    sections: dict[str, list[str]] = {}
    current: str | None = None

    for line in lines:
        if not line:
            continue
        if line in SECTION_NAMES:
            current = line
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)

    return sections


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def clean_player_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s+\(captain\)", "", name, flags=re.IGNORECASE)
    return name


def extract_players_from_lineups(lineup_lines: list[str]) -> dict[str, list[str]]:
    """
    {team: [players]}

    Handles both formats produced by the fixed transform:
        France: - Alex THEPOT
        France: - Alex VILLAPLANE (captain)
        Mexico bench: Oscar BONFIGLIO
    """
    lineups: dict[str, list[str]] = {}

    for line in lineup_lines:
        # Starter
        m = re.match(r"^(.+?):\s*-\s*(.+)$", line)
        if m:
            team, player = m.group(1).strip(), clean_player_name(m.group(2))
            if team and player:
                lineups.setdefault(team, [])
                if player not in lineups[team]:
                    lineups[team].append(player)
            continue

        # Bench
        m = re.match(r"^(.+?)\s+bench:\s*(.+)$", line, flags=re.IGNORECASE)
        if m:
            team, player = m.group(1).strip(), clean_player_name(m.group(2))
            if team and player:
                lineups.setdefault(team, [])
                if player not in lineups[team]:
                    lineups[team].append(player)

    return lineups


def extract_goal_records(goal_lines: list[str]) -> list[dict[str, str]]:
    goals: list[dict[str, str]] = []
    for line in goal_lines:
        m = re.match(
            r"^(.+?):\s*-\s*(.+?)\s*\(([^)]+)\)\s*(.*)$",
            line,
        )
        if not m:
            continue
        goals.append(
            {
                "team": m.group(1).strip(),
                "player": clean_player_name(m.group(2)),
                "minute": m.group(3).strip(),
                "note": m.group(4).strip(),
            }
        )
    return goals


def extract_goal_scorers(goal_lines: list[str]) -> list[str]:
    scorers: list[str] = []
    for g in extract_goal_records(goal_lines):
        if g["player"] and g["player"] not in scorers:
            scorers.append(g["player"])
    return scorers


def extract_score(text: str, sections: dict[str, list[str]]) -> dict[str, Any]:
    """Support both score formats from transform.py."""
    result: dict[str, Any] = {
        "final_score": None,
        "full_time": None,
        "extra_time": None,
        "penalties": None,
    }

    score_lines = sections.get("Score", [])
    score_text = " ".join(score_lines) if score_lines else text

    # Preferred clean format
    m = re.search(r"Final score:\s*(.+)", score_text, flags=re.IGNORECASE)
    if m:
        result["final_score"] = m.group(1).strip()

    # Legacy formats
    m = re.search(r"Full time:\s*\[?([^\];]+)\]?", score_text, flags=re.IGNORECASE)
    if m:
        result["full_time"] = m.group(1).strip()

    m = re.search(r"Extra time:\s*\[?([^\];]+)\]?", score_text, flags=re.IGNORECASE)
    if m:
        result["extra_time"] = m.group(1).strip()

    m = re.search(
        r"Penalt(?:y|ies)(?:\s+shootout)?:\s*\[?([^\];]+)\]?",
        score_text,
        flags=re.IGNORECASE,
    )
    if m:
        result["penalties"] = m.group(1).strip()

    return result


def _parse_two_numbers(text: str | None) -> tuple[int, int] | None:
    """
    Parse the first two integers out of a free-form score string like
    "3, 2", "3-2", or "[3, 2]". Returns None if it doesn't cleanly parse.
    """
    if not text:
        return None
    nums = re.findall(r"-?\d+", text)
    if len(nums) < 2:
        return None
    try:
        return int(nums[0]), int(nums[1])
    except ValueError:
        return None


def derive_result(
    metadata: dict[str, Any],
    goals: list[dict[str, str]],
    score: dict[str, Any],
) -> dict[str, Any]:
    """
    Determine the winner.

    IMPORTANT: a penalty shootout winner takes priority over the
    in-play goal tally. Matches decided on penalties are tied (often
    0-0 or otherwise level) during normal + extra time — e.g. the 1994
    final (Brazil beat Italy on penalties, 0-0 in play), 2006 final
    (Italy beat France on penalties, 1-1 in play), and 2022 final
    (Argentina beat France on penalties, 3-3 in play). Shootout kicks
    aren't recorded as "goals" in the Goals section at all, so summing
    goals alone silently produces winner=None (and is_draw=True) for
    every final ever decided by a shootout. This previously undercounted
    team World Cup win totals and wrote "Result: Draw" into the overview
    text of finals a team actually won.
    """
    team1 = metadata.get("team1")
    team2 = metadata.get("team2")

    winner = None

    penalty_score = _parse_two_numbers(score.get("penalties"))
    if penalty_score is not None:
        p1, p2 = penalty_score
        if p1 > p2:
            winner = team1
        elif p2 > p1:
            winner = team2
        # p1 == p2 shouldn't happen in real football; if it does, fall
        # through to goal-based logic below rather than guessing.

    if winner is None and goals:
        t1 = sum(1 for g in goals if g["team"] == team1)
        t2 = sum(1 for g in goals if g["team"] == team2)
        if t1 > t2:
            winner = team1
        elif t2 > t1:
            winner = team2

    round_name = str(metadata.get("round", "")).strip().lower()

    return {
        "winner": winner,
        "is_draw": winner is None and penalty_score is None and bool(goals),
        "is_final": round_name == "final",
        "went_to_extra_time": score.get("extra_time") is not None,
        "had_penalties": score.get("penalties") is not None,
    }


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def make_base_metadata(record: dict[str, Any]) -> dict[str, Any]:
    meta = dict(record.get("metadata", {}))
    text = record.get("text", "")
    sections = parse_sections(text)

    lineup_lines = sections.get("Lineups", [])
    goal_lines = sections.get("Goals", [])

    lineups = extract_players_from_lineups(lineup_lines)
    goal_records = extract_goal_records(goal_lines)
    goal_scorers = extract_goal_scorers(goal_lines)
    score = extract_score(text, sections)
    result = derive_result(meta, goal_records, score)

    players: list[str] = []
    for team_players in lineups.values():
        for p in team_players:
            if p not in players:
                players.append(p)

    teams = [t for t in (meta.get("team1"), meta.get("team2")) if t]

    return {
        "match_id": record["id"],
        "year": meta.get("year"),
        "tournament": meta.get("tournament"),
        "round": meta.get("round"),
        "date": meta.get("date"),
        "team1": meta.get("team1"),
        "team2": meta.get("team2"),
        "teams": teams,
        "ground": meta.get("ground"),
        "winner": result["winner"],
        "is_draw": result["is_draw"],
        "is_final": result["is_final"],
        "went_to_extra_time": result["went_to_extra_time"],
        "had_penalties": result["had_penalties"],
        "players": players,
        "goal_scorers": goal_scorers,
        "final_score": score.get("final_score"),
    }


# ---------------------------------------------------------------------------
# Chunk builders
# ---------------------------------------------------------------------------

def make_chunk(
    *,
    match_id: str,
    chunk_type: str,
    chunk_index: int,
    text: str,
    metadata: dict[str, Any],
    parent_id: str | None = None,
) -> dict[str, Any]:
    meta = {
        **metadata,
        "chunk_type": chunk_type,
        "chunk_index": chunk_index,
    }
    if parent_id is not None:
        meta["parent_id"] = parent_id

    return {
        "id": f"{match_id}_{chunk_type}_{chunk_index}",
        "text": text.strip(),
        "metadata": meta,
    }


def build_overview_text(record: dict[str, Any], metadata: dict[str, Any]) -> str:
    original = record.get("metadata", {})
    team1 = original.get("team1", "")
    team2 = original.get("team2", "")

    lines = [
        f"{original.get('tournament', 'FIFA World Cup')} — {original.get('round', '')}",
        "",
        f"{team1} vs {team2}",
        f"Date: {original.get('date', '')}",
        f"Venue: {original.get('ground', '')}",
    ]

    if metadata.get("final_score"):
        lines.extend(["", f"Score: {metadata['final_score']}"])
    else:
        flags = []
        if metadata.get("went_to_extra_time"):
            flags.append("Went to extra time")
        if metadata.get("had_penalties"):
            flags.append("Decided by penalty shootout")
        if flags:
            lines.extend([""] + flags)

    if metadata.get("winner"):
        lines.append(f"Winner: {metadata['winner']}")
    elif metadata.get("is_draw"):
        lines.append("Result: Draw")

    if metadata.get("goal_scorers"):
        lines.append(f"Goal scorers: {', '.join(metadata['goal_scorers'])}")

    return "\n".join(lines)


def build_chunks(record: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Structure-aware parent–child chunking.

    1. Parent  = full original match text
    2. Children = one chunk per structural section
    """
    text = record.get("text", "")
    metadata = make_base_metadata(record)
    sections = parse_sections(text)
    chunks: list[dict[str, Any]] = []

    match_id = record["id"]
    team1 = metadata.get("team1") or "Team 1"
    team2 = metadata.get("team2") or "Team 2"
    match_label = f"{team1} vs {team2}"

    # ------------------------------------------------------------------
    # PARENT – full match document
    # ------------------------------------------------------------------
    parent = make_chunk(
        match_id=match_id,
        chunk_type="parent",
        chunk_index=0,
        text=text,
        metadata=metadata,
    )
    chunks.append(parent)
    parent_id = parent["id"]

    # ------------------------------------------------------------------
    # CHILDREN – structural sections
    # ------------------------------------------------------------------

    # Overview (summary)
    chunks.append(
        make_chunk(
            match_id=match_id,
            chunk_type="overview",
            chunk_index=0,
            text=build_overview_text(record, metadata),
            metadata=metadata,
            parent_id=parent_id,
        )
    )

    # Goals
    goal_lines = sections.get("Goals", [])
    if goal_lines:
        chunks.append(
            make_chunk(
                match_id=match_id,
                chunk_type="goals",
                chunk_index=0,
                text=f"{match_label} — Goals\n\n" + "\n".join(goal_lines),
                metadata=metadata,
                parent_id=parent_id,
            )
        )

    # Lineups (one child per team)
    lineups = extract_players_from_lineups(sections.get("Lineups", []))
    for idx, (team, players) in enumerate(lineups.items()):
        if not players:
            continue
        lineup_text = (
            f"{match_label} — {team} lineup\n\n"
            + "\n".join(f"- {p}" for p in players)
        )
        chunk_meta = {**metadata, "team": team, "players": players}
        chunks.append(
            make_chunk(
                match_id=match_id,
                chunk_type="lineup",
                chunk_index=idx,
                text=lineup_text,
                metadata=chunk_meta,
                parent_id=parent_id,
            )
        )

    # Substitutions
    sub_lines = sections.get("Substitutions", [])
    if sub_lines:
        chunks.append(
            make_chunk(
                match_id=match_id,
                chunk_type="substitutions",
                chunk_index=0,
                text=f"{match_label} — Substitutions\n\n" + "\n".join(sub_lines),
                metadata=metadata,
                parent_id=parent_id,
            )
        )

    # Bookings
    booking_lines = sections.get("Bookings", [])
    if booking_lines:
        chunks.append(
            make_chunk(
                match_id=match_id,
                chunk_type="bookings",
                chunk_index=0,
                text=f"{match_label} — Bookings\n\n" + "\n".join(booking_lines),
                metadata=metadata,
                parent_id=parent_id,
            )
        )

    # Penalty shootout
    penalty_lines = sections.get("Penalty shootout", [])
    if penalty_lines:
        chunks.append(
            make_chunk(
                match_id=match_id,
                chunk_type="penalties",
                chunk_index=0,
                text=f"{match_label} — Penalty shootout\n\n" + "\n".join(penalty_lines),
                metadata=metadata,
                parent_id=parent_id,
            )
        )

    # Referees
    referee_lines = sections.get("Referees", [])
    if referee_lines:
        chunks.append(
            make_chunk(
                match_id=match_id,
                chunk_type="referees",
                chunk_index=0,
                text=f"{match_label} — Referees\n\n" + "\n".join(referee_lines),
                metadata=metadata,
                parent_id=parent_id,
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Structure-aware parent-child chunking for World Cup matches."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    matches = load_jsonl(args.input)
    chunks: list[dict[str, Any]] = []

    for match in matches:
        chunks.extend(build_chunks(match))

    write_jsonl(args.output, chunks)

    # Simple stats
    parent_count = sum(1 for c in chunks if c["metadata"]["chunk_type"] == "parent")
    child_count = len(chunks) - parent_count

    print(f"Loaded   {len(matches):,} matches")
    print(f"Created  {len(chunks):,} total chunks")
    print(f"  Parent {parent_count:,}")
    print(f"  Child  {child_count:,}")
    print(f"Wrote    {args.output}")


if __name__ == "__main__":
    main()
