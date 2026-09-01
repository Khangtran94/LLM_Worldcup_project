# src/chunk_matches.py
"""
Semantic chunking for FIFA World Cup match documents.

Reads data/processed/matches.jsonl and produces data/processed/match_chunks.jsonl.

Chunk types:
  - overview      : match summary + result + winner
  - goals         : all goals in the match
  - lineup        : one chunk per team
  - substitutions : substitutions
  - bookings      : yellow / red cards
  - referees      : referee list
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
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------

def parse_sections(text: str) -> dict[str, list[str]]:
    """Extract named sections from the match text."""
    lines = [line.strip() for line in text.splitlines()]
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
    Parse lineup lines into {team: [players]}.

    Expected formats after the transform fix:
        France: - Alex THEPOT
        France: - Alex VILLAPLANE (captain)
        Mexico bench: Oscar BONFIGLIO
    """
    lineups: dict[str, list[str]] = {}

    for line in lineup_lines:
        # Starters: "Team: - Player" or "Team: - Player (captain)"
        match = re.match(r"^(.+?):\s*-\s*(.+)$", line)
        if match:
            team = match.group(1).strip()
            player = clean_player_name(match.group(2))
            if team and player:
                lineups.setdefault(team, [])
                if player not in lineups[team]:
                    lineups[team].append(player)
            continue

        # Bench: "Team bench: Player"
        match = re.match(r"^(.+?)\s+bench:\s*(.+)$", line, flags=re.IGNORECASE)
        if match:
            team = match.group(1).strip()
            player = clean_player_name(match.group(2))
            if team and player:
                lineups.setdefault(team, [])
                if player not in lineups[team]:
                    lineups[team].append(player)

    return lineups


def extract_goal_records(goal_lines: list[str]) -> list[dict[str, str]]:
    """Parse goal lines into structured records."""
    goals: list[dict[str, str]] = []

    for line in goal_lines:
        # "France: - Lucien LAURENT (19')" or with [penalty] / [own goal]
        match = re.match(
            r"^(.+?):\s*-\s*(.+?)\s*\(([^)]+)\)\s*(.*)$",
            line,
        )
        if not match:
            continue

        goals.append(
            {
                "team": match.group(1).strip(),
                "player": clean_player_name(match.group(2)),
                "minute": match.group(3).strip(),
                "note": match.group(4).strip(),
            }
        )

    return goals


def extract_goal_scorers(goal_lines: list[str]) -> list[str]:
    scorers: list[str] = []
    for record in extract_goal_records(goal_lines):
        player = record["player"]
        if player and player not in scorers:
            scorers.append(player)
    return scorers


def extract_score(text: str, sections: dict[str, list[str]]) -> dict[str, Any]:
    """
    Support both score formats produced by transform.py:

    1. New format:  Final score: France 4–1 Mexico
    2. Legacy:      Full time: [0, 0]; Extra time: [1, 0]
    """
    result: dict[str, Any] = {
        "final_score": None,
        "full_time": None,
        "extra_time": None,
        "penalties": None,
    }

    score_lines = sections.get("Score", [])
    score_text = " ".join(score_lines) if score_lines else text

    # New clean format
    final = re.search(
        r"Final score:\s*(.+)",
        score_text,
        flags=re.IGNORECASE,
    )
    if final:
        result["final_score"] = final.group(1).strip()

    # Legacy formats
    ft = re.search(r"Full time:\s*\[?([^\];]+)\]?", score_text, flags=re.IGNORECASE)
    if ft:
        result["full_time"] = ft.group(1).strip()

    et = re.search(r"Extra time:\s*\[?([^\];]+)\]?", score_text, flags=re.IGNORECASE)
    if et:
        result["extra_time"] = et.group(1).strip()

    pen = re.search(
        r"Penalt(?:y|ies)(?:\s+shootout)?:\s*\[?([^\];]+)\]?",
        score_text,
        flags=re.IGNORECASE,
    )
    if pen:
        result["penalties"] = pen.group(1).strip()

    return result


def derive_result(
    metadata: dict[str, Any],
    goals: list[dict[str, str]],
    score: dict[str, Any],
) -> dict[str, Any]:
    team1 = metadata.get("team1")
    team2 = metadata.get("team2")

    winner = None
    if goals:
        team1_goals = sum(1 for g in goals if g["team"] == team1)
        team2_goals = sum(1 for g in goals if g["team"] == team2)
        if team1_goals > team2_goals:
            winner = team1
        elif team2_goals > team1_goals:
            winner = team2

    round_name = str(metadata.get("round", "")).strip().lower()

    return {
        "winner": winner,
        "is_draw": winner is None and bool(goals),
        "is_final": round_name == "final",
        "went_to_extra_time": score.get("extra_time") is not None,
        "had_penalties": score.get("penalties") is not None,
    }


# ---------------------------------------------------------------------------
# Metadata & chunk builders
# ---------------------------------------------------------------------------

def make_base_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(record.get("metadata", {}))
    text = record.get("text", "")
    sections = parse_sections(text)

    lineup_lines = sections.get("Lineups", [])
    goal_lines = sections.get("Goals", [])

    lineups = extract_players_from_lineups(lineup_lines)
    goal_records = extract_goal_records(goal_lines)
    goal_scorers = extract_goal_scorers(goal_lines)
    score = extract_score(text, sections)
    result = derive_result(metadata, goal_records, score)

    players: list[str] = []
    for team_players in lineups.values():
        for player in team_players:
            if player not in players:
                players.append(player)

    teams = [t for t in (metadata.get("team1"), metadata.get("team2")) if t]

    return {
        "match_id": record["id"],
        "year": metadata.get("year"),
        "tournament": metadata.get("tournament"),
        "round": metadata.get("round"),
        "date": metadata.get("date"),
        "team1": metadata.get("team1"),
        "team2": metadata.get("team2"),
        "teams": teams,
        "ground": metadata.get("ground"),
        "winner": result["winner"],
        "is_draw": result["is_draw"],
        "is_final": result["is_final"],
        "went_to_extra_time": result["went_to_extra_time"],
        "had_penalties": result["had_penalties"],
        "players": players,
        "goal_scorers": goal_scorers,
        "final_score": score.get("final_score"),
    }


def make_chunk(
    *,
    match_id: str,
    chunk_type: str,
    chunk_index: int,
    text: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": f"{match_id}_{chunk_type}_{chunk_index}",
        "text": text.strip(),
        "metadata": {
            **metadata,
            "chunk_type": chunk_type,
            "chunk_index": chunk_index,
        },
    }


def build_overview_chunk(record: dict[str, Any], metadata: dict[str, Any]) -> str:
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
    elif metadata.get("went_to_extra_time") or metadata.get("had_penalties"):
        lines.append("")
        if metadata.get("went_to_extra_time"):
            lines.append("Went to extra time")
        if metadata.get("had_penalties"):
            lines.append("Decided by penalty shootout")

    if metadata.get("winner"):
        lines.append(f"Winner: {metadata['winner']}")
    elif metadata.get("is_draw"):
        lines.append("Result: Draw")

    if metadata.get("goal_scorers"):
        lines.append(f"Goal scorers: {', '.join(metadata['goal_scorers'])}")

    return "\n".join(lines)


def build_chunks(record: dict[str, Any]) -> list[dict[str, Any]]:
    text = record.get("text", "")
    metadata = make_base_metadata(record)
    sections = parse_sections(text)
    chunks: list[dict[str, Any]] = []

    team1 = metadata.get("team1") or "Team 1"
    team2 = metadata.get("team2") or "Team 2"
    match_label = f"{team1} vs {team2}"

    # 1. Overview
    chunks.append(
        make_chunk(
            match_id=record["id"],
            chunk_type="overview",
            chunk_index=0,
            text=build_overview_chunk(record, metadata),
            metadata=metadata,
        )
    )

    # 2. Goals
    goal_lines = sections.get("Goals", [])
    if goal_lines:
        goal_text = f"{match_label} — Goals\n\n" + "\n".join(goal_lines)
        chunks.append(
            make_chunk(
                match_id=record["id"],
                chunk_type="goals",
                chunk_index=0,
                text=goal_text,
                metadata=metadata,
            )
        )

    # 3. Lineups (one chunk per team)
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
                match_id=record["id"],
                chunk_type="lineup",
                chunk_index=idx,
                text=lineup_text,
                metadata=chunk_meta,
            )
        )

    # 4. Substitutions
    sub_lines = sections.get("Substitutions", [])
    if sub_lines:
        sub_text = f"{match_label} — Substitutions\n\n" + "\n".join(sub_lines)
        chunks.append(
            make_chunk(
                match_id=record["id"],
                chunk_type="substitutions",
                chunk_index=0,
                text=sub_text,
                metadata=metadata,
            )
        )

    # 5. Bookings
    booking_lines = sections.get("Bookings", [])
    if booking_lines:
        booking_text = f"{match_label} — Bookings\n\n" + "\n".join(booking_lines)
        chunks.append(
            make_chunk(
                match_id=record["id"],
                chunk_type="bookings",
                chunk_index=0,
                text=booking_text,
                metadata=metadata,
            )
        )

    # 6. Penalty shootout
    penalty_lines = sections.get("Penalty shootout", [])
    if penalty_lines:
        penalty_text = f"{match_label} — Penalty shootout\n\n" + "\n".join(penalty_lines)
        chunks.append(
            make_chunk(
                match_id=record["id"],
                chunk_type="penalties",
                chunk_index=0,
                text=penalty_text,
                metadata=metadata,
            )
        )

    # 7. Referees
    referee_lines = sections.get("Referees", [])
    if referee_lines:
        referee_text = f"{match_label} — Referees\n\n" + "\n".join(referee_lines)
        chunks.append(
            make_chunk(
                match_id=record["id"],
                chunk_type="referees",
                chunk_index=0,
                text=referee_text,
                metadata=metadata,
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create semantic RAG chunks from World Cup match documents."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    matches = load_jsonl(args.input)
    chunks: list[dict[str, Any]] = []

    for match in matches:
        chunks.extend(build_chunks(match))

    write_jsonl(args.output, chunks)

    print(f"Loaded  {len(matches):,} matches")
    print(f"Created {len(chunks):,} RAG chunks")
    print(f"Wrote   {args.output}")


if __name__ == "__main__":
    main()
