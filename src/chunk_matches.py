# src/chunk_matches.py

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SECTION_NAMES = {
    "Score",
    "Goals",
    "Lineups",
    "Substitutions",
    "Bookings",
    "Referees",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

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
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def parse_sections(text: str) -> dict[str, list[str]]:
    """
    Extract named sections from the match text.

    Example:

        Goals
        France: - Player (19')
        Mexico: - Player (70')

    becomes:

        {
            "Goals": [
                "France: - Player (19')",
                "Mexico: - Player (70')"
            ]
        }
    """

    lines = [line.strip() for line in text.splitlines()]

    sections: dict[str, list[str]] = {}
    current_section: str | None = None

    for line in lines:
        if not line:
            continue

        if line in SECTION_NAMES:
            current_section = line
            sections[current_section] = []
            continue

        if current_section is not None:
            sections[current_section].append(line)

    return sections


def clean_player_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s+\(captain\)", "", name, flags=re.IGNORECASE)
    return name


def extract_players(text: str) -> list[str]:
    """
    Extract player names from lineup lines.

    Expected forms:

        France: - Alex THEPOT
        Spain: PEDRI
        Argentina: - Lionel MESSI (captain)
    """

    players: list[str] = []

    for line in text.splitlines():
        line = line.strip()

        match = re.match(
            r"^(France|Mexico|Spain|Argentina|Brazil|Germany|Italy|"
            r"England|Uruguay|Netherlands|Portugal|Belgium|Croatia|"
            r"Japan|Korea Republic|USA|Canada|Morocco|Senegal|"
            r"Switzerland|Colombia|Chile|Ecuador|Peru|Paraguay|"
            r"Bolivia|Australia|Iran|Saudi Arabia|Tunisia|Poland|"
            r"Denmark|Serbia|Wales|Scotland|Austria|Czech Republic|"
            r"Turkey|Ghana|Nigeria|Cameroon|Costa Rica|Honduras|"
            r"Panama|Jamaica|Haiti|New Zealand|South Africa|"
            r"Algeria|Egypt|Iraq|Qatar|Russia|Ukraine|"
            r"Sweden|Norway|Finland|Greece|Romania|Bulgaria|"
            r"Paraguay|Venezuela|Bolivia)"
            r":\s*-?\s*(.+)$",
            line,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        player = clean_player_name(match.group(2))

        # Don't treat section metadata as players.
        if player and player not in players:
            players.append(player)

    return players


def extract_players_from_lineups(
    lineup_lines: list[str],
) -> dict[str, list[str]]:
    """
    Parse lineup lines into team -> players.

    We deliberately preserve the source data rather than trying
    to infer whether the source has a data-quality error.
    """

    lineups: dict[str, list[str]] = {}

    for line in lineup_lines:
        match = re.match(
            r"^(.+?):\s*-?\s*(.+)$",
            line,
        )

        if not match:
            continue

        team = match.group(1).strip()
        player = clean_player_name(match.group(2))

        # Avoid accidentally treating unrelated lines as lineups.
        if team.lower() in {
            "france",
            "mexico",
            "spain",
            "argentina",
            "brazil",
            "germany",
            "italy",
            "uruguay",
            "england",
            "netherlands",
            "belgium",
            "croatia",
            "portugal",
        }:
            lineups.setdefault(team, [])

            if player not in lineups[team]:
                lineups[team].append(player)

    return lineups


def extract_goal_scorers(
    goal_lines: list[str],
) -> list[str]:
    scorers: list[str] = []

    for line in goal_lines:
        match = re.match(
            r"^(.+?):\s*-?\s*(.+?)\s*\(([^)]+)\)\s*$",
            line,
        )

        if not match:
            continue

        scorer = clean_player_name(match.group(2))

        if scorer not in scorers:
            scorers.append(scorer)

    return scorers


def extract_goal_records(
    goal_lines: list[str],
) -> list[dict[str, str]]:
    goals = []

    for line in goal_lines:
        match = re.match(
            r"^(.+?):\s*-?\s*(.+?)\s*\(([^)]+)\)\s*$",
            line,
        )

        if not match:
            continue

        goals.append(
            {
                "team": match.group(1).strip(),
                "player": clean_player_name(match.group(2)),
                "minute": match.group(3).strip(),
            }
        )

    return goals


def extract_score(text: str) -> dict[str, Any]:
    """
    Supports score formats such as:

        Full time: [0, 0]; Extra time: [1, 0]

    Returns a small normalized structure.
    """

    result: dict[str, Any] = {
        "full_time": None,
        "extra_time": None,
        "penalties": None,
    }

    full_time = re.search(
        r"Full time:\s*\[([^\]]+)\]",
        text,
        flags=re.IGNORECASE,
    )

    if full_time:
        result["full_time"] = full_time.group(1).strip()

    extra_time = re.search(
        r"Extra time:\s*\[([^\]]+)\]",
        text,
        flags=re.IGNORECASE,
    )

    if extra_time:
        result["extra_time"] = extra_time.group(1).strip()

    penalties = re.search(
        r"Penalt(?:y|ies):\s*\[([^\]]+)\]",
        text,
        flags=re.IGNORECASE,
    )

    if penalties:
        result["penalties"] = penalties.group(1).strip()

    return result


def derive_result(
    metadata: dict[str, Any],
    goals: list[dict[str, str]],
    score: dict[str, Any],
) -> dict[str, Any]:

    team1 = metadata.get("team1")
    team2 = metadata.get("team2")

    # For historical records where there is no explicit score,
    # derive the result from goals.
    if goals:
        team1_goals = sum(
            1 for goal in goals if goal["team"] == team1
        )
        team2_goals = sum(
            1 for goal in goals if goal["team"] == team2
        )

        if team1_goals > team2_goals:
            winner = team1
        elif team2_goals > team1_goals:
            winner = team2
        else:
            winner = None
    else:
        winner = None

    is_draw = winner is None

    extra_time = score.get("extra_time")
    penalties = score.get("penalties")

    went_to_extra_time = extra_time is not None
    had_penalties = penalties is not None

    round_name = str(metadata.get("round", ""))

    return {
        "winner": winner,
        "is_draw": is_draw,
        "is_final": round_name.strip().lower() == "final",
        "went_to_extra_time": went_to_extra_time,
        "had_penalties": had_penalties,
    }


def make_base_metadata(
    record: dict[str, Any],
) -> dict[str, Any]:

    metadata = dict(record.get("metadata", {}))

    text = record.get("text", "")
    sections = parse_sections(text)

    lineup_lines = sections.get("Lineups", [])
    goal_lines = sections.get("Goals", [])

    lineups = extract_players_from_lineups(lineup_lines)
    goal_records = extract_goal_records(goal_lines)

    players = []

    for team_players in lineups.values():
        for player in team_players:
            if player not in players:
                players.append(player)

    goal_scorers = extract_goal_scorers(goal_lines)

    score = extract_score(text)

    result = derive_result(
        metadata=metadata,
        goals=goal_records,
        score=score,
    )

    teams = [
        metadata.get("team1"),
        metadata.get("team2"),
    ]

    teams = [
        team for team in teams
        if team
    ]

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


def build_overview_chunk(
    record: dict[str, Any],
    metadata: dict[str, Any],
) -> str:

    original_metadata = record.get("metadata", {})

    lines = [
        f"{original_metadata.get('tournament', '')} — "
        f"{original_metadata.get('round', '')}",
        "",
        f"{original_metadata.get('team1', '')} vs "
        f"{original_metadata.get('team2', '')}",
        f"Date: {original_metadata.get('date', '')}",
        f"Venue: {original_metadata.get('ground', '')}",
    ]

    score = extract_score(record.get("text", ""))

    if score["full_time"] or score["extra_time"]:
        lines.extend(
            [
                "",
                "Result:",
                f"Full time: {score['full_time']}",
                f"Extra time: {score['extra_time']}",
                f"Penalties: {score['penalties']}",
            ]
        )

    if metadata.get("winner"):
        lines.append(
            f"Winner: {metadata['winner']}"
        )
    elif metadata.get("is_draw"):
        lines.append("Result: Draw")

    return "\n".join(lines)


def build_chunks(
    record: dict[str, Any],
) -> list[dict[str, Any]]:

    text = record.get("text", "")
    metadata = make_base_metadata(record)

    sections = parse_sections(text)

    chunks = []

    # ---------------------------------------------------------
    # 1. Match overview
    # ---------------------------------------------------------

    chunks.append(
        make_chunk(
            match_id=record["id"],
            chunk_type="overview",
            chunk_index=0,
            text=build_overview_chunk(
                record,
                metadata,
            ),
            metadata=metadata,
        )
    )

    # ---------------------------------------------------------
    # 2. Goals
    # ---------------------------------------------------------

    goal_lines = sections.get("Goals", [])

    if goal_lines:
        goal_text = (
            f"{metadata['team1']} vs {metadata['team2']} — Goals\n\n"
            + "\n".join(goal_lines)
        )

        chunks.append(
            make_chunk(
                match_id=record["id"],
                chunk_type="goals",
                chunk_index=0,
                text=goal_text,
                metadata=metadata,
            )
        )

    # ---------------------------------------------------------
    # 3. Lineups
    # ---------------------------------------------------------

    lineups = extract_players_from_lineups(
        sections.get("Lineups", [])
    )

    lineup_index = 0

    for team, players in lineups.items():
        if not players:
            continue

        lineup_text = (
            f"{metadata['team1']} vs {metadata['team2']} — "
            f"{team} lineup\n\n"
            + "\n".join(
                f"- {player}"
                for player in players
            )
        )

        chunk_metadata = {
            **metadata,
            "team": team,
            "players": players,
        }

        chunks.append(
            make_chunk(
                match_id=record["id"],
                chunk_type="lineup",
                chunk_index=lineup_index,
                text=lineup_text,
                metadata=chunk_metadata,
            )
        )

        lineup_index += 1

    # ---------------------------------------------------------
    # 4. Substitutions
    # ---------------------------------------------------------

    substitution_lines = sections.get(
        "Substitutions",
        [],
    )

    if substitution_lines:
        substitution_text = (
            f"{metadata['team1']} vs {metadata['team2']} — "
            "Substitutions\n\n"
            + "\n".join(substitution_lines)
        )

        chunks.append(
            make_chunk(
                match_id=record["id"],
                chunk_type="substitutions",
                chunk_index=0,
                text=substitution_text,
                metadata=metadata,
            )
        )

    # ---------------------------------------------------------
    # 5. Bookings
    # ---------------------------------------------------------

    booking_lines = sections.get(
        "Bookings",
        [],
    )

    if booking_lines:
        booking_text = (
            f"{metadata['team1']} vs {metadata['team2']} — "
            "Bookings\n\n"
            + "\n".join(booking_lines)
        )

        chunks.append(
            make_chunk(
                match_id=record["id"],
                chunk_type="bookings",
                chunk_index=0,
                text=booking_text,
                metadata=metadata,
            )
        )

    # ---------------------------------------------------------
    # 6. Referees
    # ---------------------------------------------------------

    referee_lines = sections.get(
        "Referees",
        [],
    )

    if referee_lines:
        referee_text = (
            f"{metadata['team1']} vs {metadata['team2']} — "
            "Referees\n\n"
            + "\n".join(referee_lines)
        )

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create semantic RAG chunks from match JSONL."
    )

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "matches.jsonl",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "match_chunks.jsonl",
    )

    args = parser.parse_args()

    matches = load_jsonl(args.input)

    chunks = []

    for match in matches:
        chunks.extend(build_chunks(match))

    write_jsonl(args.output, chunks)

    print(f"Loaded {len(matches):,} matches")
    print(f"Created {len(chunks):,} RAG chunks")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":

    main()