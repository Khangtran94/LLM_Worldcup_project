from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "matches.jsonl"


def clean_text(value: Any) -> str | None:
    """Convert a value to clean human-readable text."""
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def safe_id(value: Any) -> str:
    """Create a stable ID-safe string."""
    value = clean_text(value)
    if not value:
        return "unknown"
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "unknown"


def format_score(score: Any, team1: str, team2: str) -> str | None:
    """Format the source score, which is normally [team1, team2]."""
    if isinstance(score, list) and len(score) >= 2:
        parts = [f"Final score: {team1} {score[0]}–{score[1]} {team2}"]
        return "\n".join(parts)

    if isinstance(score, dict):
        parts = []
        if score.get("ft") is not None:
            parts.append(f"Full time: {score['ft']}")
        if score.get("et") is not None:
            parts.append(f"Extra time: {score['et']}")
        if score.get("p") is not None:
            parts.append(f"Penalty shootout: {score['p']}")
        return "; ".join(parts) if parts else None

    return None


def format_goals(goals: Any, team_name: str) -> list[str]:
    """Format goals for one team."""
    if not isinstance(goals, list):
        return []

    lines = []
    for goal in goals:
        if not isinstance(goal, dict):
            continue

        name = clean_text(goal.get("name"))
        minute = clean_text(goal.get("minute"))
        if not name:
            continue

        description = f"- {name}"
        if minute:
            description += f" ({minute}')"
        if goal.get("penalty") is True:
            description += " [penalty]"
        if goal.get("owngoal") is True:
            description += " [own goal]"

        lines.append(f"{team_name}: {description}")

    return lines


def get_team_entries(value: Any, team1: str, team2: str) -> list[tuple[str, Any]]:
    """Map the source's two positional team entries to team1/team2.

    openfootball/worldcup.json stores lineup-related data positionally:
    entry 0 belongs to team1 and entry 1 belongs to team2. The source
    does not repeat the team name inside each entry.
    """
    if not isinstance(value, list):
        return []

    result = []
    teams = (team1, team2)
    for index, entry in enumerate(value[:2]):
        if index < len(teams) and isinstance(entry, dict):
            result.append((teams[index], entry))
    return result


def format_lineup(lineup: Any, team1: str, team2: str) -> list[str]:
    """Format each team's lineup using positional source semantics."""
    lines = []

    for team_name, team_lineup in get_team_entries(lineup, team1, team2):
        starters = team_lineup.get("starter")
        bench = team_lineup.get("bench")

        if isinstance(starters, list):
            for player in starters:
                if not isinstance(player, dict):
                    continue
                name = clean_text(player.get("name"))
                if not name:
                    continue
                description = f"- {name}"
                if player.get("captain") is True:
                    description += " (captain)"
                lines.append(f"{team_name}: {description}")

        if isinstance(bench, list):
            for player in bench:
                if not isinstance(player, dict):
                    continue
                name = clean_text(player.get("name"))
                if name:
                    lines.append(f"{team_name} bench: {name}")

    return lines


def format_substitutions(lineup: Any, team1: str, team2: str) -> list[str]:
    """Format substitutions using the same positional team mapping."""
    lines = []

    for team_name, team_lineup in get_team_entries(lineup, team1, team2):
        substitutions = team_lineup.get("subs")
        if not isinstance(substitutions, list):
            continue

        for sub in substitutions:
            if not isinstance(sub, dict):
                continue

            on = clean_text(sub.get("on"))
            off = clean_text(sub.get("off"))
            minute = clean_text(sub.get("minute"))

            description = f"- {team_name}:"
            if on:
                description += f" {on} on"
            if off:
                description += f", {off} off"
            if minute:
                description += f" ({minute}')"
            lines.append(description)

    return lines


def format_bookings(bookings: Any, team1: str, team2: str) -> list[str]:
    """Format bookings while preserving the source's team association."""
    if not isinstance(bookings, list):
        return []

    lines = []
    teams = (team1, team2)

    # The source uses one event list per team.
    for index, team_bookings in enumerate(bookings[:2]):
        team_name = teams[index] if index < 2 else "Unknown team"
        if not isinstance(team_bookings, list):
            continue

        for event in team_bookings:
            if not isinstance(event, dict):
                continue

            name = clean_text(event.get("name"))
            minute = clean_text(event.get("minute"))
            booking_type = clean_text(event.get("type"))
            if not name:
                continue

            description = f"- {team_name}: {name}"
            if booking_type:
                description += f" ({booking_type})"
            if minute:
                description += f" at {minute}'"
            lines.append(description)

    return lines


def format_penalties(penalties: Any) -> list[str]:
    """Format penalty shootout details."""
    if not isinstance(penalties, list):
        return []

    lines = []
    for penalty in penalties:
        if not isinstance(penalty, dict):
            continue

        name = clean_text(penalty.get("name"))
        score = clean_text(penalty.get("score"))
        note = clean_text(penalty.get("note"))
        if not name:
            continue

        description = f"- {name}"
        if score:
            description += f" ({score})"
        if note:
            description += f": {note}"
        lines.append(description)

    return lines


def format_referees(referees: Any) -> list[str]:
    """Format referees."""
    if not isinstance(referees, list):
        return []

    lines = []
    for referee in referees:
        if not isinstance(referee, dict):
            continue

        name = clean_text(referee.get("name"))
        country = clean_text(referee.get("country"))
        if not name:
            continue

        lines.append(f"- {name} ({country})" if country else f"- {name}")

    return lines


def build_match_text(year: int, match: dict[str, Any]) -> str:
    """Build human-readable text for a match."""
    team1 = clean_text(match.get("team1")) or "Unknown team"
    team2 = clean_text(match.get("team2")) or "Unknown team"
    round_name = clean_text(match.get("round"))
    date = clean_text(match.get("date"))
    time = clean_text(match.get("time"))
    ground = clean_text(match.get("ground"))

    sections = []

    header = f"{year} FIFA World Cup"
    if round_name:
        header += f" — {round_name}"
    sections.append(header)

    match_info = [f"{team1} vs {team2}"]
    if date:
        match_info.append(f"Date: {date}")
    if time:
        match_info.append(f"Time: {time}")
    if ground:
        match_info.append(f"Venue: {ground}")
    sections.append("\n".join(match_info))

    score_text = format_score(match.get("score"), team1, team2)
    if score_text:
        sections.append("Score\n" + score_text)

    goal_lines = []
    goal_lines.extend(format_goals(match.get("goals1"), team1))
    goal_lines.extend(format_goals(match.get("goals2"), team2))
    if goal_lines:
        sections.append("Goals\n" + "\n".join(goal_lines))

    lineup_lines = format_lineup(match.get("lineup"), team1, team2)
    if lineup_lines:
        sections.append("Lineups\n" + "\n".join(lineup_lines))

    substitution_lines = format_substitutions(match.get("lineup"), team1, team2)
    if substitution_lines:
        sections.append("Substitutions\n" + "\n".join(substitution_lines))

    booking_lines = format_bookings(match.get("bookings"), team1, team2)
    if booking_lines:
        sections.append("Bookings\n" + "\n".join(booking_lines))

    penalty_lines = format_penalties(match.get("penalties"))
    if penalty_lines:
        sections.append("Penalty shootout\n" + "\n".join(penalty_lines))

    referee_lines = format_referees(match.get("referees"))
    if referee_lines:
        sections.append("Referees\n" + "\n".join(referee_lines))

    return "\n\n".join(sections)


def transform_match(year: int, match: dict[str, Any], source_file: Path, match_index: int) -> dict[str, Any]:
    """Transform one raw match into a RAG document."""
    team1 = clean_text(match.get("team1")) or "unknown"
    team2 = clean_text(match.get("team2")) or "unknown"
    round_name = clean_text(match.get("round")) or "unknown"

    match_id = f"{year}_match_{match_index}_{safe_id(team1)}_{safe_id(team2)}"

    metadata = {
        "year": year,
        "tournament": "FIFA World Cup",
        "round": round_name,
        "date": clean_text(match.get("date")),
        "team1": team1,
        "team2": team2,
        "ground": clean_text(match.get("ground")),
        "source_file": str(source_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}

    return {
        "id": match_id,
        "text": build_match_text(year, match),
        "metadata": metadata,
    }


def process_file(file_path: Path, output_file) -> int:
    """Process one worldcup-full.json file."""
    try:
        year = int(file_path.parent.name)
    except ValueError:
        print(f"Skipping invalid year folder: {file_path}")
        return 0

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    matches = data.get("matches", [])
    if not isinstance(matches, list):
        print(f"Invalid matches structure: {file_path}")
        return 0

    count = 0
    for match_index, match in enumerate(matches, start=1):
        if not isinstance(match, dict):
            continue

        document = transform_match(year, match, file_path, match_index)
        output_file.write(json.dumps(document, ensure_ascii=False) + "\n")
        count += 1

    return count


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW_DIR.rglob("worldcup-full.json"))

    if not files:
        print("No worldcup-full.json files found.")
        return

    total_matches = 0
    with OUTPUT_FILE.open("w", encoding="utf-8") as output_file:
        for file_path in files:
            count = process_file(file_path, output_file)
            total_matches += count
            print(f"Processed: {file_path.parent.name}/{file_path.name} ({count} matches)")

    print("\n" + "=" * 60)
    print("TRANSFORMATION COMPLETE")
    print("=" * 60)
    print(f"Files processed: {len(files)}")
    print(f"Matches processed: {total_matches}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
