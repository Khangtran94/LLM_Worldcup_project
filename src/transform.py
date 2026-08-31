import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "matches.jsonl"


def clean_text(value):
    """Convert a value to clean human-readable text."""
    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


def safe_id(value):
    """Create a stable ID-safe string."""
    value = clean_text(value)

    if not value:
        return "unknown"

    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def format_score(score):
    """Format the selected score fields."""
    if not isinstance(score, dict):
        return None

    parts = []

    if score.get("ft") is not None:
        parts.append(f"Full time: {score['ft']}")

    if score.get("et") is not None:
        parts.append(f"Extra time: {score['et']}")

    if score.get("p") is not None:
        parts.append(f"Penalty shootout: {score['p']}")

    return "; ".join(parts) if parts else None


def format_goals(goals, team_name):
    """Format goals for one team."""
    if not isinstance(goals, list) or not goals:
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


def format_bookings(bookings):
    """Format bookings."""
    if not isinstance(bookings, list):
        return []

    lines = []

    for booking in bookings:
        if not isinstance(booking, list):
            continue

        for event in booking:
            if not isinstance(event, dict):
                continue

            name = clean_text(event.get("name"))
            minute = clean_text(event.get("minute"))
            booking_type = clean_text(event.get("type"))

            if not name:
                continue

            description = f"- {name}"

            if booking_type:
                description += f" ({booking_type})"

            if minute:
                description += f" at {minute}'"

            lines.append(description)

    return lines


def format_lineup(lineup, team_name):
    """Format starting lineup and bench."""
    if not isinstance(lineup, list):
        return []

    lines = []

    for team_lineup in lineup:
        if not isinstance(team_lineup, dict):
            continue

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
                    lines.append(f"{team_name}: {name}")

    return lines


def format_substitutions(lineup, team_name):
    """Format substitutions."""
    if not isinstance(lineup, list):
        return []

    lines = []

    for team_lineup in lineup:
        if not isinstance(team_lineup, dict):
            continue

        substitutions = team_lineup.get("subs")

        if not isinstance(substitutions, list):
            continue

        for sub in substitutions:
            if not isinstance(sub, dict):
                continue

            on = clean_text(sub.get("on"))
            off = clean_text(sub.get("off"))
            minute = clean_text(sub.get("minute"))

            description = f"- {team_name}"

            if on:
                description += f": {on} on"

            if off:
                description += f", {off} off"

            if minute:
                description += f" ({minute}')"

            lines.append(description)

    return lines


def format_penalties(penalties):
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


def format_referees(referees):
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

        if country:
            lines.append(f"- {name} ({country})")
        else:
            lines.append(f"- {name}")

    return lines


def build_match_text(year, match):
    """Build human-readable text for a match."""
    team1 = clean_text(match.get("team1")) or "Unknown team"
    team2 = clean_text(match.get("team2")) or "Unknown team"

    round_name = clean_text(match.get("round"))
    date = clean_text(match.get("date"))
    time = clean_text(match.get("time"))
    ground = clean_text(match.get("ground"))

    sections = []

    # Match header
    header = f"{year} FIFA World Cup"

    if round_name:
        header += f" — {round_name}"

    sections.append(header)

    # Match information
    match_info = [f"{team1} vs {team2}"]

    if date:
        match_info.append(f"Date: {date}")

    if time:
        match_info.append(f"Time: {time}")

    if ground:
        match_info.append(f"Venue: {ground}")

    sections.append("\n".join(match_info))

    # Score
    score_text = format_score(match.get("score"))

    if score_text:
        sections.append(f"Score\n{score_text}")

    # Goals
    goal_lines = []

    goal_lines.extend(
        format_goals(match.get("goals1"), team1)
    )

    goal_lines.extend(
        format_goals(match.get("goals2"), team2)
    )

    if goal_lines:
        sections.append("Goals\n" + "\n".join(goal_lines))

    # Lineups
    lineup_lines = format_lineup(
        match.get("lineup"),
        team1,
    )

    lineup_lines.extend(
        format_lineup(
            match.get("lineup"),
            team2,
        )
    )

    if lineup_lines:
        sections.append(
            "Lineups\n" + "\n".join(lineup_lines)
        )

    # Substitutions
    substitution_lines = format_substitutions(
        match.get("lineup"),
        team1,
    )

    substitution_lines.extend(
        format_substitutions(
            match.get("lineup"),
            team2,
        )
    )

    if substitution_lines:
        sections.append(
            "Substitutions\n" + "\n".join(substitution_lines)
        )

    # Bookings
    booking_lines = format_bookings(
        match.get("bookings")
    )

    if booking_lines:
        sections.append(
            "Bookings\n" + "\n".join(booking_lines)
        )

    # Penalty shootout
    penalty_lines = format_penalties(
        match.get("penalties")
    )

    if penalty_lines:
        sections.append(
            "Penalty shootout\n" + "\n".join(penalty_lines)
        )

    # Referees
    referee_lines = format_referees(
        match.get("referees")
    )

    if referee_lines:
        sections.append(
            "Referees\n" + "\n".join(referee_lines)
        )

    return "\n\n".join(sections)


def transform_match(year, match, source_file):
    """Transform one raw match into a RAG document."""
    team1 = clean_text(match.get("team1")) or "unknown"
    team2 = clean_text(match.get("team2")) or "unknown"
    round_name = clean_text(match.get("round")) or "unknown"

    match_id = (
        f"{year}_"
        f"{safe_id(round_name)}_"
        f"{safe_id(team1)}_"
        f"{safe_id(team2)}"
    )

    metadata = {
        "year": year,
        "tournament": "FIFA World Cup",
        "round": round_name,
        "date": clean_text(match.get("date")),
        "team1": team1,
        "team2": team2,
        "ground": clean_text(match.get("ground")),
        "source_file": str(
            source_file.relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
    }

    # Remove metadata fields whose values are missing.
    metadata = {
        key: value
        for key, value in metadata.items()
        if value is not None
    }

    return {
        "id": match_id,
        "text": build_match_text(year, match),
        "metadata": metadata,
    }


def process_file(file_path, output_file):
    """Process one worldcup-full.json file."""
    year = file_path.parent.name

    try:
        year = int(year)
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

    for match in matches:
        if not isinstance(match, dict):
            continue

        document = transform_match(
            year,
            match,
            file_path,
        )

        output_file.write(
            json.dumps(
                document,
                ensure_ascii=False,
            )
            + "\n"
        )

        count += 1

    return count


def main():
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        RAW_DIR.rglob("worldcup-full.json")
    )

    if not files:
        print("No worldcup-full.json files found.")
        return

    total_matches = 0

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for file_path in files:
            count = process_file(
                file_path,
                output_file,
            )

            total_matches += count

            print(
                f"Processed: "
                f"{file_path.parent.name}/"
                f"{file_path.name} "
                f"({count} matches)"
            )

    print("\n" + "=" * 60)
    print("TRANSFORMATION COMPLETE")
    print("=" * 60)
    print(f"Files processed: {len(files)}")
    print(f"Matches processed: {total_matches}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()