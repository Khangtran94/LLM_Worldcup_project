import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "matches.jsonl"


def validate_documents():
    if not PROCESSED_FILE.exists():
        print(f"File not found: {PROCESSED_FILE}")
        return

    total = 0

    missing_id = 0
    missing_text = 0
    missing_metadata = 0
    missing_year = 0
    missing_team1 = 0
    missing_team2 = 0

    invalid_json = 0

    ids = []
    years = Counter()

    text_lengths = []

    print("=" * 60)
    print("VALIDATING PROCESSED DATA")
    print("=" * 60)
    print(f"File: {PROCESSED_FILE}")

    with PROCESSED_FILE.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                document = json.loads(line)
            except json.JSONDecodeError as e:
                invalid_json += 1
                print(
                    f"\nInvalid JSON at line {line_number}: {e}"
                )
                continue

            total += 1

            # --------------------------------------------------
            # Required top-level fields
            # --------------------------------------------------

            document_id = document.get("id")
            text = document.get("text")
            metadata = document.get("metadata")

            if not document_id:
                missing_id += 1
            else:
                ids.append(document_id)

            if not text:
                missing_text += 1
            else:
                text_lengths.append(len(text))

            if not isinstance(metadata, dict):
                missing_metadata += 1
                continue

            # --------------------------------------------------
            # Required metadata
            # --------------------------------------------------

            year = metadata.get("year")
            team1 = metadata.get("team1")
            team2 = metadata.get("team2")

            if year is None:
                missing_year += 1
            else:
                years[year] += 1

            if not team1:
                missing_team1 += 1

            if not team2:
                missing_team2 += 1

    # ----------------------------------------------------------
    # Duplicate IDs
    # ----------------------------------------------------------

    id_counts = Counter(ids)

    duplicate_ids = {
        document_id: count
        for document_id, count in id_counts.items()
        if count > 1
    }

    # ----------------------------------------------------------
    # Text statistics
    # ----------------------------------------------------------

    if text_lengths:
        average_text_length = (
            sum(text_lengths) / len(text_lengths)
        )
        min_text_length = min(text_lengths)
        max_text_length = max(text_lengths)
    else:
        average_text_length = 0
        min_text_length = 0
        max_text_length = 0

    # ----------------------------------------------------------
    # Print results
    # ----------------------------------------------------------

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"Documents:             {total}")
    print(f"Invalid JSON lines:    {invalid_json}")
    print(f"Missing IDs:           {missing_id}")
    print(f"Missing text:          {missing_text}")
    print(f"Missing metadata:      {missing_metadata}")
    print(f"Missing year:          {missing_year}")
    print(f"Missing team1:         {missing_team1}")
    print(f"Missing team2:         {missing_team2}")
    print(f"Duplicate IDs:         {len(duplicate_ids)}")

    print("\n" + "=" * 60)
    print("TEXT LENGTH")
    print("=" * 60)

    print(
        f"Minimum:               {min_text_length:,} characters"
    )
    print(
        f"Average:               {average_text_length:,.0f} characters"
    )
    print(
        f"Maximum:               {max_text_length:,} characters"
    )

    print("\n" + "=" * 60)
    print("DOCUMENTS BY YEAR")
    print("=" * 60)

    for year in sorted(years):
        print(f"{year}: {years[year]}")

    # ----------------------------------------------------------
    # Duplicate IDs
    # ----------------------------------------------------------

    if duplicate_ids:
        print("\n" + "=" * 60)
        print("DUPLICATE IDs")
        print("=" * 60)

        for document_id, count in duplicate_ids.items():
            print(f"{document_id}: {count}")

    # ----------------------------------------------------------
    # Overall validation
    # ----------------------------------------------------------

    problems = (
        invalid_json
        + missing_id
        + missing_text
        + missing_metadata
        + missing_year
        + missing_team1
        + missing_team2
        + len(duplicate_ids)
    )

    print("\n" + "=" * 60)

    if problems == 0:
        print("VALIDATION PASSED")
        print("=" * 60)
        print("All documents passed the basic validation checks.")
    else:
        print("VALIDATION FAILED")
        print("=" * 60)
        print(f"Problems found: {problems}")


if __name__ == "__main__":
    validate_documents()