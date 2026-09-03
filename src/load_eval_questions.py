# src/load_eval_questions.py
"""
Load the 30 eval questions (with your ground truth) from a CSV file into
public.eval_questions.

CSV columns (header required):
    question               - the question text. Bake in year + team names +
                              round so the answer is unambiguous, e.g.
                              "Who won the 2014 World Cup final between
                              Germany and Argentina?" rather than just
                              "Who won the World Cup final?"
    category                - one of:
                              exact_fact | year_final | aggregate | lineup |
                              multihop | negative
    expected_answer         - the correct answer, from your own research
    expected_match_ids      - one or more match_ids (from matches.jsonl),
                              separated by ';' if more than one
    expected_chunk_types    - optional. chunk_type(s) separated by ';',
                              e.g. "goals" or "lineup;parent". Leave blank
                              if not relevant to the question.

Rows are matched by exact `question` text (UNIQUE) and updated in place,
so re-running this after editing the CSV never orphans eval_results
already collected for that question.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db.connection import get_connection

PROJECT_ROOT = SRC_DIR.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "eval" / "questions.csv"

VALID_CATEGORIES = {
    "exact_fact",
    "year_final",
    "aggregate",
    "lineup",
    "multihop",
    "negative",
}

UPSERT_SQL = """
INSERT INTO eval_questions (
    question, category, expected_answer, expected_match_ids, expected_chunk_types
) VALUES (
    %(question)s, %(category)s, %(expected_answer)s,
    %(expected_match_ids)s, %(expected_chunk_types)s
)
ON CONFLICT (question) DO UPDATE SET
    category = EXCLUDED.category,
    expected_answer = EXCLUDED.expected_answer,
    expected_match_ids = EXCLUDED.expected_match_ids,
    expected_chunk_types = EXCLUDED.expected_chunk_types
"""


def split_list(value: str | None) -> list[str] | None:
    value = (value or "").strip()
    if not value:
        return None
    return [v.strip() for v in value.split(";") if v.strip()]


def load_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"question", "category", "expected_answer", "expected_match_ids"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for line_no, row in enumerate(reader, start=2):
            question = (row.get("question") or "").strip()
            if not question:
                continue

            category = (row.get("category") or "").strip()
            if category not in VALID_CATEGORIES:
                raise ValueError(
                    f"Line {line_no}: invalid category '{category}'. "
                    f"Must be one of {sorted(VALID_CATEGORIES)}"
                )

            match_ids = split_list(row.get("expected_match_ids"))
            if not match_ids:
                raise ValueError(f"Line {line_no}: expected_match_ids is required")

            rows.append(
                {
                    "question": question,
                    "category": category,
                    "expected_answer": (row.get("expected_answer") or "").strip() or None,
                    "expected_match_ids": match_ids,
                    "expected_chunk_types": split_list(row.get("expected_chunk_types")),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Load eval questions into Postgres.")
    parser.add_argument("--input", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"CSV not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    rows = load_csv(args.input)
    if not rows:
        print("No rows found in CSV — nothing to load.")
        return

    print(f"Parsed {len(rows)} question(s) from {args.input}")

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1

    with get_connection() as conn:
        with conn.cursor() as cur:
            for row in rows:
                cur.execute(UPSERT_SQL, row)
        conn.commit()

    print("Loaded into eval_questions ✓")
    print("\nBy category:")
    for cat, n in sorted(counts.items()):
        print(f"  {cat:12} {n}")


if __name__ == "__main__":
    main()
