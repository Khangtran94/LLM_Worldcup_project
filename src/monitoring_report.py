# src/monitoring_report.py
"""
Quick summary over the `queries` / `feedback` tables — real usage,
not eval runs. Run any time to see what's actually been asked and
how it's landing.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db.connection import get_connection


def main() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM queries")
            total_queries = cur.fetchone()[0]

            if total_queries == 0:
                print("No queries logged yet — run src/main.py and ask something.")
                return

            cur.execute(
                "SELECT model, COUNT(*), AVG(latency_ms) FROM queries GROUP BY model ORDER BY COUNT(*) DESC"
            )
            by_model = cur.fetchall()

            cur.execute(
                """
                SELECT COUNT(*) FILTER (WHERE is_positive), COUNT(*) FILTER (WHERE NOT is_positive)
                FROM feedback
                """
            )
            positive, negative = cur.fetchone()

            cur.execute(
                """
                SELECT q.question, f.is_positive, f.comment
                FROM feedback f
                JOIN queries q ON q.id = f.query_id
                WHERE NOT f.is_positive
                ORDER BY f.created_at DESC
                LIMIT 10
                """
            )
            recent_negative = cur.fetchall()

            cur.execute(
                "SELECT question, model, latency_ms, created_at FROM queries ORDER BY created_at DESC LIMIT 10"
            )
            recent = cur.fetchall()

    print("=" * 60)
    print("PRODUCTION QUERY LOG SUMMARY")
    print("=" * 60)
    print(f"Total queries logged: {total_queries:,}\n")

    print("By model / route:")
    for model, count, avg_latency in by_model:
        print(f"  {model:20} {count:5,}   avg latency {avg_latency:.0f}ms")

    feedback_total = positive + negative
    print(f"\nFeedback: {feedback_total:,} responses ({positive:,} positive / {negative:,} negative)")
    if feedback_total:
        print(f"Positive rate: {positive / feedback_total:.1%}")

    if recent_negative:
        print("\nMost recent negative feedback:")
        for question, _, comment in recent_negative:
            preview = question[:70]
            note = f" — \"{comment}\"" if comment else ""
            print(f"  ✗ {preview}{note}")

    print("\nMost recent queries:")
    for question, model, latency_ms, created_at in recent:
        preview = question[:70]
        print(f"  [{model:8}] {latency_ms:5}ms  {preview}")


if __name__ == "__main__":
    main()
