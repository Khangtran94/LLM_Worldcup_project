# src/run_eval.py
"""
Run the eval question set through the current retrieval (and optionally
the full RAG/LLM) pipeline, score retrieval quality, and save results.

Metrics (computed per question, then averaged for the run):
    hit_at_12              - was any expected match_id retrieved in top 12?
    mrr                     - 1 / rank of the first retrieved chunk whose
                              match_id is an expected match_id (0 if none
                              land in the top 12)
    parent_hit_at_12        - was the PARENT chunk of an expected match_id
                              retrieved in top 12?
    child_type_hit_at_12    - only scored for questions that set
                              expected_chunk_types: was a chunk with
                              match_id in expected_match_ids AND chunk_type
                              in expected_chunk_types retrieved in top 12?
                              (NULL / skipped for questions without one)

`manual_score` (correct / partial / wrong) is left NULL — that's the
step where you read llm_answer against expected_answer by hand, e.g.:

    UPDATE eval_results SET manual_score = 'correct'
    WHERE id = 17;

Usage:
    python src/run_eval.py --run-label baseline_2026_09_03
    python src/run_eval.py --run-label baseline --skip-llm     # retrieval only, no OpenAI calls/cost
    python src/run_eval.py --run-label baseline --category aggregate
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db.connection import get_connection
from retrieve import retrieve
from query_router import try_direct_answer


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_questions(category: str | None) -> list[dict[str, Any]]:
    where = ""
    params: list[Any] = []
    if category:
        where = "WHERE category = %s"
        params.append(category)

    sql = f"""
        SELECT id, question, category, expected_answer,
               expected_match_ids, expected_chunk_types
        FROM eval_questions
        {where}
        ORDER BY id
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    return [
        {
            "id": r[0],
            "question": r[1],
            "category": r[2],
            "expected_answer": r[3],
            "expected_match_ids": set(r[4] or []),
            "expected_chunk_types": set(r[5] or []),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    retrieved: list[dict[str, Any]],
    expected_match_ids: set[str],
    expected_chunk_types: set[str],
    top_k: int,
) -> dict[str, Any]:
    top = retrieved[:top_k]

    hit_at_12 = any(r["match_id"] in expected_match_ids for r in top)

    mrr = 0.0
    for rank, r in enumerate(top, start=1):
        if r["match_id"] in expected_match_ids:
            mrr = 1.0 / rank
            break

    parent_hit_at_12 = any(
        r["match_id"] in expected_match_ids and r["chunk_type"] == "parent"
        for r in top
    )

    child_hit_at_12 = None
    if expected_chunk_types:
        child_hit_at_12 = any(
            r["match_id"] in expected_match_ids
            and r["chunk_type"] in expected_chunk_types
            for r in top
        )

    return {
        "hit_at_12": hit_at_12,
        "mrr": mrr,
        "parent_hit_at_12": parent_hit_at_12,
        "child_type_hit_at_12": child_hit_at_12,
    }


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------

INSERT_RESULT_SQL = """
INSERT INTO eval_results (
    question_id, run_label, retrieved_ids,
    hit_at_12, mrr, parent_hit_at_12, child_type_hit_at_12,
    llm_answer, latency_ms
) VALUES (
    %(question_id)s, %(run_label)s, %(retrieved_ids)s,
    %(hit_at_12)s, %(mrr)s, %(parent_hit_at_12)s, %(child_type_hit_at_12)s,
    %(llm_answer)s, %(latency_ms)s
)
"""


def save_result(row: dict[str, Any]) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(INSERT_RESULT_SQL, row)
        conn.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG eval question set.")
    parser.add_argument(
        "--run-label",
        default=datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S"),
        help="Label to group this eval run (default: timestamp).",
    )
    parser.add_argument("--category", default=None, help="Only run one category.")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Only compute retrieval metrics; don't call the LLM (faster, no API cost).",
    )
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    questions = load_questions(args.category)
    if not questions:
        print("No eval questions found. Run src/load_eval_questions.py first.")
        return

    print(f"Run label: {args.run_label}")
    print(f"Questions: {len(questions)}")
    print(f"LLM calls: {'skipped' if args.skip_llm else 'enabled'}\n")

    per_question_rows: list[dict[str, Any]] = []

    for q in questions:
        start = time.perf_counter()

        retrieved = retrieve(q["question"], top_k=args.top_k)
        metrics = compute_metrics(
            retrieved, q["expected_match_ids"], q["expected_chunk_types"], args.top_k
        )

        llm_answer = None
        if not args.skip_llm:
            direct = try_direct_answer(q["question"])
            if direct is not None:
                llm_answer = direct
            else:
                from prompt import build_messages
                from rag import client, MODEL

                messages = build_messages(q["question"], retrieved)
                response = client.chat.completions.create(
                    model=MODEL, messages=messages, temperature=0.1
                )
                llm_answer = response.choices[0].message.content.strip()

        latency_ms = int((time.perf_counter() - start) * 1000)

        row = {
            "question_id": q["id"],
            "run_label": args.run_label,
            "retrieved_ids": [r["match_id"] for r in retrieved[: args.top_k]],
            "hit_at_12": metrics["hit_at_12"],
            "mrr": metrics["mrr"],
            "parent_hit_at_12": metrics["parent_hit_at_12"],
            "child_type_hit_at_12": metrics["child_type_hit_at_12"],
            "llm_answer": llm_answer,
            "latency_ms": latency_ms,
        }
        save_result(row)
        per_question_rows.append(
            {**row, "category": q["category"], "question": q["question"]}
        )

        status = "✓" if metrics["hit_at_12"] else "✗"
        print(f"  [{status}] ({q['category']:12}) {q['question'][:70]}")

    print_summary(per_question_rows)


def print_summary(rows: list[dict[str, Any]]) -> None:
    n = len(rows)
    hit_rate = sum(r["hit_at_12"] for r in rows) / n
    mrr = sum(r["mrr"] for r in rows) / n
    parent_hit_rate = sum(r["parent_hit_at_12"] for r in rows) / n

    child_scored = [r for r in rows if r["child_type_hit_at_12"] is not None]
    child_hit_rate = (
        sum(r["child_type_hit_at_12"] for r in child_scored) / len(child_scored)
        if child_scored
        else None
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Questions evaluated:        {n}")
    print(f"Hit Rate @12:                {hit_rate:.1%}")
    print(f"MRR:                         {mrr:.3f}")
    print(f"Parent Hit Rate @12:         {parent_hit_rate:.1%}")
    if child_hit_rate is not None:
        print(f"Relevant Child Type Hit @12: {child_hit_rate:.1%} (n={len(child_scored)})")
    else:
        print("Relevant Child Type Hit @12: n/a (no questions had expected_chunk_types)")

    print("\nBy category:")
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    for cat, cat_rows in sorted(by_cat.items()):
        cn = len(cat_rows)
        cat_hit_rate = sum(r["hit_at_12"] for r in cat_rows) / cn
        print(f"  {cat:12} n={cn:2}  hit@12={cat_hit_rate:.1%}")


if __name__ == "__main__":
    main()
