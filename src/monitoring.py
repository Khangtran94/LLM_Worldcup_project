# src/monitoring.py
"""
Production logging for the World Cup RAG pipeline.

Writes to the `queries` and `feedback` tables (already defined in
schema.sql, previously unused). This is separate from the eval harness
(`eval_questions` / `eval_results`) on purpose: this module logs real
user interactions from `rag.answer()`, the eval harness logs test-set
runs. Keeping them apart means real traffic never pollutes eval history
and eval runs never pollute production logs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db.connection import get_connection


INSERT_QUERY_SQL = """
INSERT INTO queries (
    question, rewritten_question, retrieved_ids, response, model, latency_ms
) VALUES (
    %(question)s, %(rewritten_question)s, %(retrieved_ids)s,
    %(response)s, %(model)s, %(latency_ms)s
)
RETURNING id
"""

INSERT_FEEDBACK_SQL = """
INSERT INTO feedback (query_id, is_positive, comment)
VALUES (%(query_id)s, %(is_positive)s, %(comment)s)
"""


def log_query(
    *,
    question: str,
    response: str,
    model: str,
    latency_ms: int,
    retrieved_ids: list[str] | None = None,
    rewritten_question: str | None = None,
) -> int:
    """
    Log one answered question. Returns the new queries.id, which the
    caller should hold onto to attach feedback later.

    retrieved_ids: match_ids of the chunks that were retrieved and fed
    to the LLM. Pass None when the router answered directly (no
    retrieval happened).
    """
    params = {
        "question": question,
        "rewritten_question": rewritten_question,
        "retrieved_ids": retrieved_ids,
        "response": response,
        "model": model,
        "latency_ms": latency_ms,
    }
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(INSERT_QUERY_SQL, params)
            query_id = cur.fetchone()[0]
        conn.commit()
    return query_id


def log_feedback(query_id: int, is_positive: bool, comment: str | None = None) -> None:
    """Attach a thumbs up/down (and optional comment) to a logged query."""
    params: dict[str, Any] = {
        "query_id": query_id,
        "is_positive": is_positive,
        "comment": comment,
    }
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(INSERT_FEEDBACK_SQL, params)
        conn.commit()
