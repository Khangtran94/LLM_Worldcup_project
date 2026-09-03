# src/rag.py
"""Main RAG pipeline: router → retrieve → prompt → LLM.

Every call to answer() now logs itself to the `queries` table via
monitoring.log_query() and returns the new query_id alongside the
answer text, so callers can attach feedback afterwards with
monitoring.log_feedback(query_id, ...).

NOTE: this changes answer()'s return type from `str` to `tuple[str, int]`.
The only caller in this codebase is main.py, updated to match.
run_eval.py deliberately does NOT call answer() — it drives retrieve()
and the OpenAI client directly, so eval runs never hit this logging path
and never mix into production query history.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prompt import build_messages
from retrieve import retrieve
from query_router import try_direct_answer
from monitoring import log_query

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"


def answer(question: str, top_k: int = 12, debug: bool = False) -> tuple[str, int, dict]:
    """
    Full RAG flow.

    Step 0: try the deterministic router (SQL aggregates / exact lookups).
    If it can answer confidently, skip retrieval + LLM entirely.

    Returns (answer_text, query_id, meta).
      query_id: the row just inserted into `queries` — pass to
                monitoring.log_feedback() to record a thumbs up/down.
      meta:     {"route": "router" | "rag", "contexts": list[dict] | None}
                contexts is None for router answers (no retrieval ran);
                for rag answers it's the retrieved chunks (text,
                chunk_type, match_id, year, is_final, score), useful for
                a "show sources" UI panel.
    Set debug=True to print retrieved chunks / router decisions.
    """
    start = time.perf_counter()

    direct = try_direct_answer(question)
    if direct is not None:
        if debug:
            print("\n=== Router: answered directly, RAG skipped ===")
            print(direct)
            print("===============================================\n")

        latency_ms = int((time.perf_counter() - start) * 1000)
        query_id = log_query(
            question=question,
            response=direct,
            model="router",
            latency_ms=latency_ms,
            retrieved_ids=None,
        )
        return direct, query_id, {"route": "router", "contexts": None}

    contexts = retrieve(question, top_k=top_k)

    if debug:
        print("\n=== Retrieved chunks ===")
        for i, c in enumerate(contexts, 1):
            preview = c["text"][:130].replace("\n", " ")
            print(
                f"{i:2}. [{c['chunk_type']:12}] score={c['score']:.3f} "
                f"year={c.get('year')} final={c.get('is_final')} | {preview}..."
            )
        print("========================\n")

    messages = build_messages(question, contexts)

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,
    )

    text = response.choices[0].message.content.strip()
    latency_ms = int((time.perf_counter() - start) * 1000)

    query_id = log_query(
        question=question,
        response=text,
        model=MODEL,
        latency_ms=latency_ms,
        retrieved_ids=[c["match_id"] for c in contexts],
    )

    return text, query_id, {"route": "rag", "contexts": contexts}
