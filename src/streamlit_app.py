# src/streamlit_app.py
"""
Streamlit chat interface for the World Cup RAG pipeline.

Every question goes through the same rag.answer() used by main.py, so
this UI shares the router, retrieval, and production logging (queries /
feedback tables) with the CLI — no separate code path.

Run with:
    streamlit run src/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag import answer
from monitoring import log_feedback

st.set_page_config(page_title="World Cup RAG", page_icon="⚽", layout="centered")

with st.sidebar:
    st.header("⚽ World Cup RAG")
    st.caption(
        "Ask about any FIFA World Cup match, player, lineup, or final "
        "in the dataset (1930–2026)."
    )
    st.markdown("**Examples**")
    st.markdown(
        "- Who won the 2014 World Cup final?\n"
        "- How many goals did Klose score in 2006?\n"
        "- How many times has Brazil won the World Cup?\n"
        "- Which year did Argentina win the World Cup?"
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

st.title("FIFA World Cup Q&A")

if "messages" not in st.session_state:
    st.session_state.messages = []  # each: role, content, query_id, meta, feedback, show_comment


def render_sources(meta: dict | None) -> None:
    if not meta or meta.get("route") != "rag" or not meta.get("contexts"):
        return
    with st.expander(f"Show sources ({len(meta['contexts'])} retrieved)"):
        for i, c in enumerate(meta["contexts"], 1):
            st.markdown(
                f"**{i}. [{c['chunk_type']}]** score={c['score']:.3f} "
                f"· year={c.get('year')} · `{c.get('match_id')}`"
            )
            preview = c["text"][:280]
            st.text(preview + ("..." if len(c["text"]) > 280 else ""))


def render_feedback(msg: dict, idx: int) -> None:
    if msg.get("feedback") is not None:
        st.caption("Feedback recorded: " + ("👍" if msg["feedback"] else "👎"))
        return

    col1, col2, _ = st.columns([1, 1, 6])
    with col1:
        if st.button("👍", key=f"up_{idx}"):
            log_feedback(msg["query_id"], True)
            msg["feedback"] = True
            st.rerun()
    with col2:
        if st.button("👎", key=f"down_{idx}"):
            msg["show_comment"] = True

    if msg.get("show_comment"):
        comment = st.text_input(
            "What was wrong? (optional)", key=f"comment_{idx}"
        )
        if st.button("Submit feedback", key=f"submit_{idx}"):
            log_feedback(msg["query_id"], False, comment or None)
            msg["feedback"] = False
            msg["show_comment"] = False
            st.rerun()


def route_caption(meta: dict | None) -> str | None:
    route = (meta or {}).get("route")
    if route == "router":
        return "⚡ Answered by SQL router (exact/aggregate lookup)"
    if route == "rag":
        return "🔎 Answered via retrieval + LLM"
    return None


# --- Render chat history ---
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            caption = route_caption(msg.get("meta"))
            if caption:
                st.caption(caption)
            render_sources(msg.get("meta"))
            render_feedback(msg, idx)

# --- New question ---
question = st.chat_input("Ask a question about the World Cup...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Thinking..."):
        text, query_id, meta = answer(question, top_k=12)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": text,
            "query_id": query_id,
            "meta": meta,
            "feedback": None,
            "show_comment": False,
        }
    )
    st.rerun()
