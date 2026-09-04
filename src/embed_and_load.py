# src/embed_and_load.py
"""
Embed match chunks and load them into Postgres (pgvector).

Reads:  data/processed/match_chunks.jsonl
Writes: public.chunks table

Model: sentence-transformers/all-mpnet-base-v2  (768 dims)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import duckdb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Allow `python src/embed_and_load.py` from project root
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db.connection import get_connection


PROJECT_ROOT = SRC_DIR.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "match_chunks.jsonl"
DEFAULT_DUCKDB_PATH = PROJECT_ROOT / "data" / "processed" / "worldcup_ingest.duckdb"
DEFAULT_DATASET = "worldcup_staging"
DEFAULT_TABLE = "match_chunks"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 64

def load_chunks_from_duckdb(
    db_path: Path,
    dataset_name: str = DEFAULT_DATASET,
    table_name: str = DEFAULT_TABLE,
) -> list[dict[str, Any]]:
    """
    Read chunks landed by the dlt ingestion pipeline (src/ingestion/pipeline.py).

    dlt flattens the nested "metadata" dict into metadata__<field> columns
    on the main table, and normalizes list fields (teams, players,
    goal_scorers) into separate child tables linked by _dlt_id ->
    _dlt_parent_id. This reassembles both back into the original
    {"id", "text", "metadata": {...}} shape row_from_chunk() expects.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        # Child tables use dlt's default child-table column names:
        # _dlt_parent_id (FK back to match_chunks._dlt_id) and "value"
        # (the actual list element). list_aggregate + GROUP BY rebuilds
        # each row's array.
        list_fields = {
            "teams": f"{table_name}__metadata__teams",
            "players": f"{table_name}__metadata__players",
            "goal_scorers": f"{table_name}__metadata__goal_scorers",
        }

        list_aggs = ",\n            ".join(
            f"""(
                SELECT list(value ORDER BY _dlt_list_idx)
                FROM {dataset_name}.{child_table} AS c
                WHERE c._dlt_parent_id = m._dlt_id
            ) AS {field}"""
            for field, child_table in list_fields.items()
        )

        query = f"""
            SELECT m.*,
            {list_aggs}
            FROM {dataset_name}.{table_name} AS m
        """
        df = con.sql(query).fetchdf()
    finally:
        con.close()

    non_metadata_cols = {"id", "text", "_dlt_load_id", "_dlt_id"}
    list_field_names = set(list_fields.keys())

    chunks: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        metadata: dict[str, Any] = {}

        for col, value in row.items():
            if col in non_metadata_cols or col in list_field_names:
                continue
            if col.startswith("metadata__"):
                key = col[len("metadata__"):]
                if value is not None and not (isinstance(value, float) and pd.isna(value)):
                    metadata[key] = value

        for field in list_field_names:
            value = row.get(field)
            if value is not None and len(value) > 0:
                metadata[field] = list(value)

        chunks.append(
            {
                "id": row["id"],
                "text": row.get("text") or "",
                "metadata": metadata,
            }
        )
    return chunks

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc
    return records


def parse_date(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def to_vector_literal(embedding: list[float]) -> str:
    """Format embedding as a pgvector literal: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"


def row_from_chunk(chunk: dict[str, Any], embedding: list[float]) -> dict[str, Any]:
    meta = chunk.get("metadata") or {}

    return {
        "id": chunk["id"],
        "match_id": meta.get("match_id"),
        "chunk_type": meta.get("chunk_type"),
        "chunk_index": meta.get("chunk_index", 0),
        "parent_id": meta.get("parent_id"),
        "text": chunk.get("text") or "",
        "embedding": to_vector_literal(embedding),
        "year": meta.get("year"),
        "tournament": meta.get("tournament"),
        "round": meta.get("round"),
        "date": parse_date(meta.get("date")),
        "team1": meta.get("team1"),
        "team2": meta.get("team2"),
        "teams": meta.get("teams") or None,
        "ground": meta.get("ground"),
        "winner": meta.get("winner"),
        "is_draw": meta.get("is_draw"),
        "is_final": meta.get("is_final"),
        "went_to_extra_time": meta.get("went_to_extra_time"),
        "had_penalties": meta.get("had_penalties"),
        "team": meta.get("team"),
        "players": meta.get("players") or None,
        "goal_scorers": meta.get("goal_scorers") or None,
        "final_score": meta.get("final_score"),
        "metadata": json.dumps(meta, ensure_ascii=False),
    }


UPSERT_SQL = """
INSERT INTO chunks (
    id, match_id, chunk_type, chunk_index, parent_id, text, embedding,
    year, tournament, round, date, team1, team2, teams, ground, winner,
    is_draw, is_final, went_to_extra_time, had_penalties,
    team, players, goal_scorers, final_score, metadata
) VALUES (
    %(id)s, %(match_id)s, %(chunk_type)s, %(chunk_index)s, %(parent_id)s,
    %(text)s, %(embedding)s::vector,
    %(year)s, %(tournament)s, %(round)s, %(date)s, %(team1)s, %(team2)s,
    %(teams)s, %(ground)s, %(winner)s,
    %(is_draw)s, %(is_final)s, %(went_to_extra_time)s, %(had_penalties)s,
    %(team)s, %(players)s, %(goal_scorers)s, %(final_score)s,
    %(metadata)s::jsonb
)
ON CONFLICT (id) DO UPDATE SET
    match_id = EXCLUDED.match_id,
    chunk_type = EXCLUDED.chunk_type,
    chunk_index = EXCLUDED.chunk_index,
    parent_id = EXCLUDED.parent_id,
    text = EXCLUDED.text,
    embedding = EXCLUDED.embedding,
    year = EXCLUDED.year,
    tournament = EXCLUDED.tournament,
    round = EXCLUDED.round,
    date = EXCLUDED.date,
    team1 = EXCLUDED.team1,
    team2 = EXCLUDED.team2,
    teams = EXCLUDED.teams,
    ground = EXCLUDED.ground,
    winner = EXCLUDED.winner,
    is_draw = EXCLUDED.is_draw,
    is_final = EXCLUDED.is_final,
    went_to_extra_time = EXCLUDED.went_to_extra_time,
    had_penalties = EXCLUDED.had_penalties,
    team = EXCLUDED.team,
    players = EXCLUDED.players,
    goal_scorers = EXCLUDED.goal_scorers,
    final_score = EXCLUDED.final_score,
    metadata = EXCLUDED.metadata
"""


def embed_batches(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[start : start + batch_size]
        emb = model.encode(
            batch,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vectors.extend(emb.tolist())
    return vectors


def load_into_postgres(rows: list[dict[str, Any]]) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            for row in tqdm(rows, desc="Inserting"):
                cur.execute(UPSERT_SQL, row)
        conn.commit()


# def main() -> None:
#     parser = argparse.ArgumentParser(
#         description="Embed match chunks and load them into Postgres."
#     )
#     parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
#     parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
#     parser.add_argument(
#         "--limit",
#         type=int,
#         default=None,
#         help="Optional: only process the first N chunks (for testing).",
#     )
#     args = parser.parse_args()

#     if not args.input.exists():
#         print(f"Input file not found: {args.input}", file=sys.stderr)
#         sys.exit(1)

#     print(f"Loading chunks from {args.input}")
#     chunks = load_jsonl(args.input)
#     if args.limit is not None:
#         chunks = chunks[: args.limit]
#     print(f"Chunks to process: {len(chunks):,}")

#     print(f"Loading model: {MODEL_NAME}")
#     model = SentenceTransformer(MODEL_NAME)

#     texts = [c.get("text") or "" for c in chunks]
#     embeddings = embed_batches(model, texts, args.batch_size)

#     rows = [
#         row_from_chunk(chunk, emb)
#         for chunk, emb in zip(chunks, embeddings, strict=True)
#     ]

#     print("Writing to Postgres...")
#     load_into_postgres(rows)

#     with get_connection() as conn:
#         with conn.cursor() as cur:
#             cur.execute("SELECT COUNT(*) FROM chunks")
#             total = cur.fetchone()[0]
#             cur.execute(
#                 "SELECT chunk_type, COUNT(*) FROM chunks GROUP BY chunk_type ORDER BY COUNT(*) DESC"
#             )
#             by_type = cur.fetchall()

#     print(f"\nDone. Total rows in chunks: {total:,}")
#     print("By chunk_type:")
#     for chunk_type, count in by_type:
#         print(f"  {chunk_type:15} {count:,}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed match chunks and load them into Postgres."
    )
    parser.add_argument("--duckdb-path", type=Path, default=DEFAULT_DUCKDB_PATH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: only process the first N chunks (for testing).",
    )
    args = parser.parse_args()

    if not args.duckdb_path.exists():
        print(f"DuckDB file not found: {args.duckdb_path}", file=sys.stderr)
        print("Run src/ingestion/pipeline.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading chunks from {args.duckdb_path} ({DEFAULT_DATASET}.{DEFAULT_TABLE})")
    chunks = load_chunks_from_duckdb(args.duckdb_path)
    if args.limit is not None:
        chunks = chunks[: args.limit]
    print(f"Chunks to process: {len(chunks):,}")

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [c.get("text") or "" for c in chunks]
    embeddings = embed_batches(model, texts, args.batch_size)

    rows = [
        row_from_chunk(chunk, emb)
        for chunk, emb in zip(chunks, embeddings, strict=True)
    ]

    print("Writing to Postgres...")
    load_into_postgres(rows)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            total = cur.fetchone()[0]
            cur.execute(
                "SELECT chunk_type, COUNT(*) FROM chunks GROUP BY chunk_type ORDER BY COUNT(*) DESC"
            )
            by_type = cur.fetchall()

    print(f"\nDone. Total rows in chunks: {total:,}")
    print("By chunk_type:")
    for chunk_type, count in by_type:
        print(f"  {chunk_type:15} {count:,}")

if __name__ == "__main__":
    main()
