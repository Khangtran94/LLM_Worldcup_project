# src/ingestion/pipeline.py
"""
Run the dlt ingestion pipeline: download -> transform -> chunk.

Lands rows in a local duckdb file (data/processed/worldcup_staging.duckdb,
dataset "worldcup_staging", table "match_chunks"). embed_and_load.py reads
from there to embed + upsert into Postgres — that step is intentionally
NOT part of this pipeline.

Usage:
    python src/ingestion/pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import dlt

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ingestion.resources import match_chunks

PROJECT_ROOT = SRC_DIR.parent
DUCKDB_PATH = PROJECT_ROOT / "data" / "processed" / "worldcup_ingest.duckdb"

def run() -> None:
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)

    pipeline = dlt.pipeline(
        pipeline_name="worldcup_ingest",
        destination=dlt.destinations.duckdb(str(DUCKDB_PATH)),
        dataset_name="worldcup_staging",
    )

    info = pipeline.run(match_chunks)
    print(info)

    with pipeline.sql_client() as client:
        row = client.execute_sql(
            "SELECT COUNT(*) FROM worldcup_staging.match_chunks"
        )
        print(f"Rows in match_chunks: {row[0][0]:,}")


if __name__ == "__main__":
    run()